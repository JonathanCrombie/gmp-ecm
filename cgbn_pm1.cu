/* Batched P-1 exponentiation and P+1 Lucas ladders. GPL-3.0-or-later.
   One cooperative group per modulus; one common, arbitrarily long exponent. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>
#include <gmp.h>
/* Upstream CGBN defines these non-inline host helpers in cgbn.h. Give this
   translation unit its own names so it can link beside the ECM kernels. */
#define cgbn_error_report_alloc pm1_cgbn_error_report_alloc
#define cgbn_error_report_free pm1_cgbn_error_report_free
#define cgbn_error_report_check pm1_cgbn_error_report_check
#define cgbn_error_report_reset pm1_cgbn_error_report_reset
#define cgbn_error_string pm1_cgbn_error_string
#include <cgbn.h>
#include "cgbn_pm1.h"
#include "gpu_batch_policy.h"
#include "gpu_load_monitor.h"

template<unsigned threads, unsigned bits> struct pm1_config {
  static const uint32_t TPI=threads, BITS=bits, TPB=256;
  static const uint32_t MAX_ROTATION=4, SHM_LIMIT=0;
  static const bool CONSTANT_TIME=false;
};

template<class P, bool PP1>
__global__ void pm1_power(cgbn_error_report_t *report, uint32_t *data,
                          uint32_t count, const uint32_t *exponent,
                          uint64_t length, uint64_t start, uint64_t steps) {
  uint32_t i=(blockIdx.x*blockDim.x+threadIdx.x)/P::TPI;
  if(i>=count) return;
  typedef cgbn_context_t<P::TPI, P> context_t;
  typedef cgbn_env_t<context_t, P::BITS> env_t;
  typedef cgbn_mem_t<P::BITS> mem_t;
  context_t context(cgbn_report_monitor, report, i);
  env_t env(context);
  typename env_t::cgbn_t n, a, x, y, cross, two;
  mem_t *values=reinterpret_cast<mem_t *>(data)+(PP1 ? 4 : 3)*size_t(i);
  cgbn_load(env, n, values);
  cgbn_load(env, a, values+1);
  cgbn_load(env, x, values+2);
  uint32_t np0=cgbn_bn2mont(env, a, a, n);
  cgbn_bn2mont(env, x, x, n);
  if(PP1) {
    cgbn_load(env, y, values+3);
    cgbn_bn2mont(env, y, y, n);
    cgbn_set_ui32(env, two, 2);
    cgbn_bn2mont(env, two, two, n);
  }
  for(uint64_t b=start; b<start+steps; ++b) {
    uint64_t bit=length-1-b;
    bool odd=(exponent[bit/32]>>(bit%32))&1;
    if(PP1) {
      /* (V_k,V_(k+1)) -> (V_(2k),V_(2k+1)) or (V_(2k+1),V_(2k+2)).
         Reduce before subtraction; adding n after a borrow also works when
         n occupies the full CGBN width (unsigned wrap cancels the borrow). */
      cgbn_mont_mul(env, cross, x, y, n, np0);
      if(cgbn_compare(env, cross, n)>=0) cgbn_sub(env, cross, cross, n);
      if(cgbn_sub(env, cross, cross, a)) cgbn_add(env, cross, cross, n);
      if(odd) {
        cgbn_mont_sqr(env, y, y, n, np0);
        if(cgbn_compare(env, y, n)>=0) cgbn_sub(env, y, y, n);
        if(cgbn_sub(env, y, y, two)) cgbn_add(env, y, y, n);
        cgbn_set(env, x, cross);
      } else {
        cgbn_mont_sqr(env, x, x, n, np0);
        if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
        if(cgbn_sub(env, x, x, two)) cgbn_add(env, x, x, n);
        cgbn_set(env, y, cross);
      }
    } else {
      cgbn_mont_sqr(env, x, x, n, np0);
      /* CGBN Montgomery results need not be fully reduced. */
      if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
      if(odd) {
        cgbn_mont_mul(env, x, x, a, n, np0);
        if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
      }
    }
  }
  cgbn_mont2bn(env, x, x, n, np0);
  if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
  cgbn_store(env, values+2, x);
  if(PP1) {
    cgbn_mont2bn(env, y, y, n, np0);
    if(cgbn_compare(env, y, n)>=0) cgbn_sub(env, y, y, n);
    cgbn_store(env, values+3, y);
  }
}

typedef void (*pm1_kernel)(cgbn_error_report_t *, uint32_t *, uint32_t,
                           const uint32_t *, uint64_t, uint64_t, uint64_t);

static pm1_kernel select_kernel(size_t max_bits, bool pp1, unsigned &width, unsigned &tpi) {
  pm1_kernel kernel=nullptr;
#define PM1_KERNEL(B,T) if(!kernel && max_bits<=B) { \
    width=B; tpi=T; kernel=pp1 ? pm1_power<pm1_config<T,B>,true> : pm1_power<pm1_config<T,B>,false>; }
  PM1_KERNEL(256,4)
  PM1_KERNEL(512,4)
  PM1_KERNEL(768,8)
  PM1_KERNEL(1024,8)
#ifndef IS_DEV_BUILD
  PM1_KERNEL(1536,8)
  PM1_KERNEL(2048,8)
  PM1_KERNEL(3072,16)
  PM1_KERNEL(4096,16)
  PM1_KERNEL(6144,16)
  PM1_KERNEL(8192,16)
  PM1_KERNEL(12288,32)
#endif
#undef PM1_KERNEL
  return kernel;
}

/* NVML reports time busy, not occupied SMs. Admission keeps a batch within
   approximately one resident wave; pacing controls time busy separately.
   All members still consume exactly the same exponent prefix per launch. */
struct cgbn_p1_autotune {
  using clock=std::chrono::steady_clock;
  int device, sms, utilization=-1;
  double duty=ECM_GPU_MAX_DUTY, filtered=-1;
  clock::time_point sampled=clock::now();
  std::mutex mutex;
  ecm_gpu_load_monitor monitor;
#ifdef _WIN32
  /* Ordinary Sleep/sleep_for can round a 5 ms pause up to a whole Windows
     timer tick, needlessly reducing utilization. A private high-resolution
     waitable timer avoids changing the system-wide timer resolution. */
  HANDLE timer=ECM_GPU_TARGET_UTILIZATION<100 ?
    CreateWaitableTimerExA(nullptr, nullptr, 0x00000002,
                          TIMER_MODIFY_STATE | SYNCHRONIZE) : nullptr;
  ~cgbn_p1_autotune() { if(timer) CloseHandle(timer); }
#endif

  void sleep_for(clock::duration duration) {
#ifdef _WIN32
    if(timer) {
      LARGE_INTEGER due;
      due.QuadPart=-std::max<long long>(1,
        (std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count()+99)/100);
      if(SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE) &&
         WaitForSingleObject(timer, INFINITE)==WAIT_OBJECT_0) return;
    }
#endif
    std::this_thread::sleep_for(duration);
  }

  void update() {
    std::lock_guard<std::mutex> lock(mutex);
    auto now=clock::now();
    if(now-sampled<std::chrono::seconds(1)) return;
    sampled=now;
    utilization=monitor.sample();
    if(utilization<0) { duty=ECM_GPU_MAX_DUTY; filtered=-1; return; }
    filtered=filtered<0 ? utilization : (filtered+utilization)/2;
    duty=ecm_gpu_adjust_duty(duty, filtered);
  }
  double current_duty() {
    std::lock_guard<std::mutex> lock(mutex);
    return duty;
  }
};

cgbn_p1_autotune *cgbn_p1_autotune_create(int device) {
  cudaDeviceProp properties;
  cudaError_t error;
  if(device<0 && (error=cudaGetDevice(&device))!=cudaSuccess) {
    fprintf(stderr, "GPU auto batch: %s\n", cudaGetErrorString(error)); return nullptr;
  }
  if((error=cudaSetDevice(device))!=cudaSuccess ||
     (error=cudaGetDeviceProperties(&properties, device))!=cudaSuccess) {
    fprintf(stderr, "GPU auto batch: %s\n", cudaGetErrorString(error)); return nullptr;
  }
  try {
    auto *tuner=new cgbn_p1_autotune;
    tuner->device=device; tuner->sms=properties.multiProcessorCount;
    char bus[32];
    if(cudaDeviceGetPCIBusId(bus, sizeof(bus), device)==cudaSuccess) tuner->monitor.open(bus);
    tuner->utilization=tuner->monitor.sample();
    tuner->duty=ecm_gpu_initial_duty(tuner->utilization);
    return tuner;
  } catch(...) { return nullptr; }
}

void cgbn_p1_autotune_destroy(cgbn_p1_autotune *tuner) { delete tuner; }

int cgbn_p1_plan_batch(cgbn_p1_autotune *tuner, size_t max_bits, int pp1,
                     cgbn_p1_batch_plan *plan) {
  unsigned width=0, tpi=0;
  int blocks=0;
  pm1_kernel kernel=select_kernel(max_bits, pp1!=0, width, tpi);
  if(!tuner || !plan || !kernel) {
    fprintf(stderr, "GPU auto batch: no compiled kernel can hold %zu bits\n", max_bits);
    return -1;
  }
  cudaError_t error=cudaSetDevice(tuner->device);
  if(error==cudaSuccess) error=cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, 256, 0);
  if(error!=cudaSuccess || blocks<=0) {
    fprintf(stderr, "GPU auto batch: cannot determine kernel occupancy: %s\n", cudaGetErrorString(error));
    return -1;
  }
  std::lock_guard<std::mutex> lock(tuner->mutex);
  plan->width=width; plan->tpi=tpi;
  plan->resident_inputs=blocks*tuner->sms*(256/tpi);
  plan->batch_inputs=ecm_gpu_batch_inputs(blocks*tuner->sms, 256/tpi, tuner->duty);
  plan->utilization=tuner->utilization; plan->duty=tuner->duty;
  return 0;
}

static void export_value(uint32_t *out, mpz_srcptr value) {
  size_t written;
  mpz_export(out, &written, -1, sizeof(uint32_t), 0, 0, value);
}

/* Return errors to the caller so a failed CUDA operation cannot produce a
   save record claiming that stage 1 completed. */
static void progress_message(const cgbn_pm1_callbacks *callbacks, const char *format, ...) {
  char message[256];
  va_list args;
  va_start(args, format);
  vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  if(callbacks && callbacks->progress) callbacks->progress(callbacks->arg, message);
  else { fputs(message, stdout); fflush(stdout); }
}

static int p1_stage1_run(mpz_ptr *residues, mpz_ptr *next,
                   mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   const cgbn_pm1_callbacks *callbacks, cgbn_p1_autotune *tuner=nullptr) {
  using clock=std::chrono::steady_clock;
  uint32_t *gpu_data=nullptr, *gpu_exponent=nullptr;
  cgbn_error_report_t *report=nullptr;
  cudaEvent_t begin=nullptr, end=nullptr;
  int result=-1;
  const char *method=next ? "P+1" : "P-1";
  const size_t stride=next ? 4 : 3;
  try {
    if(!count || mpz_sgn(exponent)<=0) return -1;
    const uint64_t length=mpz_sizeinbase(exponent, 2);
    if(*bits_done>length) return -1;
    size_t max_bits=0;
    for(uint32_t i=0; i<count; ++i) {
      if(mpz_cmp_ui(moduli[i], 1)<=0 || !mpz_odd_p(moduli[i]) ||
         mpz_sgn(bases[i])<0 || mpz_cmp(bases[i], moduli[i])>=0 ||
         mpz_sgn(residues[i])<0 || mpz_cmp(residues[i], moduli[i])>=0 ||
         (next && (mpz_sgn(next[i])<0 || mpz_cmp(next[i], moduli[i])>=0))) {
        fprintf(stderr, "GPU %s: invalid modulus, base or residue at input %u\n", method, i+1);
        return -1;
      }
      max_bits=std::max(max_bits, mpz_sizeinbase(moduli[i], 2));
    }
    unsigned width=0, tpi=0;
    pm1_kernel kernel=select_kernel(max_bits, next!=nullptr, width, tpi);
    if(!kernel) {
      fprintf(stderr, "GPU %s: no compiled kernel can hold %zu bits\n", method, max_bits);
      return -1;
    }
    const size_t limbs=width/32;
    if(count>SIZE_MAX/(stride*limbs*sizeof(uint32_t))) return -1;
    std::vector<uint32_t> data(size_t(count)*stride*limbs, 0);
    std::vector<uint32_t> powers((length+31)/32, 0);
    export_value(powers.data(), exponent);
    for(uint32_t i=0; i<count; ++i) {
      export_value(data.data()+(stride*i)*limbs, moduli[i]);
      export_value(data.data()+(stride*i+1)*limbs, bases[i]);
      export_value(data.data()+(stride*i+2)*limbs, residues[i]);
      if(next) export_value(data.data()+(stride*i+3)*limbs, next[i]);
    }
#define PM1_CUDA(call) do { cudaError_t err=(call); if(err!=cudaSuccess) { \
      fprintf(stderr, "GPU %s: %s: %s\n", method, #call, cudaGetErrorString(err)); \
      throw 1; } } while(0)
    if(tuner) device=tuner->device;
    if(device>=0) PM1_CUDA(cudaSetDevice(device));
    PM1_CUDA(cudaMalloc((void **)&gpu_data, data.size()*sizeof(uint32_t)));
    PM1_CUDA(cudaMalloc((void **)&gpu_exponent, powers.size()*sizeof(uint32_t)));
    PM1_CUDA(cudaMemcpy(gpu_data, data.data(), data.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
    PM1_CUDA(cudaMemcpy(gpu_exponent, powers.data(), powers.size()*sizeof(uint32_t), cudaMemcpyHostToDevice));
    PM1_CUDA(cgbn_error_report_alloc(&report));
    PM1_CUDA(cudaEventCreate(&begin));
    PM1_CUDA(cudaEventCreate(&end));
    unsigned ipb=256/tpi;
    size_t blocks=(size_t(count)+ipb-1)/ipb;
    if(verbose>=1)
      progress_message(callbacks, "GPU %s: %u numbers, largest %zu bits, CGBN<%u,%u>, %llu exponent bits\n",
             method, count, max_bits, width, tpi, (unsigned long long)length);
    auto snapshot=[&]() {
      PM1_CUDA(cudaMemcpy(data.data(), gpu_data, data.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
      for(uint32_t i=0; i<count; ++i) {
        mpz_import(residues[i], limbs, -1, sizeof(uint32_t), 0, 0,
                   data.data()+(stride*i+2)*limbs);
        if(next) mpz_import(next[i], limbs, -1, sizeof(uint32_t), 0, 0,
                           data.data()+(stride*i+3)*limbs);
      }
      if(callbacks && callbacks->checkpoint &&
         !callbacks->checkpoint(callbacks->arg, *bits_done)) throw 1;
    };
    uint64_t steps=128;
    *milliseconds=0;
    auto started=clock::now(), last_save=started, last_progress=started;
    while(*bits_done<length && !(callbacks && callbacks->stop && callbacks->stop(callbacks->arg))) {
      steps=std::min(steps, length-*bits_done);
      auto launch_started=clock::now();
      PM1_CUDA(cudaEventRecord(begin));
      kernel<<<blocks,256>>>(report, gpu_data, count, gpu_exponent, length, *bits_done, steps);
      PM1_CUDA(cudaGetLastError());
      PM1_CUDA(cudaEventRecord(end));
      PM1_CUDA(cudaEventSynchronize(end));
      if(cgbn_error_report_check(report)) {
        fprintf(stderr, "GPU %s: %s (instance %u)\n", method, cgbn_error_string(report), report->_instance);
        throw 1;
      }
      *bits_done+=steps;
      float elapsed;
      PM1_CUDA(cudaEventElapsedTime(&elapsed, begin, end));
      *milliseconds+=elapsed;
      if(tuner) {
        tuner->update();
        /* Use actual cycle time so host/driver overhead also counts as idle
           time. Sleep in small pieces to retain prompt Ctrl-C checkpoints. */
        double duty=tuner->current_duty();
        double cycle=duty<1.0 ? elapsed/1000.0/duty : 0.0;
        auto deadline=launch_started+std::chrono::duration_cast<clock::duration>(
          std::chrono::duration<double>(cycle));
        while(clock::now()<deadline &&
              !(callbacks && callbacks->stop && callbacks->stop(callbacks->arg))) {
          auto remaining=deadline-clock::now();
          if(remaining>clock::duration::zero())
            tuner->sleep_for(std::min(remaining,
              std::chrono::duration_cast<clock::duration>(std::chrono::milliseconds(10))));
        }
      }
      auto now=clock::now();
      if(verbose>=2 && now-last_progress>=std::chrono::seconds(5)) {
        progress_message(callbacks, "GPU %s: %.1f%% of exponent, %.3f seconds GPU time\n",
               method, 100.0*(*bits_done)/length, *milliseconds/1000.0);
        last_progress=now;
        if(tuner) {
          std::lock_guard<std::mutex> lock(tuner->mutex);
          progress_message(callbacks, "GPU auto batch: load %d%% (-1 = unavailable), duty %.1f%%, target %d%%\n",
            tuner->utilization, 100*tuner->duty, ECM_GPU_TARGET_UTILIZATION);
        }
      }
      if(callbacks && callbacks->checkpoint && now-last_save>=std::chrono::minutes(10)) {
        snapshot();
        last_save=now;
      }
      /* Short launches keep display GPUs responsive and bound stop latency. */
      if(elapsed<(tuner ? 40 : 80)) steps=std::min<uint64_t>(65536, steps+steps/4+1);
      else if(elapsed>(tuner ? 60 : 120)) steps=std::max<uint64_t>(1, steps*3/4);
    }
    snapshot();
    if(verbose>=1)
      progress_message(callbacks, "GPU %s: %s stage 1 in %.3f seconds (%.2f numbers/second)\n",
             method, *bits_done==length ? "completed" : "interrupted", *milliseconds/1000.0,
             *milliseconds>0 ? 1000.0*count/(*milliseconds) : 0.0);
    if(tuner && verbose>=1) {
      double wall=std::chrono::duration<double>(clock::now()-started).count();
      progress_message(callbacks, "GPU auto batch: %.3f seconds wall time including pacing (%.2f numbers/second)\n",
                       wall, wall>0 ? count/wall : 0.0);
    }
    result=0;
  } catch(const std::exception &e) {
    fprintf(stderr, "GPU %s: %s\n", method, e.what());
  } catch(...) {}
  if(end) cudaEventDestroy(end);
  if(begin) cudaEventDestroy(begin);
  if(report) cgbn_error_report_free(report);
  if(gpu_exponent) cudaFree(gpu_exponent);
  if(gpu_data) cudaFree(gpu_data);
  return result;
}

int cgbn_p1_stage1_adaptive(mpz_ptr *residues, mpz_ptr *next,
                   mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   const cgbn_pm1_callbacks *callbacks, cgbn_p1_autotune *tuner) {
  return p1_stage1_run(residues, next, bases, moduli, count, exponent, bits_done,
                      device, milliseconds, verbose, callbacks, tuner);
}

int cgbn_pm1_stage1_run(mpz_ptr *residues, mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   const cgbn_pm1_callbacks *callbacks) {
  return p1_stage1_run(residues, nullptr, bases, moduli, count, exponent, bits_done,
                      device, milliseconds, verbose, callbacks);
}

int cgbn_pp1_stage1_run(mpz_ptr *residues, mpz_ptr *next,
                   mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   const cgbn_pm1_callbacks *callbacks) {
  if(!next) return -1;
  return p1_stage1_run(residues, next, bases, moduli, count, exponent, bits_done,
                      device, milliseconds, verbose, callbacks);
}

/* Preserve the synchronous library API and its original callback contract. */
struct legacy_callbacks {
  int (*stop)(void);
  int (*checkpoint)(void *, uint64_t);
  void *arg;
};
static int legacy_stop(void *arg) {
  auto *c=static_cast<legacy_callbacks *>(arg);
  return c->stop && c->stop();
}
static int legacy_checkpoint(void *arg, uint64_t bits) {
  auto *c=static_cast<legacy_callbacks *>(arg);
  return !c->checkpoint || c->checkpoint(c->arg, bits);
}
int cgbn_pm1_stage1(mpz_ptr *residues, mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   int (*stop_asap)(void),
                   int (*checkpoint)(void *, uint64_t), void *checkpoint_arg) {
  legacy_callbacks legacy{stop_asap, checkpoint, checkpoint_arg};
  cgbn_pm1_callbacks callbacks{legacy_stop, checkpoint ? legacy_checkpoint : nullptr, nullptr, &legacy};
  return cgbn_pm1_stage1_run(residues, bases, moduli, count, exponent, bits_done,
                           device, milliseconds, verbose, &callbacks);
}
