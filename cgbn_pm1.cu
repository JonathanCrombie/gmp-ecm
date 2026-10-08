/* Batched P-1 modular exponentiation. GPL-3.0-or-later.
   One cooperative group per modulus; one common, arbitrarily long exponent. */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
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

template<unsigned threads, unsigned bits> struct pm1_config {
  static const uint32_t TPI=threads, BITS=bits, TPB=256;
  static const uint32_t MAX_ROTATION=4, SHM_LIMIT=0;
  static const bool CONSTANT_TIME=false;
};

template<class P>
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
  typename env_t::cgbn_t n, a, x;
  mem_t *values=reinterpret_cast<mem_t *>(data)+3*i;
  cgbn_load(env, n, values);
  cgbn_load(env, a, values+1);
  cgbn_load(env, x, values+2);
  uint32_t np0=cgbn_bn2mont(env, a, a, n);
  cgbn_bn2mont(env, x, x, n);
  for(uint64_t b=start; b<start+steps; ++b) {
    cgbn_mont_sqr(env, x, x, n, np0);
    /* CGBN Montgomery results need not be fully reduced for small moduli.
       Keep every operand < n, including when n uses the entire width. */
    if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
    uint64_t bit=length-1-b;
    if((exponent[bit/32]>>(bit%32))&1) {
      cgbn_mont_mul(env, x, x, a, n, np0);
      if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
    }
  }
  cgbn_mont2bn(env, x, x, n, np0);
  if(cgbn_compare(env, x, n)>=0) cgbn_sub(env, x, x, n);
  cgbn_store(env, values+2, x);
}

typedef void (*pm1_kernel)(cgbn_error_report_t *, uint32_t *, uint32_t,
                           const uint32_t *, uint64_t, uint64_t, uint64_t);

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

int cgbn_pm1_stage1_run(mpz_ptr *residues, mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   const cgbn_pm1_callbacks *callbacks) {
  using clock=std::chrono::steady_clock;
  uint32_t *gpu_data=nullptr, *gpu_exponent=nullptr;
  cgbn_error_report_t *report=nullptr;
  cudaEvent_t begin=nullptr, end=nullptr;
  int result=-1;
  try {
    if(!count || mpz_sgn(exponent)<=0) return -1;
    const uint64_t length=mpz_sizeinbase(exponent, 2);
    if(*bits_done>length) return -1;
    size_t max_bits=0;
    for(uint32_t i=0; i<count; ++i) {
      if(mpz_cmp_ui(moduli[i], 1)<=0 || !mpz_odd_p(moduli[i]) ||
         mpz_sgn(bases[i])<0 || mpz_cmp(bases[i], moduli[i])>=0 ||
         mpz_sgn(residues[i])<0 || mpz_cmp(residues[i], moduli[i])>=0) {
        fprintf(stderr, "GPU P-1: invalid modulus, base or residue at input %u\n", i+1);
        return -1;
      }
      max_bits=std::max(max_bits, mpz_sizeinbase(moduli[i], 2));
    }
    unsigned width=0, tpi=0;
    pm1_kernel kernel=nullptr;
#define PM1_KERNEL(B,T) if(!kernel && max_bits<=B) { \
      width=B; tpi=T; kernel=pm1_power<pm1_config<T,B>>; }
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
    if(!kernel) {
      fprintf(stderr, "GPU P-1: no compiled kernel can hold %zu bits\n", max_bits);
      return -1;
    }
    const size_t limbs=width/32;
    if(count>SIZE_MAX/(3*limbs*sizeof(uint32_t))) return -1;
    std::vector<uint32_t> data(size_t(count)*3*limbs, 0);
    std::vector<uint32_t> powers((length+31)/32, 0);
    export_value(powers.data(), exponent);
    for(uint32_t i=0; i<count; ++i) {
      export_value(data.data()+(3*size_t(i))*limbs, moduli[i]);
      export_value(data.data()+(3*size_t(i)+1)*limbs, bases[i]);
      export_value(data.data()+(3*size_t(i)+2)*limbs, residues[i]);
    }
#define PM1_CUDA(call) do { cudaError_t err=(call); if(err!=cudaSuccess) { \
      fprintf(stderr, "GPU P-1: %s: %s\n", #call, cudaGetErrorString(err)); \
      throw 1; } } while(0)
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
      progress_message(callbacks, "GPU P-1: %u numbers, largest %zu bits, CGBN<%u,%u>, %llu exponent bits\n",
             count, max_bits, width, tpi, (unsigned long long)length);
    auto snapshot=[&]() {
      PM1_CUDA(cudaMemcpy(data.data(), gpu_data, data.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost));
      for(uint32_t i=0; i<count; ++i)
        mpz_import(residues[i], limbs, -1, sizeof(uint32_t), 0, 0,
                   data.data()+(3*size_t(i)+2)*limbs);
      if(callbacks && callbacks->checkpoint &&
         !callbacks->checkpoint(callbacks->arg, *bits_done)) throw 1;
    };
    uint64_t steps=128;
    *milliseconds=0;
    auto last_save=clock::now(), last_progress=last_save;
    while(*bits_done<length && !(callbacks && callbacks->stop && callbacks->stop(callbacks->arg))) {
      steps=std::min(steps, length-*bits_done);
      PM1_CUDA(cudaEventRecord(begin));
      kernel<<<blocks,256>>>(report, gpu_data, count, gpu_exponent, length, *bits_done, steps);
      PM1_CUDA(cudaGetLastError());
      PM1_CUDA(cudaEventRecord(end));
      PM1_CUDA(cudaEventSynchronize(end));
      if(cgbn_error_report_check(report)) {
        fprintf(stderr, "GPU P-1: %s (instance %u)\n", cgbn_error_string(report), report->_instance);
        throw 1;
      }
      *bits_done+=steps;
      float elapsed;
      PM1_CUDA(cudaEventElapsedTime(&elapsed, begin, end));
      *milliseconds+=elapsed;
      auto now=clock::now();
      if(verbose>=2 && now-last_progress>=std::chrono::seconds(5)) {
        progress_message(callbacks, "GPU P-1: %.1f%% of exponent, %.3f seconds GPU time\n",
               100.0*(*bits_done)/length, *milliseconds/1000.0);
        last_progress=now;
      }
      if(callbacks && callbacks->checkpoint && now-last_save>=std::chrono::minutes(10)) {
        snapshot();
        last_save=now;
      }
      /* Short launches keep display GPUs responsive and bound stop latency. */
      if(elapsed<80) steps=std::min<uint64_t>(65536, steps+steps/4+1);
      else if(elapsed>120) steps=std::max<uint64_t>(1, steps*3/4);
    }
    snapshot();
    if(verbose>=1)
      progress_message(callbacks, "GPU P-1: %s stage 1 in %.3f seconds (%.2f numbers/second)\n",
             *bits_done==length ? "completed" : "interrupted", *milliseconds/1000.0,
             *milliseconds>0 ? 1000.0*count/(*milliseconds) : 0.0);
    result=0;
  } catch(const std::exception &e) {
    fprintf(stderr, "GPU P-1: %s\n", e.what());
  } catch(...) {}
  if(end) cudaEventDestroy(end);
  if(begin) cudaEventDestroy(begin);
  if(report) cgbn_error_report_free(report);
  if(gpu_exponent) cudaFree(gpu_exponent);
  if(gpu_data) cudaFree(gpu_data);
  return result;
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
