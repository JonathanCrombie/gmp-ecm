/* P-1 across different input numbers. GPL-3.0-or-later.
   Pipeline GPU stage 1 with ordinary CPU P-1 stage 2 and CLI reporting. */
#include "ecm-impl.h"
#include "pm1_gpu.h"
#ifdef WITH_GPU
#include "cgbn_pm1.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

typedef struct {
  mpcandi_t n;
  mpz_t x, x0, factor;
  double start;
  pm1_gpu_state gpu;
  int status;
} pm1_job;

static void job_init(pm1_job *j) {
  mpcandi_t_init(&j->n);
  mpz_init(j->x); mpz_init(j->x0); mpz_init(j->factor);
  mpz_init(j->gpu.base);
  j->start=1; j->gpu.present=0; j->gpu.bits=0;
  j->gpu.target=0; j->status=ECM_NO_FACTOR_FOUND;
}

static void job_clear(pm1_job *j) {
  mpcandi_t_free(&j->n);
  mpz_clear(j->x); mpz_clear(j->x0); mpz_clear(j->factor);
  mpz_clear(j->gpu.base);
}

/* 1 = candidate, 0 = EOF, -1 = malformed/unsupported input. */
static int read_job(pm1_job *j, FILE *input, FILE *resume, ecm_params p,
                    double target, double start, int primetest,
                    int specific_x0, mpq_t x0, mpgocandi_t *go) {
  int r;
  job_clear(j); job_init(j);
  j->start=start;
  if(resume) {
    int method, type, param;
    char program[256], who[256], when[256], comment[256];
    mpz_t y, sigma, a, y0;
    mpz_inits(y, sigma, a, y0, NULL);
    r=read_resumefile_line(&method, j->x, y, &j->n, sigma, a, j->x0, y0,
                          &type, &param, &j->start, program, who, when,
                          comment, resume, &j->gpu);
    mpz_clears(y, sigma, a, y0, NULL);
    if(r<=0) return r;
    if(method!=ECM_PM1 || !isfinite(j->start) || j->start<0 || j->start>target) {
      fprintf(stderr, "GPU P-1 requires P-1 residues with B1 <= requested B1.\n");
      return -1;
    }
    if(j->gpu.present && j->gpu.target!=target) {
      fprintf(stderr, "Finish the interrupted GPU exponent at B1=%.0f before changing B1.\n", j->gpu.target);
      return -1;
    }
    if(j->gpu.present) return 1;
    mpz_set(j->gpu.base, j->x);
    if(mpz_sgn(j->x0)==0) mpz_set(j->x0, j->x);
  } else {
    do {
      if(p->stop_asap && p->stop_asap()) return 0;
      r=read_number(&j->n, input, primetest, 1);
      if(!r) return ferror(input) ? -1 : 0;
    } while(mpz_cmp_ui(j->n.n, 1)<=0 || j->n.isPrp);
    if(specific_x0) {
      if(!mpz_invert(j->gpu.base, mpq_denref(x0), j->n.n)) {
        mpz_gcd(j->factor, mpq_denref(x0), j->n.n);
        j->status=ECM_FACTOR_FOUND_STEP1;
        mpz_set_ui(j->gpu.base, 1);
      } else {
        mpz_mul(j->gpu.base, j->gpu.base, mpq_numref(x0));
        mpz_mod(j->gpu.base, j->gpu.base, j->n.n);
      }
    } else if(mpz_even_p(j->n.n)) mpz_set_ui(j->gpu.base, 1);
    else __ECM(pm1_random_seed)(j->gpu.base, j->n.n, p->rng);
    mpz_mod(j->gpu.base, j->gpu.base, j->n.n);
    mpz_set(j->x0, j->gpu.base);
    mpgocandi_fixup_with_N(go, &j->n);
    if(mpz_cmp_ui(go->Candi.n, 1)>0)
      mpz_powm(j->gpu.base, j->gpu.base, go->Candi.n, j->n.n);
  }
  if(mpz_cmp_ui(j->n.n, 1)<=0) return -1;
  if(mpz_even_p(j->n.n)) {
    mpz_set_ui(j->factor, 2);
    j->status=ECM_FACTOR_FOUND_STEP1;
  } else if(j->status==ECM_NO_FACTOR_FOUND) {
    mpz_gcd(j->factor, j->gpu.base, j->n.n);
    if(mpz_cmp_ui(j->factor, 1)>0) j->status=ECM_FACTOR_FOUND_STEP1;
  }
  j->gpu.target=target;
  j->gpu.bits=0;
  j->gpu.present=(j->status==ECM_NO_FACTOR_FOUND);
  mpz_set_ui(j->x, 1);
  if(j->status!=ECM_NO_FACTOR_FOUND) mpz_set(j->x, j->gpu.base);
  return 1;
}

static int save_job(char *filename, pm1_job *j, ecm_params p) {
  mpz_set(p->x, j->x);
  p->B1done=j->start;
  return write_resumefile(filename, ECM_PM1, p, &j->n, j->n.n,
                         j->x0, NULL, "", &j->gpu);
}

/* Only the CUDA worker touches its private residues. Published jobs, progress
   messages and completion state are protected by this batch's mutex. All CLI
   parsing, CPU factoring, console output and file writing stay on the main
   thread, because those routines also use process-wide GMP-ECM state. */
typedef struct pm1_message {
  struct pm1_message *next;
  char text[256];
} pm1_message;

typedef struct {
  pm1_job *jobs;
  mpz_t *values;
  mpz_ptr *residues;
  mpz_srcptr *bases, *moduli;
  mpz_t exponent;
  unsigned capacity, count, reported, active;
  unsigned snapshot, saved_snapshot;
  uint64_t read_count;
  double target;
  int device, verbose, running, done, cancel, result;
  int (*stop_asap)(void);
  pm1_message *messages, *last_message;
#ifdef _WIN32
  CRITICAL_SECTION mutex;
  HANDLE thread;
#else
  pthread_mutex_t mutex;
  pthread_t thread;
#endif
} pm1_batch;

static void batch_lock(pm1_batch *b) {
#ifdef _WIN32
  EnterCriticalSection(&b->mutex);
#else
  pthread_mutex_lock(&b->mutex);
#endif
}
static void batch_unlock(pm1_batch *b) {
#ifdef _WIN32
  LeaveCriticalSection(&b->mutex);
#else
  pthread_mutex_unlock(&b->mutex);
#endif
}

static int batch_init(pm1_batch *b, unsigned capacity) {
  unsigned i;
  memset(b, 0, sizeof(*b));
  b->jobs=(pm1_job *)calloc(capacity, sizeof(pm1_job));
  b->values=(mpz_t *)malloc((size_t)capacity*sizeof(mpz_t));
  b->residues=(mpz_ptr *)malloc((size_t)capacity*sizeof(mpz_ptr));
  b->bases=(mpz_srcptr *)malloc((size_t)capacity*sizeof(mpz_srcptr));
  b->moduli=(mpz_srcptr *)malloc((size_t)capacity*sizeof(mpz_srcptr));
  if(!b->jobs || !b->values || !b->residues || !b->bases || !b->moduli) goto fail;
#ifdef _WIN32
  InitializeCriticalSection(&b->mutex);
#else
  if(pthread_mutex_init(&b->mutex, NULL)) goto fail;
#endif
  b->capacity=capacity;
  mpz_init(b->exponent);
  for(i=0; i<capacity; ++i) { job_init(b->jobs+i); mpz_init(b->values[i]); }
  return 1;
fail:
  free(b->jobs); free(b->values); free(b->residues); free(b->bases); free(b->moduli);
  return 0;
}

/* Detach before printing: a slow console must never hold the worker's lock. */
static void batch_progress(pm1_batch *b) {
  pm1_message *m, *next;
  batch_lock(b);
  m=b->messages; b->messages=b->last_message=NULL;
  batch_unlock(b);
  while(m) { next=m->next; fputs(m->text, stdout); free(m); m=next; }
}
static void queue_progress(void *arg, const char *text) {
  pm1_batch *b=(pm1_batch *)arg;
  pm1_message *m=(pm1_message *)malloc(sizeof(*m));
  if(!m) return; /* Losing diagnostics must not invalidate arithmetic. */
  m->next=NULL;
  snprintf(m->text, sizeof(m->text), "%s", text);
  batch_lock(b);
  if(b->last_message) b->last_message->next=m;
  else b->messages=m;
  b->last_message=m;
  batch_unlock(b);
}
static void batch_clear(pm1_batch *b) {
  unsigned i;
  batch_progress(b);
  for(i=0; i<b->capacity; ++i) { job_clear(b->jobs+i); mpz_clear(b->values[i]); }
  mpz_clear(b->exponent);
  free(b->jobs); free(b->values); free(b->residues); free(b->bases); free(b->moduli);
#ifdef _WIN32
  DeleteCriticalSection(&b->mutex);
#else
  pthread_mutex_destroy(&b->mutex);
#endif
}
static int worker_stop(void *arg) {
  pm1_batch *b=(pm1_batch *)arg;
  int cancel;
  batch_lock(b); cancel=b->cancel; batch_unlock(b);
  return cancel || (b->stop_asap && b->stop_asap());
}
static void batch_cancel(pm1_batch *b) {
  batch_lock(b); b->cancel=1; batch_unlock(b);
}
static int publish_snapshot(void *arg, uint64_t bits) {
  pm1_batch *b=(pm1_batch *)arg;
  unsigned i, active=0;
  uint64_t length=mpz_sizeinbase(b->exponent, 2);
  batch_lock(b);
  for(i=0; i<b->count; ++i) {
    pm1_job *j=b->jobs+i;
    if(j->status!=ECM_NO_FACTOR_FOUND) continue;
    mpz_set(j->x, b->values[active++]);
    j->gpu.bits=bits;
    if(bits==length) { j->gpu.present=0; j->start=b->target; }
  }
  ++b->snapshot;
  batch_unlock(b);
  return 1;
}
#ifdef _WIN32
static unsigned __stdcall batch_worker(void *arg)
#else
static void *batch_worker(void *arg)
#endif
{
  pm1_batch *b=(pm1_batch *)arg;
  cgbn_pm1_callbacks callbacks={worker_stop, publish_snapshot, queue_progress, b};
  uint64_t bits=b->jobs[0].gpu.bits;
  float milliseconds=0;
  int result=cgbn_pm1_stage1_run(b->residues, b->bases, b->moduli, b->active,
                    b->exponent, &bits, b->device, &milliseconds, b->verbose, &callbacks);
  batch_lock(b); b->result=result; b->done=1; batch_unlock(b);
#ifdef _WIN32
  return 0;
#else
  return NULL;
#endif
}

static int fill_batch(pm1_batch *b, pm1_job *pending, int *have_pending,
                       int *eof, uint64_t *read_count, FILE *input, FILE *resume,
                       ecm_params p, double B1, double B1done, int primetest,
                       int specific_x0, mpq_t x0, mpgocandi_t *go) {
  b->count=b->reported=b->snapshot=b->saved_snapshot=0;
  b->done=b->cancel=b->result=0;
  if(*have_pending) {
    pm1_job swap=b->jobs[0]; b->jobs[0]=*pending; *pending=swap;
    b->count=1; *have_pending=0;
  }
  while(b->count<b->capacity && !*eof && !(p->stop_asap && p->stop_asap())) {
    pm1_job *j=b->jobs+b->count;
    int r=read_job(j, input, resume, p, B1, B1done, primetest, specific_x0, x0, go);
    if(r<0) return 0;
    if(!r) { *eof=1; break; }
    ++*read_count;
    if(b->count && (j->start!=b->jobs[0].start || j->gpu.bits!=b->jobs[0].gpu.bits)) {
      pm1_job swap=*pending; *pending=*j; *j=swap;
      *have_pending=1; break;
    }
    ++b->count;
  }
  b->read_count=*read_count;
  return 1;
}

static int start_batch(pm1_batch *b, ecm_params p, double target,
                        mpz_t saved_exponent, mpz_t continuation, double *cached_start) {
  unsigned i;
  mpz_t divisor;
  char message[160];
  b->active=0;
  b->target=target; b->device=p->gpu_device; b->verbose=p->verbose;
  b->stop_asap=p->stop_asap;
  for(i=0; i<b->count; ++i) if(b->jobs[i].status==ECM_NO_FACTOR_FOUND) {
    mpz_set(b->values[b->active], b->jobs[i].x);
    b->residues[b->active]=b->values[b->active];
    b->bases[b->active]=b->jobs[i].gpu.base;
    b->moduli[b->active]=b->jobs[i].n.n;
    ++b->active;
  }
  if(p->verbose) {
    snprintf(message, sizeof(message), "GPU P-1 batch: %u inputs (%llu read so far)\n",
             b->count, (unsigned long long)b->read_count);
    queue_progress(b, message);
  }
  if(!b->active) { b->done=1; ++b->snapshot; return 1; }
  if(*cached_start!=b->jobs[0].start) {
    if(mpz_sgn(saved_exponent)==0) {
      if(target<2) mpz_set_ui(saved_exponent, 1);
      else compute_s(saved_exponent, (ecm_uint)target, NULL);
    }
    mpz_set(continuation, saved_exponent);
    if(b->jobs[0].start>1) {
      mpz_init(divisor);
      compute_s(divisor, (ecm_uint)b->jobs[0].start, NULL);
      if(!mpz_divisible_p(continuation, divisor)) { mpz_clear(divisor); return 0; }
      mpz_divexact(continuation, continuation, divisor);
      mpz_clear(divisor);
    }
    *cached_start=b->jobs[0].start;
  }
  mpz_set(b->exponent, continuation);
#ifdef _WIN32
  b->thread=(HANDLE)_beginthreadex(NULL, 0, batch_worker, b, 0, NULL);
  if(!b->thread) return 0;
#else
  if(pthread_create(&b->thread, NULL, batch_worker, b)) return 0;
#endif
  b->running=1;
  return 1;
}

/* The main thread alone writes snapshots. Include every read but unreported
   input, in source order, including a prefetched batch and a boundary lookahead.
   A worker publishes coherent residues before requesting a checkpoint. */
static int checkpoint_buffers(char *filename, pm1_batch *a, pm1_batch *b,
                               pm1_job *pending, ecm_params p, int force) {
  unsigned i;
  char *temporary;
  FILE *f;
  int ok=1;
  if(!filename) return 1;
  batch_lock(a); batch_lock(b);
  if(!force && a->snapshot==a->saved_snapshot && b->snapshot==b->saved_snapshot) goto unlock;
  temporary=(char *)malloc(strlen(filename)+5);
  if(!temporary) { ok=0; goto unlock; }
  sprintf(temporary, "%s.tmp", filename);
  f=fopen(temporary, "w");
  if(!f) { perror(temporary); free(temporary); ok=0; goto unlock; }
  if(fclose(f)) ok=0;
  for(i=a->reported; i<a->count && ok; ++i) ok=save_job(temporary, a->jobs+i, p);
  for(i=b->reported; i<b->count && ok; ++i) ok=save_job(temporary, b->jobs+i, p);
  if(ok && pending) ok=save_job(temporary, pending, p);
  if(ok) {
#ifdef _WIN32
    ok=MoveFileExA(temporary, filename, MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)!=0;
#else
    ok=rename(temporary, filename)==0;
#endif
  }
  free(temporary);
  if(ok) { a->saved_snapshot=a->snapshot; b->saved_snapshot=b->snapshot; }
unlock:
  batch_unlock(b); batch_unlock(a);
  if(!ok) fprintf(stderr, "Could not write GPU P-1 checkpoint %s\n", filename);
  return ok;
}

static int wait_batch(pm1_batch *a, pm1_batch *b, pm1_job *pending,
                       char *checkpoint, ecm_params p) {
  int done, ok=1;
  for(;;) {
    batch_lock(a); done=a->done; batch_unlock(a);
    if(done || !a->running) break;
    batch_progress(a);
    if(ok && !checkpoint_buffers(checkpoint, a, b, pending, p, 0)) {
      ok=0; batch_cancel(a);
    }
#ifdef _WIN32
    Sleep(20);
#else
    usleep(20000);
#endif
  }
  if(a->running) {
#ifdef _WIN32
    WaitForSingleObject(a->thread, INFINITE); CloseHandle(a->thread);
#else
    pthread_join(a->thread, NULL);
#endif
    a->running=0;
  }
  return ok && !a->result;
}

/* Keep CPU stage 2 and all reporting serial, while the other batch's dedicated
   worker continues issuing CUDA launches even if stdout is blocked. */
static int report_batch(pm1_batch *a, pm1_batch *b, pm1_job *pending,
                         ecm_params p, double B1, char *savefile, char *checkpoint,
                         int timestamp, int *result) {
  mpz_t f, dummy, original;
  int ok=1, verbose=p->verbose;
  mpz_inits(f, dummy, original, NULL);
  batch_progress(a);
  while(a->reported<a->count && !(p->stop_asap && p->stop_asap())) {
    pm1_job *j=a->jobs+a->reported;
    unsigned remaining=0;
    int was_prp=0;
    if(timestamp && verbose) { time_t now=time(NULL); printf("[%.24s]\n", ctime(&now)); }
    print_brent_source(&j->n, verbose ? stdout : stderr);
    if(verbose) gmp_printf("Input number is %Zd (%u digits)\n", j->n.n, j->n.ndigits);
    p->B1done=j->start;
    mpz_set(p->x, j->x);
    mpz_set(p->B2actual, p->B2);
    mpz_set(original, j->n.n);
    if(j->status==ECM_NO_FACTOR_FOUND) {
      mpz_sub_ui(f, j->x, 1);
      mpz_gcd(j->factor, f, j->n.n);
      if(mpz_cmp_ui(j->factor, 1)>0) j->status=ECM_FACTOR_FOUND_STEP1;
    }
    if(j->status==ECM_NO_FACTOR_FOUND) j->status=ecm_factor(j->factor, j->n.n, B1, p);
    else if(verbose>=3) gmp_printf("x=%Zd\n", j->x);
    if(j->status==ECM_ERROR || j->status==ECM_USER_ERROR) { ok=0; break; }
    if(j->status!=ECM_NO_FACTOR_FOUND)
      *result=process_newfactor(j->factor, j->status, &j->n, ECM_PM1, 0, 0,
               &remaining, &was_prp, dummy, NULL, verbose, 1, p, p->sigma, B1);
    if(!verbose && mpz_cmp_ui(j->n.n, 1)>0) gmp_printf("%Zd\n", j->n.n);
    if(savefile && !j->n.isPrp && mpz_cmp_ui(j->n.n, 1)>0 &&
       !write_resumefile(savefile, ECM_PM1, p, &j->n, original, j->x0, NULL, "", NULL)) {
      ok=0; break;
    }
    ++a->reported;
    if(!checkpoint_buffers(checkpoint, a, b, pending, p, 0)) { ok=0; break; }
  }
  mpz_clears(f, dummy, original, NULL);
  return ok;
}

int pm1_gpu_run(FILE *input, FILE *resume, ecm_params p,
                double B1, double B1done, unsigned capacity, int primetest,
                int specific_x0, mpq_t x0, mpgocandi_t *go,
                char *savefile, char *checkpoint, int timestamp) {
  pm1_batch buffers[2], *a=buffers, *b=buffers+1, *swap;
  pm1_job pending;
  mpz_t saved_exponent, continuation;
  double cached_start=-1;
  uint64_t read_count=0;
  unsigned i;
  int have_pending=0, eof=0, result=0, error=0;
  if(!capacity) capacity=4096;
  if(!batch_init(a, capacity)) return ECM_EXIT_ERROR;
  if(!batch_init(b, capacity)) { batch_clear(a); return ECM_EXIT_ERROR; }
  job_init(&pending);
  mpz_inits(saved_exponent, continuation, NULL);
  p->gpu=0; p->chkfilename=NULL;
  mpz_set_ui(p->go, 1); mpz_set_ui(p->y, 0);
  p->param=ECM_PARAM_DEFAULT;
  set_verbose(p->verbose);
  if(!fill_batch(a, &pending, &have_pending, &eof, &read_count, input, resume,
                p, B1, B1done, primetest, specific_x0, x0, go)) error=1;
  if(!error && a->count && !start_batch(a, p, B1, saved_exponent, continuation, &cached_start)) error=1;
  while(!error && a->count) {
    /* Parsing happens on the main thread while stage 1 runs on the worker. */
    if(!fill_batch(b, &pending, &have_pending, &eof, &read_count, input, resume,
                  p, B1, B1done, primetest, specific_x0, x0, go)) { error=1; break; }
    if(!checkpoint_buffers(checkpoint, a, b, have_pending ? &pending : NULL, p, 1)) { error=1; break; }
    if(!wait_batch(a, b, have_pending ? &pending : NULL, checkpoint, p)) { error=1; break; }
    if(p->stop_asap && p->stop_asap()) break;
    /* Submit the next batch BEFORE printing or saving the completed one. */
    if(b->count && !start_batch(b, p, B1, saved_exponent, continuation, &cached_start)) { error=1; break; }
    if(!checkpoint_buffers(checkpoint, a, b, have_pending ? &pending : NULL, p, 0) ||
       !report_batch(a, b, have_pending ? &pending : NULL, p, B1, savefile, checkpoint, timestamp, &result)) {
      error=1; break;
    }
    if(p->stop_asap && p->stop_asap()) break;
    a->count=a->reported=0;
    swap=a; a=b; b=swap;
  }
  if(error || (p->stop_asap && p->stop_asap())) {
    batch_cancel(a); batch_cancel(b);
    /* Join before saving/freeing any GMP objects owned by a worker. */
    if(!wait_batch(a, b, have_pending ? &pending : NULL, NULL, p)) error=1;
    if(!wait_batch(b, a, NULL, NULL, p)) error=1;
    if(!checkpoint_buffers(checkpoint, a, b, have_pending ? &pending : NULL, p, 1)) error=1;
    if(savefile) {
      for(i=a->reported; i<a->count; ++i) if(!save_job(savefile, a->jobs+i, p)) error=1;
      for(i=b->reported; i<b->count; ++i) if(!save_job(savefile, b->jobs+i, p)) error=1;
      if(have_pending && !save_job(savefile, &pending, p)) error=1;
    }
    fprintf(stderr, "GPU P-1 stopped after reading %llu inputs; saved buffered work%s. Unread inputs remain in the input file.\n",
            (unsigned long long)read_count, savefile || checkpoint ? "" : " nowhere (no save/checkpoint file)");
  }
  batch_clear(a); batch_clear(b); job_clear(&pending);
  mpz_clears(saved_exponent, continuation, NULL);
  return error ? ECM_EXIT_ERROR : result;
}
#endif
