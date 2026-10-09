/* Independent CPU jobs with a bounded queue and ordered result collection.
   Parsing, candidate/cofactor updates, APRCL, logging and saves are performed
   only by the controller. Workers own all writable arithmetic and output. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "ecm-impl.h"
#include "cpu_jobs.h"
#include "torsions.h"
#ifdef _OPENMP
#include <omp.h>
#endif

typedef struct {
  mpcandi_t candidate;
  mpz_t original;
  unsigned refs;
  int stopped;
} cpu_group;

typedef struct cpu_job {
  struct cpu_job *next;
  cpu_group *group;
  ecm_params p;
  mpz_t n, factor, x0, y0;
  FILE *output, *errors;
  char comment[256];
  char *checkpoint, *tree;
  uint64_t id;
  unsigned run;
  int result, done;
} cpu_job;

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t work, completed;
  cpu_job *head, *tail;
  int shutdown, abort;
  int (*stop)(void);
  double B1;
} cpu_pool;

static ECM_THREAD_LOCAL cpu_pool *current_pool;
#ifdef HAVE_GWNUM
extern void __ecm_gw_thread_prepare(void);
extern void __ecm_gw_thread_cleanup(void);
#endif

static int worker_stop(void)
{
  cpu_pool *pool = current_pool;
  int stop;
  pthread_mutex_lock(&pool->mutex);
  stop = pool->abort;
  pthread_mutex_unlock(&pool->mutex);
  return stop || (pool->stop && pool->stop());
}

static void *worker(void *arg)
{
  cpu_pool *pool = arg;
  current_pool = pool;
  __ecm_cpu_worker = 1;
#ifdef _OPENMP
  /* Independent jobs own the CPU budget; do not create nested teams. */
  omp_set_num_threads(1);
#endif
  for (;;) {
    cpu_job *j;
    pthread_mutex_lock(&pool->mutex);
    while (!pool->head && !pool->shutdown)
      pthread_cond_wait(&pool->work, &pool->mutex);
    j = pool->head;
    if (!j) { pthread_mutex_unlock(&pool->mutex); break; }
    pool->head = j->next;
    if (!pool->head) pool->tail = NULL;
    pthread_mutex_unlock(&pool->mutex);
    j->p->stop_asap = worker_stop;
    if (j->result == ECM_NO_FACTOR_FOUND)
      {
        /* Queued jobs still need a valid curve/base for their save record,
           but should not start another full stage 1 after a stop request. */
        double bound = worker_stop() ? MAX(1.0, j->p->B1done) : pool->B1;
        j->result = ecm_factor(j->factor, j->n, bound, j->p);
      }
    /* The library selects an automatic parametrization by value. Persist
       the actual choice so the saved curve can be reconstructed on resume. */
    if (j->p->method == ECM_ECM && j->p->param == ECM_PARAM_DEFAULT)
      j->p->param = j->p->curve_param;
    fflush(j->output); fflush(j->errors);
    pthread_mutex_lock(&pool->mutex);
    j->done = 1;
    pthread_cond_broadcast(&pool->completed);
    pthread_mutex_unlock(&pool->mutex);
  }
  rhoinit(1, 0);
#ifdef HAVE_GWNUM
  __ecm_gw_thread_cleanup();
#endif
  current_pool = NULL;
  return NULL;
}

static void group_release(cpu_group *g)
{
  if (g && --g->refs == 0) {
    mpcandi_t_free(&g->candidate);
    mpz_clear(g->original);
    free(g);
  }
}

static cpu_group *group_new(const mpcandi_t *n)
{
  cpu_group *g = calloc(1, sizeof(*g));
  if (!g) return NULL;
  mpcandi_t_init(&g->candidate);
  mpcandi_t_add_candidate(&g->candidate, (mpz_ptr)n->n, n->cpExpr, 0);
  strcpy(g->candidate.brent_label, n->brent_label);
  g->candidate.isPrp = n->isPrp;
  mpz_init_set(g->original, n->n);
  g->refs = 1; /* producer reference, separate from queued jobs */
  return g;
}

static char *job_path(const char *base, uint64_t id)
{
  char *path;
  if (!base) return NULL;
  path = malloc(strlen(base) + 40);
  if (path) sprintf(path, "%s.job-%llu", base, (unsigned long long)id);
  return path;
}

static void job_free(cpu_job *j)
{
  if (!j) return;
  if (j->output) fclose(j->output);
  if (j->errors) fclose(j->errors);
  ecm_clear(j->p);
  mpz_clears(j->n, j->factor, j->x0, j->y0, NULL);
  group_release(j->group);
  free(j->checkpoint); free(j->tree); free(j);
}

static cpu_job *job_new(ecm_params src, unsigned threads, uint64_t id)
{
  cpu_job *j = calloc(1, sizeof(*j));
  ecm_params_ptr p;
  if (!j) return NULL;
  ecm_init(j->p);
  mpz_inits(j->n, j->factor, j->x0, j->y0, NULL);
  j->id = id;
  j->output = tmpfile();
  j->errors = tmpfile();
  j->checkpoint = job_path(src->chkfilename, id);
  j->tree = job_path(src->TreeFilename, id);
  if (!j->output || !j->errors || (src->chkfilename && !j->checkpoint) ||
      (src->TreeFilename && !j->tree)) { job_free(j); return NULL; }
  p = j->p;
  p->method = src->method;
  p->verbose = src->verbose;
  p->repr = src->repr;
  p->nobase2step2 = src->nobase2step2;
  p->k = src->k; p->S = src->S;
  p->maxmem = src->maxmem / threads;
  p->stage1time = src->stage1time;
  p->gw_cl_flag = src->gw_cl_flag;
  p->os = j->output; p->es = j->errors;
  p->chkfilename = j->checkpoint;
  p->TreeFilename = j->tree;
  mpz_set(p->B2, src->B2); mpz_set(p->B2min, src->B2min);
  mpz_set(p->B2actual, src->B2);
  mpz_set(p->batch_s, src->batch_s);
  p->batch_last_B1_used = src->batch_last_B1_used;
  /* Seed before dispatch: no random state is shared by running workers. */
  mpz_urandomb(j->factor, src->rng, 128);
  mpz_setbit(j->factor, 128);
  gmp_randseed(p->rng, j->factor);
  mpz_set_ui(j->factor, 0);
  return j;
}

/* A noninvertible rational denominator is a stage-1 factor, reported by the
   same ordered collector as factors returned by the library. */
static int rational(cpu_job *j, mpz_ptr out, mpq_srcptr value)
{
  if (!mpz_invert(out, mpq_denref(value), j->n)) {
    mpz_gcd(j->factor, mpq_denref(value), j->n);
    j->result = ECM_FACTOR_FOUND_STEP1;
    return 0;
  }
  mpz_mul(out, out, mpq_numref(value));
  mpz_mod(out, out, j->n);
  return 1;
}

#ifdef HAVE_GWNUM
static void setup_gwnum(cpu_job *j, int resumed)
{
  ecm_params_ptr p = j->p;
  mpcandi_t *n = &j->group->candidate;
  int brent = 0;
  if (mpz_sizeinbase(j->n, 2) < 350 || p->method != ECM_ECM) return;
  if (p->gw_cl_flag >= 0)
    brent = brent_kbnc(&p->gw_b, &p->gw_n, &p->gw_c, n);
  if (brent) p->gw_k = 1;
  if (!brent && !kbnc_str(&p->gw_k, &p->gw_b, &p->gw_n, &p->gw_c,
                          n->cpExpr, j->n) &&
      !kbnc_z(&p->gw_k, &p->gw_b, &p->gw_n, &p->gw_c, j->n))
    p->gw_b = 0;
  if (brent && !resumed && p->param == ECM_PARAM_DEFAULT &&
      !p->sigma_is_A &&
      p->gw_n * log2((double)p->gw_b) >=
        (p->gw_cl_flag > 0 ? 350 : GWNUM_KBNC_THRESHOLD))
    p->param = ECM_PARAM_SUYAMA;
}
#endif

static int prepare_fresh(cpu_job *j, const cpu_job_options *o)
{
  ecm_params_ptr p = j->p;
  mpz_t a;
  p->B1done = o->B1done;
  p->param = o->param;
  mpz_set(p->sigma, o->sigma);
  if (o->specific_x0 && !rational(j, p->x, o->x0)) return 1;
  if (o->specific_y0 && !rational(j, p->y, o->y0)) return 1;
  mpz_set(j->x0, p->x); mpz_set(j->y0, p->y);
  if (o->specific_A) {
    mpz_init(a);
    if (o->param == ECM_PARAM_TWISTED_HESSIAN) {
      mpz_mod(p->E->a4, mpq_numref(o->A), j->n);
      mpz_mod(p->E->a6, mpq_denref(o->A), j->n);
    } else if (!rational(j, a, o->A)) { mpz_clear(a); return 1; }
    p->sigma_is_A = 1;
    mpz_set(p->sigma, a);
    if (o->param == ECM_PARAM_WEIERSTRASS) {
      mpz_mul(p->E->a6, p->y, p->y);
      mpz_mul(p->E->a4, p->x, p->x);
      mpz_add(p->E->a4, p->E->a4, a);
      mpz_mul(p->E->a4, p->E->a4, p->x);
      mpz_sub(p->E->a6, p->E->a6, p->E->a4);
      mpz_mod(p->E->a6, p->E->a6, j->n);
      mpz_set(p->E->a4, a);
      p->E->type = ECM_EC_TYPE_WEIERSTRASS;
    } else if (o->param == ECM_PARAM_HESSIAN) {
      mpz_set(p->E->a4, a);
      p->E->type = ECM_EC_TYPE_HESSIAN;
    } else if (o->param == ECM_PARAM_TWISTED_HESSIAN)
      p->E->type = ECM_EC_TYPE_TWISTED_HESSIAN;
    if (p->E->type != ECM_EC_TYPE_MONTGOMERY) {
      p->sigma_is_A = -1;
      p->E->law = ECM_LAW_HOMOGENEOUS;
    }
    mpz_clear(a);
  } else if (o->torsion) {
    p->param = ECM_PARAM_TORSION; p->sigma_is_A = -1;
    j->result = build_curves_with_torsion2(j->factor, j->n, p->E,
                                          p->x, p->y, o->torsion, p->sigma);
  }
  /* Record the actual starting base, including for random P-/+1 jobs. */
  if (!o->specific_x0 && p->method != ECM_ECM && mpz_odd_p(j->n)) {
    if (p->method == ECM_PM1) __ECM(pm1_random_seed)(p->x, j->n, p->rng);
    else __ECM(pp1_random_seed)(p->x, j->n, p->rng);
    mpz_set(j->x0, p->x);
  }
  return 1;
}

/* Produce jobs in serial input order. The producer keeps just the current
   input group; queued jobs hold references to older groups until collected. */
static int next_job(cpu_job **out, cpu_group **current, unsigned *issued,
                    uint64_t *id, FILE *input, FILE *resume, ecm_params base,
                    const cpu_job_options *o)
{
  cpu_job *j = NULL;
  mpcandi_t n;
  mpz_t a;
  int r;
  mpcandi_t_init(&n); mpz_init(a);
  *out = NULL;
  for (;;) {
    if (base->stop_asap && base->stop_asap()) { r = 0; break; }
    j = job_new(base, o->threads, *id + 1);
    if (!j) { fprintf(stderr, "Cannot allocate CPU job or temporary output files.\n"); r = -1; break; }
    if (resume) {
      char program[256], who[256], when[256];
      r = read_resumefile_line(&j->p->method, j->p->x, j->p->y, &n,
            j->p->sigma, a, j->x0, j->y0, &j->p->E->type, &j->p->param,
            &j->p->B1done, program, who, when, j->comment, resume, NULL);
      if (r <= 0) break;
      if (!*current || mpz_cmp(n.n, (*current)->original)) {
        group_release(*current); *current = group_new(&n); *issued = 0;
        if (!*current) { r = -1; break; }
      }
      j->p->sigma_is_A = mpz_sgn(j->p->sigma) == 0;
      if (j->p->E->type != ECM_EC_TYPE_MONTGOMERY) j->p->sigma_is_A = -1;
      if (j->p->sigma_is_A) mpz_set(j->p->sigma, a);
    } else if (!*current || (o->count && *issued == o->count) || (*current)->stopped) {
      do {
        r = read_number(&n, input, o->primetest, 1);
        if (!r) { r = ferror(input) ? -1 : 0; break; }
      } while (n.isPrp || mpz_cmp_ui(n.n, 1) <= 0);
      if (r <= 0) break;
      group_release(*current); *current = group_new(&n); *issued = 0;
      if (!*current) { r = -1; break; }
    }
    if ((*current)->stopped) { job_free(j); j = NULL; continue; }
    j->group = *current; ++j->group->refs;
    j->run = ++*issued;
    j->id = ++*id;
    mpz_set(j->n, j->group->candidate.n);
    if (!resume) prepare_fresh(j, o);
    mpgocandi_fixup_with_N(o->go, &j->group->candidate);
    mpz_set(j->p->go, o->go->Candi.n);
    j->p->use_ntt = o->use_ntt;
    if (o->use_ntt == 1 && (j->p->method == ECM_ECM || j->p->S != ECM_DEFAULT_S))
      j->p->use_ntt = mpz_size(j->n) <= NTT_SIZE_THRESHOLD;
#ifdef HAVE_GWNUM
    setup_gwnum(j, resume != NULL);
#endif
    *out = j; j = NULL; r = 1; break;
  }
  job_free(j); mpz_clear(a); mpcandi_t_free(&n);
  return r;
}

static int report_job(cpu_job *j, const cpu_job_options *o, int resumed,
                      int *returncode)
{
  cpu_group *g = j->group;
  mpcandi_t *n = &g->candidate;
  int verbose = j->p->verbose, ok = 1, was_prp = 0;
  unsigned remaining = 1;
  char buffer[8192];
  size_t size;
  mpz_t dummy;
  /* Earlier ordered results may have finished this candidate while the job
     was already running. Its residue must not be saved for a finished N. */
  if (g->stopped) return 1;
  if (o->timestamp && verbose) { time_t t = time(NULL); printf("[%.24s]\n", ctime(&t)); }
  print_brent_source(n, verbose ? stdout : stderr);
  if (verbose) {
    if (resumed) printf("Resuming %s residue\n", j->p->method == ECM_ECM ? "ECM" :
                        j->p->method == ECM_PM1 ? "P-1" : "P+1");
    if (!resumed && !o->count) printf("Run %u:\n", j->run);
    if (!resumed && o->count > 1) printf("Run %u out of %u:\n", j->run, o->count);
    gmp_printf("Input number is %Zd (%u digits)\n", j->n, nb_digits(j->n));
  }
  rewind(j->output);
  while ((size = fread(buffer, 1, sizeof(buffer), j->output)) != 0)
    if (fwrite(buffer, 1, size, stdout) != size) { ok = 0; break; }
  if (ferror(j->output)) ok = 0;
  rewind(j->errors);
  while ((size = fread(buffer, 1, sizeof(buffer), j->errors)) != 0)
    if (fwrite(buffer, 1, size, stderr) != size) { ok = 0; break; }
  if (ferror(j->errors)) ok = 0;
  if (j->result == ECM_ERROR || j->result == ECM_USER_ERROR) ok = 0;
  mpz_init_set_ui(dummy, 1);
  if (ok && j->result != ECM_NO_FACTOR_FOUND) {
    /* Two curves can discover the same factor before either is collected. */
    mpz_gcd(j->factor, j->factor, n->n);
    if (mpz_cmp_ui(j->factor, 1) > 0) {
      *returncode = process_newfactor(j->factor, j->result, n, j->p->method,
        0, 0, &remaining, &was_prp, dummy, NULL, verbose, o->deep,
        j->p, j->p->sigma, o->B1);
      if (!remaining) g->stopped = 1;
    }
  }
  mpz_clear(dummy);
  if (!verbose && (resumed || j->run == o->count || g->stopped) && mpz_cmp_ui(n->n, 1) > 0)
    gmp_printf("%Zd\n", n->n);
  if (j->result != ECM_ERROR && j->result != ECM_USER_ERROR &&
      o->savefile && !n->isPrp && mpz_cmp_ui(n->n, 1) > 0 &&
      !write_resumefile(o->savefile, j->p->method, j->p, n, j->n,
                        j->x0, j->y0, j->comment, NULL)) ok = 0;
  fflush(stdout);
  return ok;
}

int cpu_jobs_run(FILE *input, FILE *resume, ecm_params base, const cpu_job_options *o)
{
  cpu_pool pool;
  cpu_group *current = NULL;
  cpu_job **ordered;
  pthread_t *threads;
  unsigned started = 0, issued = 0, count = 0, head = 0, capacity = 2 * o->threads, i;
  uint64_t id = 0;
  int error = 0, eof = 0, result = 0;
  memset(&pool, 0, sizeof(pool));
  ordered = calloc(capacity, sizeof(*ordered));
  threads = calloc(o->threads, sizeof(*threads));
  if (!ordered || !threads) { free(ordered); free(threads); return ECM_EXIT_ERROR; }
  if (pthread_mutex_init(&pool.mutex, NULL)) { free(ordered); free(threads); return ECM_EXIT_ERROR; }
  if (pthread_cond_init(&pool.work, NULL)) { pthread_mutex_destroy(&pool.mutex); free(ordered); free(threads); return ECM_EXIT_ERROR; }
  if (pthread_cond_init(&pool.completed, NULL)) { pthread_cond_destroy(&pool.work); pthread_mutex_destroy(&pool.mutex); free(ordered); free(threads); return ECM_EXIT_ERROR; }
  pool.stop = base->stop_asap; pool.B1 = o->B1;
  set_verbose(base->verbose);
#ifdef HAVE_GWNUM
  __ecm_gw_thread_prepare();
#endif
  /* Shared batch exponent files are read/written by the controller once. */
  if (o->load_s) {
    error = read_s_from_file(base->batch_s, o->load_s, o->load_s_mmap, o->B1) != 0;
    base->batch_last_B1_used = o->B1;
  } else if (IS_BATCH_MODE(o->param) && !resume) {
    compute_s(base->batch_s, (ecm_uint)o->B1, NULL);
    base->batch_last_B1_used = o->B1;
  }
  if (!error && o->save_s) {
    if (base->batch_last_B1_used != o->B1) compute_s(base->batch_s, (ecm_uint)o->B1, NULL);
    error = !write_s_in_file(o->save_s, base->batch_s, o->save_s_mmap, (uint64_t)o->B1);
  }
  if (base->verbose) printf("CPU jobs: %u threads; output and saves in input order\n", o->threads);
  for (; !error && started < o->threads; ++started)
    if (pthread_create(threads + started, NULL, worker, &pool)) { fprintf(stderr, "Cannot create CPU worker %u.\n", started + 1); error = 1; break; }
  while (!error || count) {
    while (!error && !eof && count < capacity && !(pool.stop && pool.stop())) {
      cpu_job *j;
      int r = next_job(&j, &current, &issued, &id, input, resume, base, o);
      if (r <= 0) { eof = 1; if (r < 0) error = 1; break; }
      ordered[(head + count) % capacity] = j; ++count;
      pthread_mutex_lock(&pool.mutex);
      if (pool.tail) pool.tail->next = j; else pool.head = j;
      pool.tail = j;
      pthread_cond_signal(&pool.work);
      pthread_mutex_unlock(&pool.mutex);
    }
    if (error) {
      pthread_mutex_lock(&pool.mutex); pool.abort = 1; pthread_mutex_unlock(&pool.mutex);
    }
    if (!count) break;
    {
      cpu_job *j = ordered[head];
      pthread_mutex_lock(&pool.mutex);
      while (!j->done) pthread_cond_wait(&pool.completed, &pool.mutex);
      pthread_mutex_unlock(&pool.mutex);
      if (!report_job(j, o, resume != NULL, &result)) error = 1;
      job_free(j); ordered[head] = NULL;
      head = (head + 1) % capacity; --count;
    }
  }
  pthread_mutex_lock(&pool.mutex); pool.shutdown = 1;
  pthread_cond_broadcast(&pool.work); pthread_mutex_unlock(&pool.mutex);
  for (i = 0; i < started; ++i) pthread_join(threads[i], NULL);
  if (pool.stop && pool.stop())
    fprintf(stderr, "CPU jobs stopped after dispatching %llu jobs; buffered residues saved%s in input order. Unread inputs remain in the input file.\n",
      (unsigned long long)id, o->savefile ? "" : " nowhere (no -save file)");
  group_release(current);
  pthread_cond_destroy(&pool.completed); pthread_cond_destroy(&pool.work);
  pthread_mutex_destroy(&pool.mutex); free(ordered); free(threads);
  return error ? ECM_EXIT_ERROR : result;
}
