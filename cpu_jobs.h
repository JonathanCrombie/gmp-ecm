/* Ordered command-line CPU jobs. The public single-job API is unchanged. */
#ifndef CPU_JOBS_H
#define CPU_JOBS_H
#include "ecm-ecm.h"
typedef struct {
  unsigned threads, count;
  int primetest, deep, timestamp, use_ntt, param;
  int specific_x0, specific_y0, specific_A;
  mpq_srcptr x0, y0, A;
  mpz_srcptr sigma;
  mpgocandi_t *go;
  double B1, B1done;
  char *savefile, *torsion, *load_s, *save_s;
  int load_s_mmap, save_s_mmap;
} cpu_job_options;
#ifdef HAVE_GWNUM
int brent_kbnc(unsigned long *, unsigned long *, signed long *, const mpcandi_t *);
#endif
int cpu_jobs_run(FILE *, FILE *, ecm_params, const cpu_job_options *);
#endif
