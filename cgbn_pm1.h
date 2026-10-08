/* Batched P-1 stage 1.  GPL-3.0-or-later, like cgbn_stage1.cu. */
#ifndef CGBN_PM1_H
#define CGBN_PM1_H
#include <stdint.h>
#include <gmp.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Context-aware callbacks let a CLI worker publish snapshots and queue progress
   without writing to a potentially blocked console. All run on the caller's
   thread. A missing progress callback retains the ordinary stdout output. */
typedef struct {
  int (*stop)(void *);
  int (*checkpoint)(void *, uint64_t);
  void (*progress)(void *, const char *);
  void *arg;
} cgbn_pm1_callbacks;
int cgbn_pm1_stage1_run(mpz_ptr *residues, mpz_srcptr *bases, mpz_srcptr *moduli,
                       uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                       int device, float *milliseconds, int verbose,
                       const cgbn_pm1_callbacks *callbacks);
/* All arrays have count entries. Inputs/residues are ordinary integers,
   not Montgomery values. bits_done counts exponent bits consumed from MSB.
   Initialize residues to 1 and bits_done to 0 for a fresh exponentiation.
   Returns 0 on success (including a requested stop), -1 on error. Check
   bits_done against the exponent length before treating stage 1 as complete.
   checkpoint is called with current residues, periodically and at return. */
int cgbn_pm1_stage1(mpz_ptr *residues, mpz_srcptr *bases, mpz_srcptr *moduli,
                   uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                   int device, float *milliseconds, int verbose,
                   int (*stop_asap)(void),
                   int (*checkpoint)(void *, uint64_t), void *checkpoint_arg);
#ifdef __cplusplus
}
#endif
#endif
