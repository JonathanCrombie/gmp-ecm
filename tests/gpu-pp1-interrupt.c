/* Test-only CLI wrapper: compile main.c with -Dmain=ecm_cli_main and
   pm1_gpu.c with -Dcgbn_pp1_stage1_run=test_pp1_stage1_run. A real signal
   after two launches exercises the ordinary CLI save/checkpoint path. */
#include <signal.h>
#include <stdlib.h>
#include "cgbn_pm1.h"

typedef struct {
  const cgbn_pm1_callbacks *original;
  unsigned polls;
} interrupt_context;

static int stop(void *arg) {
  interrupt_context *c=(interrupt_context *)arg;
  if(++c->polls>2) { raise(SIGTERM); return 1; }
  return c->original->stop && c->original->stop(c->original->arg);
}
static int checkpoint(void *arg, uint64_t bits) {
  interrupt_context *c=(interrupt_context *)arg;
  return !c->original->checkpoint || c->original->checkpoint(c->original->arg, bits);
}
static void progress(void *arg, const char *text) {
  interrupt_context *c=(interrupt_context *)arg;
  if(c->original->progress) c->original->progress(c->original->arg, text);
}
int test_pp1_stage1_run(mpz_ptr *residues, mpz_ptr *next,
                       mpz_srcptr *bases, mpz_srcptr *moduli,
                       uint32_t count, mpz_srcptr exponent, uint64_t *bits_done,
                       int device, float *milliseconds, int verbose,
                       const cgbn_pm1_callbacks *callbacks) {
  static unsigned batches;
  const char *selection=getenv("ECM_TEST_STOP_BATCH");
  unsigned batch=selection ? (unsigned)atoi(selection) : 1;
  interrupt_context context={callbacks, 0};
  cgbn_pm1_callbacks wrapped={stop, checkpoint, progress, &context};
  return cgbn_pp1_stage1_run(residues, next, bases, moduli, count, exponent,
                           bits_done, device, milliseconds, verbose,
                           ++batches==batch ? &wrapped : callbacks);
}
int ecm_cli_main(int, char **);
int main(int argc, char **argv) { return ecm_cli_main(argc, argv); }
