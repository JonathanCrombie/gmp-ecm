/* Exercise a real stop between CUDA launches and continuation of those values.
   Link with the GPU libecm, GMP and CUDA runtime. GPL-3.0-or-later. */
#include <stdio.h>
#include <stdlib.h>
#include "cgbn_pm1.h"

#define COUNT 37
#define REQUIRE(x) do { if(!(x)) { fprintf(stderr, "Failed: %s at %d\n", #x, __LINE__); return 1; } } while(0)
static int polls, snapshots;
static int stop_after_two_calls(void) { return ++polls>2; }
static int snapshot(void *unused, uint64_t bits) {
  (void)unused;
  ++snapshots;
  return bits>0;
}

int main(void) {
  mpz_t n[COUNT], a[COUNT], x[COUNT], exponent, prefix, expected;
  mpz_ptr residues[COUNT];
  mpz_srcptr bases[COUNT], moduli[COUNT];
  gmp_randstate_t rng;
  uint64_t done=0, length;
  float milliseconds=0;
  unsigned i;
  gmp_randinit_default(rng); gmp_randseed_ui(rng, 7102026);
  mpz_inits(exponent, prefix, expected, NULL);
  mpz_urandomb(exponent, rng, 4096); mpz_setbit(exponent, 4095);
  length=mpz_sizeinbase(exponent, 2);
  for(i=0; i<COUNT; ++i) {
    mpz_init(n[i]); mpz_init(a[i]); mpz_init_set_ui(x[i], 1);
    mpz_urandomb(n[i], rng, 256+i*8); mpz_setbit(n[i], 255+i*8); mpz_setbit(n[i], 0);
    mpz_urandomm(a[i], rng, n[i]);
    residues[i]=x[i]; bases[i]=a[i]; moduli[i]=n[i];
  }
  REQUIRE(cgbn_pm1_stage1(residues, bases, moduli, COUNT, exponent, &done,
            -1, &milliseconds, 0, stop_after_two_calls, snapshot, NULL)==0);
  REQUIRE(done>0 && done<length && snapshots==1);
  mpz_fdiv_q_2exp(prefix, exponent, length-done);
  for(i=0; i<COUNT; ++i) {
    mpz_powm(expected, a[i], prefix, n[i]);
    REQUIRE(mpz_cmp(x[i], expected)==0);
  }
  REQUIRE(cgbn_pm1_stage1(residues, bases, moduli, COUNT, exponent, &done,
            -1, &milliseconds, 0, NULL, snapshot, NULL)==0);
  REQUIRE(done==length && snapshots==2);
  for(i=0; i<COUNT; ++i) {
    mpz_powm(expected, a[i], exponent, n[i]);
    REQUIRE(mpz_cmp(x[i], expected)==0);
    mpz_clear(n[i]); mpz_clear(a[i]); mpz_clear(x[i]);
  }
  mpz_clears(exponent, prefix, expected, NULL); gmp_randclear(rng);
  puts("PASS: stopped GPU residues and resumed results match GMP for 37 mixed moduli");
  return 0;
}
