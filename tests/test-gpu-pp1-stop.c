/* Actual CUDA interruption and continuation, checked with companion-matrix
   exponentiation rather than the GPU Lucas ladder. GPL-3.0-or-later. */
#include <stdio.h>
#include <stdlib.h>
#include "cgbn_pm1.h"

#define COUNT 37
#define REQUIRE(x) do { if(!(x)) { fprintf(stderr, "Failed: %s at %d\n", #x, __LINE__); return 1; } } while(0)
static int polls, snapshots;
static int stop(void *arg) { (void)arg; return ++polls>2; }
static int snapshot(void *arg, uint64_t bits) {
  (void)arg; ++snapshots; return bits>0;
}

static void matrix_mul(mpz_t *r, mpz_t *a, mpz_t *b, mpz_srcptr n) {
  mpz_t t[4], product;
  unsigned i, j;
  mpz_init(product);
  for(i=0; i<4; ++i) {
    mpz_init(t[i]);
    for(j=0; j<2; ++j) {
      mpz_mul(product, a[(i/2)*2+j], b[2*j+i%2]);
      mpz_add(t[i], t[i], product);
    }
    mpz_mod(t[i], t[i], n);
  }
  for(i=0; i<4; ++i) { mpz_set(r[i], t[i]); mpz_clear(t[i]); }
  mpz_clear(product);
}

static void oracle(mpz_t x, mpz_t y, mpz_srcptr a, mpz_srcptr e, mpz_srcptr n) {
  mpz_t r[4], m[4];
  mp_bitcnt_t bit;
  unsigned i;
  for(i=0; i<4; ++i) { mpz_init(r[i]); mpz_init(m[i]); }
  mpz_set_ui(r[0], 1); mpz_set_ui(r[3], 1);
  mpz_set(m[0], a); mpz_set_si(m[1], -1); mpz_set_ui(m[2], 1);
  for(bit=0; bit<mpz_sizeinbase(e, 2); ++bit) {
    if(mpz_tstbit(e, bit)) matrix_mul(r, r, m, n);
    matrix_mul(m, m, m, n);
  }
  mpz_add(x, r[0], r[3]); mpz_mod(x, x, n);
  mpz_mul(y, a, r[0]); mpz_add(y, y, r[1]); mpz_sub(y, y, r[2]); mpz_mod(y, y, n);
  for(i=0; i<4; ++i) { mpz_clear(r[i]); mpz_clear(m[i]); }
}

int main(void) {
  mpz_t n[COUNT], a[COUNT], x[COUNT], y[COUNT], exponent, prefix, u, v;
  mpz_ptr residues[COUNT], adjacent[COUNT];
  mpz_srcptr bases[COUNT], moduli[COUNT];
  gmp_randstate_t rng;
  cgbn_pm1_callbacks callbacks={stop, snapshot, NULL, NULL};
  uint64_t done=0, length;
  float milliseconds=0;
  unsigned i;
  gmp_randinit_default(rng); gmp_randseed_ui(rng, 11102026);
  mpz_inits(exponent, prefix, u, v, NULL);
  mpz_urandomb(exponent, rng, 4096); mpz_setbit(exponent, 4095);
  length=mpz_sizeinbase(exponent, 2);
  for(i=0; i<COUNT; ++i) {
    mpz_init(n[i]); mpz_init(a[i]); mpz_init_set_ui(x[i], 2); mpz_init(y[i]);
    mpz_urandomb(n[i], rng, 256+i*8); mpz_setbit(n[i], 255+i*8); mpz_setbit(n[i], 0);
    mpz_urandomm(a[i], rng, n[i]);
    if(i==0) mpz_set_ui(a[i], 0);
    if(i==1) mpz_sub_ui(a[i], n[i], 1);
    mpz_set(y[i], a[i]);
    residues[i]=x[i]; adjacent[i]=y[i]; bases[i]=a[i]; moduli[i]=n[i];
  }
  REQUIRE(cgbn_pp1_stage1_run(residues, adjacent, bases, moduli, COUNT, exponent,
                            &done, -1, &milliseconds, 0, &callbacks)==0);
  REQUIRE(done>0 && done<length && snapshots==1);
  mpz_fdiv_q_2exp(prefix, exponent, length-done);
  for(i=0; i<COUNT; ++i) {
    oracle(u, v, a[i], prefix, n[i]);
    REQUIRE(mpz_cmp(x[i], u)==0 && mpz_cmp(y[i], v)==0);
  }
  callbacks.stop=NULL;
  REQUIRE(cgbn_pp1_stage1_run(residues, adjacent, bases, moduli, COUNT, exponent,
                            &done, -1, &milliseconds, 0, &callbacks)==0);
  REQUIRE(done==length && snapshots==2);
  for(i=0; i<COUNT; ++i) {
    oracle(u, v, a[i], exponent, n[i]);
    REQUIRE(mpz_cmp(x[i], u)==0 && mpz_cmp(y[i], v)==0);
    mpz_clear(n[i]); mpz_clear(a[i]); mpz_clear(x[i]); mpz_clear(y[i]);
  }
  mpz_clears(exponent, prefix, u, v, NULL); gmp_randclear(rng);
  puts("PASS: GPU P+1 stopped/resumed Lucas pairs match matrix oracle for 37 moduli");
  return 0;
}
