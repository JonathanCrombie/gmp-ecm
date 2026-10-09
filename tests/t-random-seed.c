/* Verify that automatic RNG initialization consumes all 64 seed bits and
   preserves an initialized stream. GPL-3.0-or-later. */
#include "ecm-impl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t supplied_seed;
static unsigned entropy_calls;
static size_t entropy_bytes;

/* Supply deterministic OS entropy to the real initialization code. */
#if defined (_MSC_VER) || defined (__MINGW32__)
#include <windows.h>
#include <wincrypt.h>
static BOOL WINAPI test_acquire(HCRYPTPROV *provider, LPCSTR container,
                               LPCSTR name, DWORD type, DWORD flags)
{
  (void)container; (void)name; (void)type; (void)flags;
  *provider = 1;
  return TRUE;
}
static BOOL WINAPI test_random(HCRYPTPROV provider, DWORD length, BYTE *buffer)
{
  (void)provider;
  entropy_calls++;
  entropy_bytes = length;
  if (length > sizeof(supplied_seed)) return FALSE;
  memcpy(buffer, &supplied_seed, length);
  return TRUE;
}
static BOOL WINAPI test_release(HCRYPTPROV provider, DWORD flags)
{
  (void)provider; (void)flags;
  return TRUE;
}
#undef CryptAcquireContext
#define CryptAcquireContext test_acquire
#define CryptGenRandom test_random
#define CryptReleaseContext test_release
#else
static FILE *test_open(const char *path, const char *mode)
{
  (void)mode;
  if (strcmp(path, "/dev/urandom") != 0) abort();
  return (FILE *)&supplied_seed;
}
static size_t test_read(void *buffer, size_t size, size_t count, FILE *stream)
{
  (void)stream;
  entropy_calls++;
  entropy_bytes = size * count;
  if (entropy_bytes > sizeof(supplied_seed)) return 0;
  memcpy(buffer, &supplied_seed, entropy_bytes);
  return count;
}
static int test_close(FILE *stream) { (void)stream; return 0; }
#define fopen test_open
#define fread test_read
#define fclose test_close
#endif
#include "../random.c"

#define REQUIRE(condition) do { if (!(condition)) { \
  fprintf(stderr, "Failed: %s (line %d)\n", #condition, __LINE__); \
  return EXIT_FAILURE; } } while (0)

int main(void)
{
  static const char *seeds[] = {
    "1234567887654321", "fedcba9887654321", "0",
    "8000000000000000", "ffffffffffffffff"
  };
  ecm_params actual;
  gmp_randstate_t expected;
  mpz_t seed, a, b, n;
  size_t i;
  unsigned draw;
  mpz_inits(seed, a, b, n, NULL);
  mpz_set_ui(n, 1);
  mpz_mul_2exp(n, n, 127);
  mpz_sub_ui(n, n, 1);
  for (i = 0; i < sizeof(seeds) / sizeof(seeds[0]); i++)
    {
      gmp_randinit_default(expected);
      REQUIRE(mpz_set_str(seed, seeds[i], 16) == 0);
      supplied_seed = 0;
      mpz_export(&supplied_seed, NULL, 1, sizeof(supplied_seed), 0, 0, seed);
      entropy_calls = 0;
      entropy_bytes = 0;
      ecm_init(actual);
      REQUIRE(entropy_calls == 1 && entropy_bytes == 8);
      gmp_randseed(expected, seed);
      for (draw = 0; draw < 16; draw++)
        {
          mpz_urandomb(a, actual->rng, 128);
          mpz_urandomb(b, expected, 128);
          REQUIRE(mpz_cmp(a, b) == 0);
        }
      /* P-/+1 must preserve the existing stream, as must repeated ECM
         attempts and explicit per-job seeds wider than 64 bits. */
      pm1_random_seed(a, n, actual->rng);
      pm1_random_seed(b, n, expected);
      REQUIRE(entropy_calls == 1 && mpz_cmp(a, b) == 0);
      pp1_random_seed(a, n, actual->rng);
      pp1_random_seed(b, n, expected);
      REQUIRE(entropy_calls == 1 && mpz_cmp(a, b) == 0);
      for (draw = 0; draw < 3; draw++)
        {
          if (draw == 2)
            {
              mpz_setbit(seed, 128);
              gmp_randseed(actual->rng, seed);
              gmp_randseed(expected, seed);
            }
          ecm_reset(actual);
          actual->param = ECM_PARAM_SUYAMA;
          mpz_set_ui(actual->B2, 0);
          mpz_urandomb(b, expected, 64);
          REQUIRE(ecm_factor(a, n, 10, actual) != ECM_ERROR);
          REQUIRE(entropy_calls == 1 && mpz_cmp(actual->sigma, b) == 0);
        }
      ecm_clear(actual);
      gmp_randclear(expected);
    }
  mpz_clears(seed, a, b, n, NULL);
  puts("PASS: full 64-bit OS seeds and preserved RNG state (105 comparisons)");
  return EXIT_SUCCESS;
}
