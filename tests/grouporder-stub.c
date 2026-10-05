/* Test the external-program boundary without linking PARI. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main (int argc, char **argv)
{
  const char *mode = getenv ("ECM_TEST_HELPER_MODE");
  (void) argv;
  if (argc != 4) return 9;
  if (mode && strcmp (mode, "oversize") == 0)
    {
      int i;
      for (i = 0; i < 1100000; i++) putchar ('x');
    }
  else if (mode && strcmp (mode, "malformed") == 0)
    puts ("Group Order: invalid\nFactored: also invalid");
  else
    puts ("Group Order: 120\nFactored: 2^3 * 3 * 5");
  return mode && strcmp (mode, "failure") == 0 ? 7 : 0;
}
