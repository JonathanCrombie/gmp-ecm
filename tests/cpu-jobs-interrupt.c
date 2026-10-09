/* Test-only executable: compile main.c with -Dmain=ecm_cli_main and link this
   entry point to exercise the real CLI signal/save path on Windows or POSIX. */
#include <signal.h>
#include <pthread.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
static void *interrupt_later(void *arg)
{
  (void)arg;
#ifdef _WIN32
  Sleep(1000);
#else
  sleep(1);
#endif
  raise(SIGTERM);
  return NULL;
}
int ecm_cli_main(int, char **);
int main(int argc, char **argv)
{
  pthread_t thread;
  int result;
  if (pthread_create(&thread, NULL, interrupt_later, NULL)) return 1;
  result = ecm_cli_main(argc, argv);
  pthread_join(thread, NULL);
  return result;
}
