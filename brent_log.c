/* Brent factor records and an optional, separately supplied grouporder helper.
   Released under the GNU GPL, version 3 or later; see COPYING. */
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <fcntl.h>
#endif
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ecm-impl.h"
#include "ecm-ecm.h"

#define HELPER_PATH_SIZE 32768
#define HELPER_OUTPUT_SIZE (1024 * 1024)

static void
free_gmp_string (char *s)
{
  void (*release) (void *, size_t);
  mp_get_memory_functions (NULL, NULL, &release);
  if (s != NULL) release (s, strlen (s) + 1);
}

/* Prefer the executable beside ECM; fall back to PATH. No shell is used. */
static int
helper_path (char *path)
{
  char *slash;
#ifdef _WIN32
  DWORD length = GetModuleFileNameA (NULL, path, HELPER_PATH_SIZE);
  if (length && length < HELPER_PATH_SIZE &&
      (slash = strrchr (path, '\\')) != NULL &&
      slash - path + sizeof ("\\grouporder.exe") <= HELPER_PATH_SIZE)
    {
      strcpy (slash, "\\grouporder.exe");
      if (GetFileAttributesA (path) != INVALID_FILE_ATTRIBUTES) return 1;
    }
  length = SearchPathA (getenv ("PATH"), "grouporder.exe", NULL,
                        HELPER_PATH_SIZE, path, NULL);
  return length && length < HELPER_PATH_SIZE;
#else
  ssize_t length = readlink ("/proc/self/exe", path, HELPER_PATH_SIZE - 1);
  const char *entry, *end;
  if (length > 0)
    {
      path[length] = 0;
      slash = strrchr (path, '/');
      if (slash && slash - path + sizeof ("/grouporder") <= HELPER_PATH_SIZE)
        {
          strcpy (slash, "/grouporder");
          if (access (path, X_OK) == 0) return 1;
        }
    }
  entry = getenv ("PATH");
  while (entry != NULL)
    {
      size_t n;
      end = strchr (entry, ':');
      n = end ? (size_t) (end - entry) : strlen (entry);
      if (n + sizeof ("/grouporder") <= HELPER_PATH_SIZE)
        {
          memcpy (path, entry, n);
          strcpy (path + n, n ? "/grouporder" : "./grouporder");
          if (access (path, X_OK) == 0) return 1;
        }
      entry = end ? end + 1 : NULL;
    }
  return 0;
#endif
}

/* Drain the pipe even if output exceeds the limit, then reject the result. */
static int
run_helper (const char *path, const char *factor, int param,
             const char *sigma, char *output)
{
  size_t used = 0;
  int failed = 0;
  char buffer[4096];
#ifdef _WIN32
  char *command = NULL;
  DWORD count, status = 1;
  HANDLE reader = NULL, writer = NULL, input = INVALID_HANDLE_VALUE;
  SECURITY_ATTRIBUTES security = { sizeof (security), NULL, TRUE };
  STARTUPINFOA startup;
  PROCESS_INFORMATION process;
  gmp_asprintf (&command, "\"%s\" %s %d %s", path, factor, param, sigma);
  if (command == NULL) return 0;
  if (!CreatePipe (&reader, &writer, &security, 0) ||
      !SetHandleInformation (reader, HANDLE_FLAG_INHERIT, 0)) goto cleanup;
  input = CreateFileA ("NUL", GENERIC_READ | GENERIC_WRITE,
                       FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
                       OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  if (input == INVALID_HANDLE_VALUE) goto cleanup;
  memset (&startup, 0, sizeof (startup));
  memset (&process, 0, sizeof (process));
  startup.cb = sizeof (startup);
  startup.dwFlags = STARTF_USESTDHANDLES;
  startup.hStdInput = input;
  startup.hStdOutput = writer;
  startup.hStdError = input;
  if (!CreateProcessA (path, command, NULL, NULL, TRUE, CREATE_NO_WINDOW,
                       NULL, NULL, &startup, &process)) goto cleanup;
  CloseHandle (writer);
  writer = NULL;
  while (ReadFile (reader, buffer, sizeof (buffer), &count, NULL) && count)
    {
      if (count > HELPER_OUTPUT_SIZE - used) failed = 1;
      else if (!failed) { memcpy (output + used, buffer, count); used += count; }
    }
  if (GetLastError () != ERROR_BROKEN_PIPE) failed = 1;
  WaitForSingleObject (process.hProcess, INFINITE);
  GetExitCodeProcess (process.hProcess, &status);
  CloseHandle (process.hThread);
  CloseHandle (process.hProcess);
cleanup:
  if (reader != NULL) CloseHandle (reader);
  if (writer != NULL) CloseHandle (writer);
  if (input != INVALID_HANDLE_VALUE) CloseHandle (input);
  free_gmp_string (command);
  output[used] = 0;
  return status == 0 && !failed;
#else
  int pipes[2], status = 0;
  pid_t child, waited;
  ssize_t count;
  char parameter[16];
  snprintf (parameter, sizeof (parameter), "%d", param);
  if (pipe (pipes) != 0) return 0;
  child = fork ();
  if (child == 0)
    {
      int nullfd = open ("/dev/null", O_RDWR);
      if (nullfd < 0 || dup2 (nullfd, STDIN_FILENO) < 0 ||
          dup2 (nullfd, STDERR_FILENO) < 0 ||
          dup2 (pipes[1], STDOUT_FILENO) < 0) _exit (127);
      if (nullfd > STDERR_FILENO) close (nullfd);
      close (pipes[0]);
      if (pipes[1] > STDERR_FILENO) close (pipes[1]);
      execl (path, path, factor, parameter, sigma, (char *) NULL);
      _exit (127);
    }
  close (pipes[1]);
  if (child < 0) { close (pipes[0]); return 0; }
  for (;;)
    {
      count = read (pipes[0], buffer, sizeof (buffer));
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) { if (count < 0) failed = 1; break; }
      if ((size_t) count > HELPER_OUTPUT_SIZE - used) failed = 1;
      else if (!failed) { memcpy (output + used, buffer, count); used += count; }
    }
  close (pipes[0]);
  do { waited = waitpid (child, &status, 0); } while (waited < 0 && errno == EINTR);
  output[used] = 0;
  return waited == child && WIFEXITED (status) && WEXITSTATUS (status) == 0 && !failed;
#endif
}

/* Accept only the helper's single-line numeric fields, never diagnostics. */
static char *
result_line (char *text, const char *prefix, const char *alphabet)
{
  size_t n = strlen (prefix);
  char *line = text;
  while (line != NULL && *line)
    {
      char *end;
      while (*line == ' ' || *line == '\t') line++;
      if (strncmp (line, prefix, n) == 0)
        {
          line += n;
          while (*line == ' ') line++;
          end = line + strcspn (line, "\r\n");
          while (end > line && end[-1] == ' ') end--;
          if (end == line || strspn (line, alphabet) < (size_t) (end - line)) return NULL;
          *end = 0;
          return line;
        }
      line = strchr (line, '\n');
      if (line != NULL) line++;
    }
  return NULL;
}

/* Serialize appends across ECM processes, including partial writes. */
static int
write_record (const char *record)
{
  size_t length = strlen (record), offset = 0;
#ifdef _WIN32
  DWORD error = 0;
  HANDLE file = CreateFileA ("BrentFactors.log", GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
                            OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
  OVERLAPPED lock = { 0 };
  LARGE_INTEGER end;
  if (file == INVALID_HANDLE_VALUE) error = GetLastError ();
  else
    {
      if (!LockFileEx (file, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &lock))
        error = GetLastError ();
      else
        {
          end.QuadPart = 0;
          if (!SetFilePointerEx (file, end, NULL, FILE_END)) error = GetLastError ();
          while (error == 0 && offset < length)
            {
              DWORD written, chunk = (DWORD) MIN (length - offset, 0x7fffffff);
              if (!WriteFile (file, record + offset, chunk, &written, NULL))
                error = GetLastError ();
              else if (!written) error = ERROR_WRITE_FAULT;
              else offset += written;
            }
          UnlockFileEx (file, 0, MAXDWORD, MAXDWORD, &lock);
        }
      if (!CloseHandle (file) && !error) error = GetLastError ();
    }
  if (error) fprintf (stderr, "Warning: cannot append to BrentFactors.log (Windows error %lu).\n",
                       (unsigned long) error);
#else
  int error = 0;
  int file = open ("BrentFactors.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
  struct flock lock;
  if (file < 0) error = errno;
  else
    {
      memset (&lock, 0, sizeof (lock));
      lock.l_type = F_WRLCK;
      lock.l_whence = SEEK_SET;
      while (fcntl (file, F_SETLKW, &lock) < 0)
        if (errno != EINTR) { error = errno; break; }
      while (!error && offset < length)
        {
          ssize_t n = write (file, record + offset, MIN (length - offset, 0x7fffffff));
          if (n < 0) { if (errno != EINTR) error = errno; }
          else if (!n) error = EIO;
          else offset += n;
        }
      if (close (file) != 0 && !error) error = errno;
    }
  if (error) fprintf (stderr, "Warning: cannot append to BrentFactors.log: %s.\n", strerror (error));
#endif
  return error == 0;
}

void
append_brent_factor (const mpcandi_t *candidate, const mpz_t factor, int prime,
                      int param, mpz_srcptr sigma, double B1, mpz_srcptr B2, FILE *out)
{
  char path[HELPER_PATH_SIZE], *factor_text = NULL, *sigma_text = NULL;
  char *bounds = NULL, *curve = NULL, *record = NULL, *output = NULL;
  char *order = NULL, *factored = NULL;
  const char *group_field = "";
  char *group = NULL;
  int have_sigma = param >= 0 && param <= 3 && sigma && mpz_sgn (sigma) > 0;
#ifdef _WIN32
  const char *newline = "\r\n";
#else
  const char *newline = "\n";
#endif
  if (!candidate->brent_label[0]) return;
  if (ECM_IS_DEFAULT_B2 (B2)) gmp_asprintf (&bounds, "; B1=%.0f, B2=auto", B1);
  else gmp_asprintf (&bounds, "; B1=%.0f, B2=%Zd", B1, B2);
  if (have_sigma) gmp_asprintf (&curve, "; Sigma: %d:%Zd", param, sigma);
  else gmp_asprintf (&curve, "; Sigma: unavailable");

  if (prime && have_sigma && mpz_cmp_ui (factor, 2) > 0 && helper_path (path))
    {
      group_field = "; Group Order: unavailable";
      gmp_asprintf (&factor_text, "%Zd", factor);
      gmp_asprintf (&sigma_text, "%Zd", sigma);
      output = (char *) malloc (HELPER_OUTPUT_SIZE + 1);
      if (output && factor_text && sigma_text &&
          run_helper (path, factor_text, param, sigma_text, output))
        {
          /* Parse the later field first: result_line terminates it in place. */
          factored = result_line (output, "Factored:", "0123456789 *^");
          order = result_line (output, "Group Order:", "0123456789");
          if (order && factored)
            {
              gmp_asprintf (&group, "; Group Order: %s = %s", order, factored);
              if (group) group_field = group;
            }
        }
    }
  if (bounds && curve)
    gmp_asprintf (&record, "%s %Zd%s%s%s%s", candidate->brent_label,
                   factor, curve, bounds, group_field, newline);
  if (record == NULL)
    fprintf (stderr, "Warning: cannot allocate BrentFactors.log record.\n");
  else if (write_record (record) && out)
    {
      fputs ("\nAdding Log Entry --> \"", out);
      fwrite (record, 1, strlen (record) - strlen (newline), out);
      fputs ("\"\n", out);
      fflush (out);
    }
  free_gmp_string (record);
  free_gmp_string (bounds);
  free_gmp_string (curve);
  free_gmp_string (group);
  free_gmp_string (factor_text);
  free_gmp_string (sigma_text);
  free (output);
}
