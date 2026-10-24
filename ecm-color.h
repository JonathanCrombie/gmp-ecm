/* Private CLI presentation helpers. Library users keep plain output unless
   they explicitly initialize color support. File data is never styled. */
#ifndef ECM_COLOR_H
#define ECM_COLOR_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum ecm_color_style {
  ECM_COLOR_PLAIN, ECM_COLOR_HEADING, ECM_COLOR_FACTOR,
  ECM_COLOR_WARNING, ECM_COLOR_ERROR, ECM_COLOR_DIAGNOSTIC
};

/* Return zero for an invalid mode. Call only before starting worker threads.
   A nonempty NO_COLOR disables auto; explicit always overrides it. */
int ecm_color_init (const char *mode);
int ecm_color_fprintf (FILE *, enum ecm_color_style, const char *, ...);
enum ecm_color_style ecm_color_message_style (const char *);
/* Bind this thread's internal spool files to the final stdout/stderr policy.
   These are console buffers, not save files or user-supplied library streams. */
void ecm_color_bind (FILE *output, FILE *errors);

#ifdef __cplusplus
}
#endif
#endif
