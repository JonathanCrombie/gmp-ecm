/* Command-line batching; kept separate from the single-number library API. */
#ifndef PM1_GPU_H
#define PM1_GPU_H
#include "ecm-ecm.h"
#ifdef WITH_GPU
int pm1_gpu_run(FILE *input, FILE *resume, ecm_params params,
                double B1, double B1done, unsigned batch_size, int primetest,
                int specific_x0, mpq_t x0, mpgocandi_t *go,
                char *savefile, char *checkpoint, int timestamp);
#endif
#endif
