/* GPU admission and pacing policy, independent of CUDA/NVML. GPL-3.0-or-later. */
#ifndef ECM_GPU_BATCH_POLICY_H
#define ECM_GPU_BATCH_POLICY_H

#ifndef ECM_GPU_TARGET_UTILIZATION
#define ECM_GPU_TARGET_UTILIZATION 100
#endif
#define ECM_GPU_MAX_DUTY (ECM_GPU_TARGET_UTILIZATION / 100.0)
#define ECM_GPU_AUTO_MAX_INPUTS 65536U

static double ecm_gpu_clamp_duty(double duty) {
  return duty < 0.05 ? 0.05 : duty > ECM_GPU_MAX_DUTY ? ECM_GPU_MAX_DUTY : duty;
}

/* This first sample is taken before our first kernel. Later samples include
   our work and must feed the controller, not be mistaken for external load. */
static double ecm_gpu_initial_duty(int utilization) {
  /* At 100%, reserve no idle time, including when other work already has the
     device fully busy. This also avoids 0/0 and fractional rounding that
     would otherwise drop an entire block from a nominally full wave. */
  if(ECM_GPU_TARGET_UTILIZATION==100) return 1.0;
  if(utilization < 0 || utilization > 100) return ECM_GPU_MAX_DUTY;
  if(utilization==100) return 0.05;
  return ecm_gpu_clamp_duty((ECM_GPU_TARGET_UTILIZATION-utilization)/(100.0-utilization));
}

static double ecm_gpu_adjust_duty(double duty, double utilization) {
  if(ECM_GPU_TARGET_UTILIZATION==100) return 1.0;
  if(utilization > ECM_GPU_TARGET_UTILIZATION+2.0 && utilization<=100.0)
    duty *= ECM_GPU_TARGET_UTILIZATION/utilization;
  else if(utilization >= 0.0 && utilization < ECM_GPU_TARGET_UTILIZATION-2.0) duty += 0.02;
  return ecm_gpu_clamp_duty(duty);
}

/* One estimated resident wave, rounded to whole blocks. A block is the
   smallest useful admission unit, even on a very small or very busy GPU. */
static unsigned ecm_gpu_batch_inputs(unsigned resident_blocks,
                                     unsigned inputs_per_block, double duty) {
  unsigned blocks = (unsigned)(resident_blocks * ecm_gpu_clamp_duty(duty));
  unsigned limit;
  if(!inputs_per_block) return 1;
  if(!blocks) blocks = 1;
  limit = ECM_GPU_AUTO_MAX_INPUTS / inputs_per_block;
  if(blocks > limit) blocks = limit;
  return blocks * inputs_per_block;
}
#endif
