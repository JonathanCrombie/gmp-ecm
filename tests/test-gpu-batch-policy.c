/* Deterministic admission/feedback tests, no CUDA device required. */
#include "gpu_batch_policy.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  unsigned i;
  double duty;
  if(ECM_GPU_TARGET_UTILIZATION==100) {
    assert(ecm_gpu_initial_duty(-1)==1.0);
    assert(ecm_gpu_initial_duty(101)==1.0);
    for(i=0; i<=100; ++i) {
      duty=ecm_gpu_initial_duty((int)i);
      assert(duty==1.0);
      assert(ecm_gpu_adjust_duty(0.05, i)==1.0);
      assert(ecm_gpu_adjust_duty(duty, i)==1.0);
      assert(ecm_gpu_batch_inputs(112, 64, duty)==7168);
      assert(ecm_gpu_batch_inputs(84, 64, duty)==5376);
      assert(ecm_gpu_batch_inputs(112, 32, duty)==3584);
      assert(ecm_gpu_batch_inputs(84, 32, duty)==2688);
    }
    assert(ecm_gpu_adjust_duty(0.05, -1)==1.0);
    assert(ecm_gpu_clamp_duty(1.1)==1.0);
    assert(ecm_gpu_batch_inputs(1, 8, 1.0)==8);
    assert(ecm_gpu_batch_inputs(100000, 64, 1.0)==ECM_GPU_AUTO_MAX_INPUTS);
    puts("PASS GPU batch policy: 100% duty, full waves, unavailable telemetry and bounds");
    return 0;
  }
  assert(ECM_GPU_TARGET_UTILIZATION==90); /* Optional lower-target regression build. */
  assert(ecm_gpu_initial_duty(-1)==0.90);
  assert(ecm_gpu_initial_duty(101)==0.90);
  assert(ecm_gpu_initial_duty(0)>0.89);
  assert(ecm_gpu_initial_duty(50)>0.79 && ecm_gpu_initial_duty(50)<0.81);
  assert(ecm_gpu_initial_duty(90)==0.05);
  assert(ecm_gpu_initial_duty(100)==0.05);
  assert(ecm_gpu_adjust_duty(0.5, 90)==0.5);
  assert(ecm_gpu_adjust_duty(0.5, 85)>0.5);
  assert(ecm_gpu_adjust_duty(0.5, 95)<0.5);
  assert(ecm_gpu_adjust_duty(0.5, -1)==0.5);
  duty=0.90;
  for(i=0; i<100; ++i) duty=ecm_gpu_adjust_duty(duty, 100);
  assert(duty==0.05);
  assert(ecm_gpu_batch_inputs(112, 64, duty)==320);
  for(i=0; i<100; ++i) duty=ecm_gpu_adjust_duty(duty, 0);
  assert(duty==0.90);
  assert(ecm_gpu_batch_inputs(112, 64, duty)==6400);
  assert(ecm_gpu_batch_inputs(1, 8, 0.05)==8);
  assert(ecm_gpu_batch_inputs(100000, 64, 0.9)==ECM_GPU_AUTO_MAX_INPUTS);
  for(i=1; i<=1000; ++i) {
    unsigned n=ecm_gpu_batch_inputs(i, 32, 0.6);
    assert(n>=32 && n%32==0 && n<=i*32);
  }
  /* A simple independent-load model: background uses 50% of available time.
     The controller should approach 90% total rather than treating its own
     kernel utilization as an external load and collapsing the batch size. */
  duty=0.90;
  for(i=0; i<200; ++i) duty=ecm_gpu_adjust_duty(duty, 50+50*duty);
  assert(50+50*duty>=88 && 50+50*duty<=92);
  puts("PASS GPU batch policy: availability, feedback, bounds and block alignment");
  return 0;
}
