#include "L.h"

//function to compute wave velocity w.r.t position x
#pragma omp declare simd
float L(const float alpha, const float phase, const float x) {
  return expf(-alpha*(x-phase)*(x-phase));
}