#include "library.h"

double ComputeIntegral(const int n, const double a, const double b) {

  const double dx = (b - a)/n;
  double I = 0.0;

#pragma omp parallel
  {
    double I_th = 0.0;

#pragma omp for
    for (int ii = 0; ii < n; ii+=1000)
#pragma omp simd reduction(+: I_th)
      for (int i = ii; i < ii + 1000; i++) {

	const double xip12 = a + dx*(double(i) + 0.5);
	const double yip12 = BlackBoxFunction(xip12);
	const double dI = yip12*dx;
	I_th += dI;

      }

#pragma omp atomic
    I += I_th;
  }
  
  return I;
}
