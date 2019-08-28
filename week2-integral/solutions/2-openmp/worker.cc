#include "library.h"

double ComputeIntegral(const int n, const double a, const double b) {

  const double dx = (b - a)/n;
  double I = 0.0;

#pragma omp parallel for simd reduction(+: I)
  for (int i = 0; i < n; i++) {

    const double xip12 = a + dx*(double(i) + 0.5);
    const double yip12 = BlackBoxFunction(xip12);
    const double dI = yip12*dx;
    I += dI;

  }

  return I;
}
