#include <cmath>

double BlackBoxFunction(const double x) {
  return 1.0/sqrt(x);
}

double InverseDerivative(const double x) {
  return 2.0*sqrt(x);
}
