#include "library.h"
#include <mpi.h>

double ComputeIntegral(const int n, const double a, const double b, const int rank, const int nRanks) {

  const int stepsPerProcess = double(n-1)/double(nRanks);
  const int iStart = int( stepsPerProcess*rank );
  const int iEnd = int( stepsPerProcess*(rank + 1) );

  const double dx = (b - a)/n;
  double I_partial = 0.0;

#pragma omp parallel for simd reduction(+: I_partial)
  for (int i = iStart; i < iEnd; i++) {

    const double xip12 = a + dx*(double(i) + 0.5);
    const double yip12 = BlackBoxFunction(xip12);
    const double dI = yip12*dx;
    I_partial += dI;

  }

  double I = 0.0;
  MPI_Allreduce(&I_partial, &I, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return I;
}
