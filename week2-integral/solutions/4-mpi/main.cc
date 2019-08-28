#include <cmath>
#include <cstdio>
#include <omp.h>
#include <mpi.h>
#include "library.h"

const int nTrials = 10;
const int skipTrials = 3; // Skip first iteration as warm-up

double ComputeIntegral(const int n, const double a, const double b, const int rank, const int nRanks);

double Stats(double & x, double & dx) {
  x  /= (double)(nTrials - skipTrials);
  dx  = sqrt(dx/double(nTrials - skipTrials) - x*x);
}

double Accuracy(const double I, const double a, const double b) {
  const double I0 = InverseDerivative(b) - InverseDerivative(a);
  const double error = (I - I0)*(I - I0);
  const double norm  = (I + I0)*(I + I0);
  return error/norm;
}

int main(int argc, char** argv) {

  MPI_Init(&argc, &argv);

  int rank, nRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  const int n = 1000*1000*1000;

  if (rank == 0) {
  printf("\n\033[1mNumerical integration with n=%d\033[0m\n", n);
  printf("\033[1m%5s %15s %15s %15s\033[0m\n", "Step", "Time, ms", "GSteps/s", "Accuracy"); fflush(stdout);
  }

  double t, dt, f, df;

  for (int iTrial = 1; iTrial <= nTrials; iTrial++) {

    const double a = double(iTrial - 1);
    const double b = double(iTrial + 1);

    MPI_Barrier(MPI_COMM_WORLD);

    const double t0 = omp_get_wtime();
    const double I = ComputeIntegral(n, a, b, rank, nRanks);
    const double t1 = omp_get_wtime();

    const double ts   = t1-t0; // time in seconds
    const double tms  = ts*1.0e3; // time in milliseconds
    const double fpps = double(n)*1e-9/ts; // performance in steps/s

    if (iTrial > skipTrials) { // Collect statistics
      t  += tms; 
      dt += tms*tms;
      f  += fpps;
      df += fpps*fpps;
    }

    const double acc = Accuracy(I, a, b);

    if (rank == 0)
    // Output performance
    printf("%5d %15.3f %15.3f %15.3e%s\n", 
	   iTrial, tms, fpps, acc, (iTrial<=skipTrials?"*":""));
    fflush(stdout);
  }

  Stats(t, dt);  
  Stats(f, df);  

  if (rank == 0) {
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s\033[0m\n%8s   \033[42m%8.1f+-%.1f\033[0m   \033[42m%8.1f+-%.1f\033[0m\n",
	 "Average performance:", "", t, dt, f, df);
  printf("-----------------------------------------------------\n");
  printf("* - warm-up, not included in average\n\n");
  }

  MPI_Finalize();

}
