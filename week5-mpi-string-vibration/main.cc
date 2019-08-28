#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <assert.h>
#include "L.h"

// Finite difference method for stings
// d_(x, t+1) = L(x)*(d_(x+dx, t) + d_(x-dx, t))
//              + 2.0f*(1.0f-L(x))*(d_(x,t))
//              - d_(x, t-1) 

float * simulate(const float alpha, const long n_segments, const int n_steps, float *d_buf1, float *d_buf2, const int rank, const int world_size, const long segments_per_process);

// Do not modify
//reference simulate function to verify data
float * simulate_ref(const float alpha, const long n_segments, const int n_steps, float *d_buf1, float *d_buf2, const int rank, const int world_size, const long segments_per_process) {

  float* d_t  = d_buf1; // buffer for d(*, t)
  float* d_t1 = d_buf2; // buffer for d(*, t+1)

  const int start_segment = segments_per_process*((long)rank)   +1L;
  const int last_segment  = segments_per_process*((long)rank+1L)+1L;

  const float dx = 1.0f/(float)n_segments;
  const float phase = (float)n_segments/2.0f;
  
  for(int t = 0; t < n_steps; t++) {
#pragma omp parallel for simd
    for(long i = start_segment; i < last_segment; i++) {
      const float L_x = L(alpha,phase,i*dx);
      d_t1[i] = L_x*(d_t[i+1] + d_t[i-1])
                +2.0f*(1.0f-L_x)*(d_t[i]) 
                - d_t1[i]; // The algorithm calls for d(i, t-1) here, but that is currently contained in d_t1
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &d_t1[1], segments_per_process, MPI_FLOAT, MPI_COMM_WORLD);
    float* temp = d_t1; d_t1 = d_t; d_t=temp; // swap buffers
  }
  return d_t;
}
void initialize_buffers(const float alpha, const long n_segments, float *d_buf1, float *d_buf2) {
  const float dx = 1.0f/(float)n_segments;
  const float phase = (float)n_segments/2.0f;
#pragma omp parallel for
  for(long i =0; i < n_segments; i++)
    d_buf1[i] = 100.0*sinf(3.14159*(float)i*dx);
  d_buf1[0] = d_buf1[n_segments-1] = d_buf2[0] = d_buf2[n_segments-1] = 0.0f;
  for(long i = 1; i < n_segments-1; i++) 
    // d_1 = d_0 + v_0*dt + 0.5*a*dt^2
    d_buf2[i] = L(alpha,phase,i*dx)/2.0f*(d_buf1[i+1] + d_buf1[i-1]) + (1.0f-L(alpha,phase,i*dx))*(d_buf1[i]); 
}

int main(int argc, char** argv) {
  int ret = MPI_Init(&argc,&argv);
  if (ret != MPI_SUCCESS) {
    printf("error: could not initialize MPI\n");
    MPI_Abort(MPI_COMM_WORLD, ret);
  }
  float alpha;
  if (argc < 2) {
    alpha = 0.2;
  } else {
    alpha = atof(argv[1]);
  }
  
  int world_size, rank;
  MPI_Status stat;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  const int n_steps = 1<<6;
  const long n_segments = (1L<<25)+2L;
  assert((n_segments-2L)%world_size == 0); // This will make MPI gather much easier to work with
  const long segments_per_process = (n_segments-2)/(long)world_size;

  //two buffers to store current and next position
  float *d_buf1 = (float *) _mm_malloc(sizeof(float)*n_segments, 4096);
  float *d_buf2 = (float *) _mm_malloc(sizeof(float)*n_segments, 4096);

  // Getting verificatiobn data
  float *d_ref = (float *) _mm_malloc(sizeof(float)*n_segments, 4096);
  if(rank == 0) {
    initialize_buffers(alpha, n_segments, d_buf1, d_buf2);
  } 
  
  MPI_Bcast(d_buf1, n_segments, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(d_buf2, n_segments, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  //compute reference data
  float *d_ref_temp = simulate_ref(alpha, n_segments, n_steps, d_buf1, d_buf2, rank, world_size, segments_per_process);
  
  if(rank == 0) {
#pragma omp parallel for 
    for(long i = 0; i < n_segments; i++)
      d_ref[i] = d_ref_temp[i];
  }
  
  //initialize buffers in rank0 and broadcast them to all the processes
  if(rank == 0) {
    initialize_buffers(alpha, n_segments, d_buf1, d_buf2);
  }
  MPI_Bcast(d_buf1, n_segments, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(d_buf2, n_segments, MPI_FLOAT, 0, MPI_COMM_WORLD);
  
  
  //compute using the function in worker.cc and get the timing
  const double t0 = omp_get_wtime();
  float *d_final = simulate(alpha, n_segments, n_steps, d_buf1, d_buf2, rank, world_size, segments_per_process);
  const double t1 = omp_get_wtime();

  //verify computed data with the reference data in rank 0 
  if(rank == 0) {
    bool within_tolerance = true;
#pragma omp parallel for reduction(&: within_tolerance)
    for(long i = 0; i < n_segments; i++)
      within_tolerance &= ((d_ref[i] - d_final[i])*(d_ref[i] - d_final[i])) < 1.0e-6;;
    
    if(within_tolerance) {
      // Printing performance as measured on node 1
      printf("Time: %f\n", t1-t0);
    } else {
      // Verification failed
      printf("Error: verification failed %f\n", t1-t0);

    }
  }
  MPI_Finalize();
}