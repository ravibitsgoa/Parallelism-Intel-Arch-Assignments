#include <cstdio>
#include <cstdlib>
#include <mkl.h>
#include <omp.h>
#include <hbwmalloc.h>

void runFFTs( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle);

// Do not modify.
//reference funtion 
void runFFTs_ref( const size_t fft_size, const size_t num_fft, MKL_Complex8 *data, DFTI_DESCRIPTOR_HANDLE *fftHandle) {
  for(size_t i = 0; i < num_fft; i++) {
    DftiComputeForward (*fftHandle, &data[i*fft_size]);
  }
}

int main() {
  const size_t fft_size = 1L<<27;
  const size_t num_fft = 32L;
  MKL_Complex8 *data = (MKL_Complex8 *) _mm_malloc(sizeof(MKL_Complex8)*num_fft*fft_size, 4096);
  MKL_Complex8 *ref_data = (MKL_Complex8 *) _mm_malloc(sizeof(MKL_Complex8)*num_fft*fft_size, 4096);

//iniitialize data array and copy it to ref_data array   
#pragma omp parallel
  {
    long random_seed = (long)(omp_get_wtime()*1000.0*omp_get_thread_num()) % 1000L;
    VSLStreamStatePtr rnStream;
    //initialize random stream
    vslNewStream( &rnStream, VSL_BRNG_MT19937, random_seed);
#pragma omp for 
    for(size_t i = 0; i < num_fft; i++) {
      //Intel MKL Rnadom stream generation function
      vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, rnStream, 2*fft_size, (float *) &data[i*fft_size], -1.0, 1.0);
    }
//copy data to ref_data
#pragma omp for
    for(long i = 0; i < (fft_size+2)*num_fft; i++) {
      ref_data[i].real = data[i].real;
      ref_data[i].imag = data[i].imag;
    }
  }


  DFTI_DESCRIPTOR_HANDLE* fftHandle = new DFTI_DESCRIPTOR_HANDLE;
  DftiCreateDescriptor(fftHandle, DFTI_SINGLE, DFTI_COMPLEX, 1, (MKL_LONG) fft_size);
  DftiCommitDescriptor (*fftHandle);
  //compute FFT using refernce function
  runFFTs_ref(fft_size, num_fft, ref_data, fftHandle);

  //compute and time FFT using function defined in worker.cc
  const double t0 = omp_get_wtime();
  runFFTs(fft_size, num_fft, data, fftHandle);
  const double t1 = omp_get_wtime();

  //verify the comuted FFT data with the reference FFT data
  bool within_tolerance = true; 
#pragma omp parallel for reduction(&: within_tolerance)
  for(long i = 0; i < num_fft; i++) {
    for(long j = 0; j < fft_size; j++) {
      within_tolerance &= ((data[i*fft_size+j].real-ref_data[i*fft_size+j].real)
                          *(data[i*fft_size+j].real-ref_data[i*fft_size+j].real)
                          +(data[i*fft_size+j].imag-ref_data[i*fft_size+j].imag)
                          *(data[i*fft_size+j].imag-ref_data[i*fft_size+j].imag))
                          < 1.0e-6;
    }
  }
  if(within_tolerance) {
    // Printing performance
    printf("Time: %f\n", t1-t0);
  } else {
    // Verification failed
    printf("Error: Verification failed\n");
  }
  DftiFreeDescriptor (fftHandle); 
  _mm_free(ref_data);
  _mm_free(data);
}