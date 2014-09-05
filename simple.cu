// dear emacs, please treat this as -*- C++ -*- 

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 128
#define POINTS_PER_BIN    100

using namespace std;

__device__ __host__
float func( const float x )
{
  return x*x;
}


__global__
void setup_random( curandState * state )
{
   int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;  
   curand_init(1234, id, 0, &state[id]);
}


__global__
void MakeEdges( float * d_x, const float xmin, const float xmax, const int NBINS )
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if( i >= (NBINS+1) ) return;

  d_x[i] = xmin + ( xmax - xmin ) * i / NBINS;
}


__global__
void CrudeMonteCarloIntegration( const float * x, 
				 float * I, float * E, float * m,
				  curandState * state,
				  const int NBINS )
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if( i >= NBINS ) return;
  
  curandState localState = state[i];

  const float deltax = x[i+1] - x[i];

  float sum = 0.;
  float err = 0.;
  float fbar = 0.;

  for( int k = 0 ; k < POINTS_PER_BIN ; ++k ) {
    // pick a random x0 in the bin interval
    const float x0 = x[i] + deltax * curand_uniform( &localState ); 
    const float y  = func(x0);
    sum += y;
    err += y*y;
    fbar += fabs(y);
  }
  m[i] = fbar * deltax;
  I[i] = sum;
  E[i] = err;
}



int main( int argc, char ** argv )
{
  int success = 0;

  const int NBINS = ( argc > 1 ) ? atoi(argv[1]) : 10;

  float xmin = 0.0;
  float xmax = 1.0;

  float * h_x = new float[NBINS+1];
  float * h_I = new float[NBINS];
  float * h_E = new float[NBINS];
  float * h_m = new float[NBINS];

  float * d_x;
  float * d_I;
  float * d_E;
  float * d_m;

  cudaMalloc( &d_x, (NBINS+1) * sizeof(float) );
  cudaMalloc( &d_I, NBINS * sizeof(float) );
  cudaMalloc( &d_E, NBINS * sizeof(float) );
  cudaMalloc( &d_m, NBINS * sizeof(float) );

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // setup random generator
  curandState *devStates;
  cudaMalloc((void **)&devStates , THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(curandState) );

  cudaEventRecord(start, 0);  
  setup_random<<< THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( devStates );
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup random numbers = " << time << " ms" << endl;


  const dim3 dimGrid( ceil( (NBINS+THREADS_PER_BLOCK-1)/(float)THREADS_PER_BLOCK ) );
  //const dim3 dimGrid( NBINS/THREADS_PER_BLOCK );

  cudaEventRecord(start, 0);  
  MakeEdges<<< dimGrid, THREADS_PER_BLOCK >>>( d_x, xmin, xmax, NBINS );
  cudaDeviceSynchronize();  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup bin edges = " << time << " ms" << endl;


  cudaEventRecord(start, 0);  
  CrudeMonteCarloIntegration<<< dimGrid, THREADS_PER_BLOCK >>>( d_x, d_I, d_E, d_m,
								devStates, NBINS );
  cudaDeviceSynchronize();  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for kernel = " << time << " ms" << endl;


  cudaMemcpy( h_x, d_x, (NBINS+1) * sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( h_I, d_I,     NBINS * sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( h_E, d_E,     NBINS * sizeof(float), cudaMemcpyDeviceToHost );
  cudaMemcpy( h_m, d_m,     NBINS * sizeof(float), cudaMemcpyDeviceToHost );
  //cudaDeviceSynchronize();  

  // printout
  float I = 0.;
  float E = 0.;
  float mtot = 0.;
  for( int i = 0 ; i < NBINS ; ++i ) {
    const float deltax = h_x[i+1] - h_x[i];
    const float fMC    = h_I[i] / (float)( NBINS * POINTS_PER_BIN );
    const float eMC    = h_E[i] / (float)( NBINS * POINTS_PER_BIN );
    const float dI     = sqrt( ( eMC - fMC*fMC ) / ( POINTS_PER_BIN-1) );
       
    cout << "i = " << i << " Dx = [ " << h_x[i] << ", " << h_x[i+1] 
    	 << " ]   I = " << h_I[i] 
	 << "  \\int{f(x)} = " << fMC  
	 << " dI = " << dI
	 << " m_i = " << h_m[i]
	 << endl;

    mtot += h_m[i] * deltax;
    I    += h_I[i];
    E    += h_E[i];
  }
  I /= ( NBINS * POINTS_PER_BIN );
  E /= ( NBINS * POINTS_PER_BIN );
  float unc      = sqrt( ( E - I*I ) / (NBINS*POINTS_PER_BIN) );
  
  cout << "INFO: \\int{f(x)} = " <<  I << " \\pm " << unc  << endl;
  cout << "INFO: m_tot = " << mtot << endl;
  
  for( int i = 0 ; i < NBINS ; ++i ) {
    int m_i = 1 + floor( h_m[i] / mtot + 0.5 );
    cout << "INFO: subdivisions for bin " << i << " = " << m_i << endl;
  }

  cudaFree( d_x );
  cudaFree( d_I );
  cudaFree( d_E );
  cudaFree( devStates );

  delete [] h_x;
  delete [] h_I;
  delete [] h_E;

  return success;
}
