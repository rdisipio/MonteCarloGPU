// dear emacs, please treat this as -*- C++ -*- 

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define POINTS_PER_BIN    1000

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


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__global__
void RebinGPU( float * x, const float * rebin_map, const int NBINS_OLD, const int NBINS_NEW  )
{
  float * x_tmp = (float*)malloc( NBINS_NEW*sizeof(float) );

  int pos = 0;
  for( int i = 0 ; i < NBINS_OLD ; ++i ) {
    if( rebin_map[i] == 1 ) {
      x_tmp[pos] = x[i];
      pos++;
    }
    else {
      const int   Nbins  = rebin_map[i];
      const float deltax = x[i+1] - x[i];
      const float bw     = deltax / Nbins;

      for( int p = 0 ; p < Nbins; ++p ) {
	x_tmp[pos] = x[i] + p*bw;
	pos++;
      }
    }
  }

  for( int i = 0 ; i < NBINS_NEW ; ++i ) x[i] = x_tmp[i];

  free( x_tmp );
}

//__device__ __host__
void Refine( float * x_new, const float * x_old, const int * split_map, const int NBINS_OLD, const int NBINS_NEW )
{
  int pos = 0;
  for( int i = 0 ; i <= NBINS_OLD ; ++i ) {
    int Nsplit = (int)split_map[i];

    if( Nsplit < 2 ) {
      x_new[pos] = x_old[i];
      // printf("%i : %f\n", pos, x_tmp[pos] );
      pos++;
    }
    else {
      const int Nbins = split_map[i];
      //printf("rebin (%i) -> %i\n", i, Nbins );

      const float deltax = x_old[i+1] - x_old[i];
      const float bw     = deltax / (float)Nbins;

      for( int p = 0 ; p < Nbins; ++p ) {
	x_new[pos] = x_old[i] + p*bw;
	//printf("%i : %f\n", pos, x_tmp[pos] );
	pos++;
      }    
    }
  }

  for( int i = 0 ; i < NBINS_NEW ; ++i ) {
    printf( "  %2i) [ %f, %f ]\n", i, x_new[i], x_new[i+1] );
  }
  printf("DEBUG: Refine done.\n" );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
__device__ __host__
void Refine2( float * x_new, const float * x_old, const int * split_map, const int NBINS_OLD, const int NBINS_NEW )
{
  float pts_per_bin = tot_weight / NBINS_NEW;
}
*/

 //__device__ __host__
void Resize( float * x, const float * x_tmp, const float * h, const float tot_weight, const int NBINS, const int NBINS_NEW )
{
  float pts_per_bin = NBINS_NEW / NBINS;

  x[0] = x_tmp[0]; // left end

  float xold = x_tmp[0];
  float xnew = x_tmp[0];
  float dw   = 0.;
  int   j    = 1;

  for( int k = 0 ; k < ( NBINS_NEW -1) ; ++k ) {
    dw = dw + 1; //h[k];
    xold = xnew;
    xnew = x_tmp[k+1];

    while( dw > pts_per_bin ) {
      dw -= pts_per_bin;
      x[j] = xnew - ( xnew - xold ) * dw;// / h[k];
      j += 1;
    }
  }
  x[NBINS] = x_tmp[NBINS_NEW]; // right end
  
  for( int i = 0 ; i < NBINS ; ++i ) {
    const float deltax = x[i+1] - x[i];
    printf( "  %2i) [ %f, %f ] = [ %f ]\n", i, x[i], x[i+1], deltax );
  }
  printf("DEBUG: Resize done.\n" );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void RebinCPU( float * x, const float * h, const int NBINS )
{

  float tot_weight = 0.;
  for( int i = 0 ; i < NBINS ; ++i ) {
    const float deltax = x[i+1] - x[i];
    tot_weight += deltax * h[i];
  }

  int NBINS_NEW = 0;
  int * split_map = new int[NBINS];
  for( int i = 0 ; i < NBINS ; ++i ) {
    split_map[i]  = 1 + floor( h[i] / tot_weight + 0.5 );
    NBINS_NEW += split_map[i];
  }
  printf( "DEBUG: nbins new = %i\n", NBINS_NEW );

  float * x_tmp = new float[NBINS_NEW+1];

  Refine( x_tmp, x, split_map, NBINS, NBINS_NEW );

  Resize( x, x_tmp, h, tot_weight, NBINS, NBINS_NEW );

  if( !x_tmp ) printf( "ERROR: Invalid pointer to x_tmp.\n" );
  if( !split_map ) printf( "ERROR: Invalid pointer to split_map.\n" );

  delete [] x_tmp;
  delete [] split_map;  

  printf("DEBUG: Rebin done\n" );
}


/////////////////////////////////////////////


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

  int NBINS = ( argc > 1 ) ? atoi(argv[1]) : 10;
  int NITER = ( argc > 2 ) ? atoi(argv[2]) : 4;

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
  cudaMalloc( &d_I,     NBINS * sizeof(float) );
  cudaMalloc( &d_E,     NBINS * sizeof(float) );
  cudaMalloc( &d_m,     NBINS * sizeof(float) );

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


  dim3 dimGrid( ceil( (NBINS+THREADS_PER_BLOCK-1)/(float)THREADS_PER_BLOCK ) );
  //const dim3 dimGrid( NBINS/THREADS_PER_BLOCK );

  cudaEventRecord(start, 0);  
  MakeEdges<<< dimGrid, THREADS_PER_BLOCK >>>( d_x, xmin, xmax, NBINS );
  cudaDeviceSynchronize();  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup bin edges = " << time << " ms" << endl;

  for( int iIter = 0 ; iIter < NITER ; ++iIter ) {
    cout << "INFO: Iteration no. " << iIter << endl;

    int dgrid = ceil( (NBINS+THREADS_PER_BLOCK-1)/(float)THREADS_PER_BLOCK );

    cudaEventRecord(start, 0);  
    //CrudeMonteCarloIntegration<<< dgrid, THREADS_PER_BLOCK >>>( d_x, d_I, d_E, d_m, devStates, NBINS );
    CrudeMonteCarloIntegration<<< NBINS, 1 >>>( d_x, d_I, d_E, d_m, devStates, NBINS );
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
    //float mtot = 0.;
    for( int i = 0 ; i < NBINS ; ++i ) {
      const float deltax = h_x[i+1] - h_x[i];
      const float fMC    = h_I[i] / (float)( NBINS * POINTS_PER_BIN );
      const float eMC    = h_E[i] / (float)( NBINS * POINTS_PER_BIN );
      const float dI     = sqrt( ( eMC - fMC*fMC ) / ( POINTS_PER_BIN-1) );
       
      
      cout << "i = " << i << " Dx = [ " << h_x[i] << ", " << h_x[i+1]
	   << " ] = [ " << deltax <<  "] "
	   << " I = " << h_I[i] 
	   << "  \\int{f(x)} = " << fMC  
	   << " dI = " << dI
	   << " m_i = " << h_m[i]
	   << endl;
      

      //mtot += h_m[i] * deltax;
      I    += h_I[i];
      E    += h_E[i];
    }
    I /= ( NBINS * POINTS_PER_BIN );
    E /= ( NBINS * POINTS_PER_BIN );
    float unc      = sqrt( ( E - I*I ) / (NBINS*POINTS_PER_BIN) );
  
    cout << "INFO: \\int{f(x)} = " <<  I << " \\pm " << unc  << endl;
    // cout << "INFO: m_tot = " << mtot << endl;

    if( NITER > 1 ) {
      RebinCPU( h_x, h_m, NBINS );

      cudaMemcpy( d_x, h_x, (NBINS+1) * sizeof(float), cudaMemcpyHostToDevice );
      //Rebin<<<1,1>>>( d_x, h_rebin_map, NBINS, NBINS_NEW );
    }
  }

  cout << "INFO: Finished. Flushing memory." << endl;
  // flush memory
  cudaFree( d_x );
  cudaFree( d_I );
  cudaFree( d_E );
  cudaFree( d_m );
  cudaFree( devStates );
  cout << "DEBUG: GPU memory freed" << endl;

  delete [] h_x;
  delete [] h_I;
  delete [] h_E;
  delete [] h_m;
  cout << "DEBUG: CPU memory freed" << endl;
 
  return success;
}
