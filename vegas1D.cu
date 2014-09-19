// dear emacs, please treat this as -*- C++ -*- 

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <curand_kernel.h>

#define THREADS_PER_BLOCK 256
#define POINTS_PER_BIN    ( THREADS_PER_BLOCK * THREADS_PER_BLOCK )

using namespace std;

__device__ __host__
float func( const float x )
{
  //return exp( -(x-0.5)*(x-0.5)/0.1 );  // 0.546292
  return x*x; //0.333333
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
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;  

  if( tid >= (NBINS+1) ) return;

  d_x[tid] = xmin + ( xmax - xmin ) * tid / NBINS;
}


void DumpEdges( const float * x, const int NBINS )
{
  cout << "DEBUG: bin edges:" << endl;
    for( unsigned int b = 0 ; b <= NBINS ; ++b ) {
      cout << " " << x[b];
    }
    cout << endl;
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ __host__
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

  /*
  for( int i = 0 ; i < NBINS_NEW ; ++i ) {
    float deltax = x_new[i+1] - x_new[i];
    printf( "  %2i) [ %f, %f ] = [ %f ] \n", i, x_new[i], x_new[i+1], deltax  );
  }
  */

  printf("DEBUG: Refine done.\n" );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ __host__
void Resize( float * x, const float * x_tmp, const float * h, const float tot_weight, const int NBINS, const int NBINS_NEW )
{
  float pts_per_bin = (float)NBINS_NEW / (float)NBINS;

  x[0] = x_tmp[0]; // left end

  float xold = x_tmp[0];
  float xnew = x_tmp[0];
  float dw   = 0.;
  int   j    = 1;

  for( int k = 0 ; k <= ( NBINS_NEW ) ; ++k ) {
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

  // const float xrange = x[NBINS] - x[0];
  float tot_weight = 0.;
  for( int i = 0 ; i < NBINS ; ++i ) {
    const float deltax = x[i+1] - x[i];
    //const float w = deltax / xrange;
    tot_weight += deltax * h[i];
  }

  int NBINS_NEW = 0;
  int * split_map;
  cudaMallocManaged( &split_map, NBINS*sizeof(int) );

  for( int i = 0 ; i < NBINS ; ++i ) {
    split_map[i]  = 1 + floor( h[i] / tot_weight + 0.5 );
    NBINS_NEW += split_map[i];
  }
  printf( "DEBUG: nbins new = %i\n", NBINS_NEW );

  float * x_tmp; //= new float[NBINS_NEW+1];
  cudaMallocManaged( &x_tmp, (NBINS+1) * sizeof(float) );

  Refine( x_tmp, x, split_map, NBINS, NBINS_NEW );

  // check convergence here

  Resize( x, x_tmp, h, tot_weight, NBINS, NBINS_NEW );
  
  cudaFree( split_map );  
  cudaFree( x_tmp );
  
  printf("DEBUG: Rebin done\n" );
}


/////////////////////////////////////////////


__global__
void DoSampling(  const float * x, 
		 float * sum_f, float * sum_f_sq, float * sum_f_abs,
		 curandState * state,
		  const int NBINS )
{
  /*
    N bins
    M threads per block (1 block = 1 bin)
    N * M threads in total
   */

  __shared__ float s_f[THREADS_PER_BLOCK];
  __shared__ float s_f_sq[THREADS_PER_BLOCK];
  __shared__ float s_f_abs[THREADS_PER_BLOCK];

  unsigned int ibin = blockIdx.x;
  unsigned int tid  = threadIdx.x; 
  unsigned int I    = blockIdx.x*blockDim.x + threadIdx.x;
  
  if( ibin >= NBINS ) return;

  const float deltax = x[ibin+1] - x[ibin];

  curandState localState = state[I];
  
  float f     = 0.;
  float f_sq  = 0.;
  float f_abs = 0.;
  
  for( int k = 0 ; k < POINTS_PER_BIN/THREADS_PER_BLOCK ; ++k ) {

    const float x0 = x[ibin] + deltax * curand_uniform( &localState ); 	

    const float y  = func( x0 );

    f     += y;
    f_sq  += y*y;
    f_abs += fabs(y);
  }
 
  s_f[tid]     = f;
  s_f_sq[tid]  = f_sq;
  s_f_abs[tid] = f_abs;
  
  __syncthreads();

  // do reduction
  
  /*
  for(unsigned int s = 1 ; s < blockDim.x ; s *= 2) {
    if( tid % (2*s) == 0 ) {
      s_f[tid]     += s_f[tid+s];
      s_f_sq[tid]  += s_f_sq[tid+s];
      s_f_abs[tid] += s_f_abs[tid+s];
    }
    __syncthreads();
  }
  */

  
  for( unsigned int s = blockDim.x/2 ; s > 0 ; s >>= 1 ) {
    if( tid < s ) {
      s_f[tid]     += s_f[tid+s];
      s_f_sq[tid]  += s_f_sq[tid+s];
      s_f_abs[tid] += s_f_abs[tid+s];
    }
    __syncthreads();
  }
 

  /*
  for( unsigned int s = blockDim.x/2 ; s > 32 ; s >>= 1 ) {
    if( tid < s ) {
      s_f[tid]     += s_f[tid+s];
      s_f_sq[tid]  += s_f_sq[tid+s];
      s_f_abs[tid] += s_f_abs[tid+s];
    }
    __syncthreads();
  }
  if( tid < 32 ) {
    s_f[tid]     += s_f[tid + 32];
    s_f[tid]     += s_f[tid + 16];
    s_f[tid]     += s_f[tid +  8];
    s_f[tid]     += s_f[tid +  4];
    s_f[tid]     += s_f[tid +  2];
    s_f[tid]     += s_f[tid +  1];

    s_f_sq[tid]  += s_f_sq[tid + 32];
    s_f_sq[tid]  += s_f_sq[tid + 16];
    s_f_sq[tid]  += s_f_sq[tid +  8];
    s_f_sq[tid]  += s_f_sq[tid +  4];
    s_f_sq[tid]  += s_f_sq[tid +  2];
    s_f_sq[tid]  += s_f_sq[tid +  1];

    s_f_abs[tid] += s_f_abs[tid + 32];
    s_f_abs[tid] += s_f_abs[tid + 16];
    s_f_abs[tid] += s_f_abs[tid +  8];
    s_f_abs[tid] += s_f_abs[tid +  4];
    s_f_abs[tid] += s_f_abs[tid +  2];
    s_f_abs[tid] += s_f_abs[tid +  1];
  }
  */

  // write result to global memory
  if( tid == 0 ) {
    sum_f[ibin]     = s_f[0];
    sum_f_sq[ibin]  = s_f_sq[0];
    sum_f_abs[ibin] = s_f_abs[0];
  }
}


int main( int argc, char ** argv )
{
  int success = 0;

  int NBINS = ( argc > 1 ) ? atoi(argv[1]) : 10;
  int NITER = ( argc > 2 ) ? atoi(argv[2]) : 4;

  const float xmin = 0.;
  const float xmax = 1.;
  
  float * d_x;
  float * d_I;
  float * d_E;
  float * d_m;

  cudaMallocManaged( &d_x, (NBINS+1) * sizeof(float) );
  cudaMallocManaged( &d_I, NBINS * sizeof(float) );
  cudaMallocManaged( &d_E, NBINS * sizeof(float) );
  cudaMallocManaged( &d_m, NBINS * sizeof(float) );

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // setup random generator
  curandState *devStates;
  cudaMallocManaged( (void **)&devStates ,  THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(curandState) );

  cudaEventRecord(start, 0);  
  setup_random<<< THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( devStates );
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup random numbers = " << time << " ms" << endl;

  cudaEventRecord(start, 0);  
  MakeEdges<<< (NBINS+1), 1 >>>( d_x, xmin, xmax, NBINS );
  cudaDeviceSynchronize();  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup bin edges = " << time << " ms" << endl;
  DumpEdges( d_x, NBINS );

  float Ibest    = 0.;
  float * Iiter = new float[NITER];
  float * Viter = new float[NITER];
  float inv_var_tot  = 0.;
  float chi2     = 0.;
  for( int iIter = 0 ; iIter < NITER ; ++iIter ) {
    cout << "INFO: Iteration no. " << iIter << endl;

    //int dgrid = ceil( (NBINS+THREADS_PER_BLOCK-1)/(float)THREADS_PER_BLOCK );

    cudaEventRecord(start, 0);  
    DoSampling<<< NBINS, THREADS_PER_BLOCK >>>( d_x, d_I, d_E, d_m, devStates, NBINS );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for kernel = " << time << " ms" << endl;

    cudaDeviceSynchronize();  

    // printout
    float Vol = d_x[NBINS] - d_x[0];
    float I = 0.;
    float E = 0.;
    //float mtot = 0.;
    for( int i = 0 ; i < NBINS ; ++i ) {
      const float deltax = d_x[i+1] - d_x[i];
      const float fMC    = d_I[i] / (float)( NBINS );
      const float eMC    = d_E[i] / (float)( NBINS );
      const float dI     = sqrt( ( eMC - fMC*fMC ) / ( NBINS * POINTS_PER_BIN-1) );
      
      cout << "i = " << i << " Dx = [ " << d_x[i] << ", " << d_x[i+1]
	   << " ] = [ " << deltax <<  "] "
	   << " I = " << d_I[i] 
	   << "  \\int{f(x)} = " << fMC  
	   << " dI = " << dI
	   << " m_i = " << d_m[i]
	   << endl;
      
      //const int jacobian = Vol * NBINS / POINTS_PER_BIN;
      float w = NBINS * ( deltax / Vol) / POINTS_PER_BIN;

      I    += w*d_I[i];
      E    += w*d_E[i];
    }
    I /= ( NBINS );
    E /= ( NBINS );
    float var      = ( E - I*I ) / (NBINS*POINTS_PER_BIN-1);
  
    Iiter[iIter] = I;
    Viter[iIter] = E;

    Ibest       += I * (I*I/var);
    inv_var_tot += I * I / var ;

    cout << "INFO: \\int{f(x)} = " <<  I << " \\pm " << sqrt(var) << endl;
    // cout << "INFO: m_tot = " << mtot << endl;

    if( NITER > 1 ) {
      RebinCPU( d_x, d_m, NBINS );
      //Rebin<<<1,1>>>( d_x, d_m, NBINS, NDIM );
    }
  }

  cout << "INFO: Final results" << endl;
  
  Ibest /= inv_var_tot;
  float sigma_best = Ibest / sqrt( inv_var_tot );
  
  printf( "INFO: I_best = %f \\pm %f\n", Ibest, sigma_best );

  for( int m = 0 ; m < NITER ; ++m ) {
    chi2 += ( Iiter[m] - Ibest ) * ( Iiter[m] - Ibest ) / Viter[m];
  }
  double chi2ndf = chi2 / (float)( NITER-1 );
  printf( "INFO: chi2 / ndf = %f / %i = %f\n", chi2, NITER, chi2ndf );
  
  cout << "INFO: Finished. Flushing memory." << endl;
  // flush memory
  cudaFree( d_x );
  cudaFree( d_I );
  cudaFree( d_E );
  cudaFree( d_m );
  cudaFree( devStates );
  cout << "DEBUG: GPU memory freed" << endl;

  delete [] Iiter;
  delete [] Viter;

  return success;
}
