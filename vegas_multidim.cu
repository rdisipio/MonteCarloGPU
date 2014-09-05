// dear emacs, please treat this as -*- C++ -*- 

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <curand_kernel.h>

#include "CudaErrorCheck.h"

#define THREADS_PER_BLOCK 256
#define THREADS_PER_BLOCK_SAMPLING 8
#define POINTS_PER_BIN   1024
//( THREADS_PER_BLOCK * THREADS_PER_BLOCK )

using namespace std;

__device__ __host__
float func( const float * x )
{
  //return exp( -(x-0.5)*(x-0.5)/0.1 );  // 0.546292
  //return x[0]*x[0];

  //return x[0]*x[0] + x[1]*x[1]; //0.666667
  //return sin( x[0] ) * cos( x[1] ); // 0.386822
  return x[0]*x[0] + sin(x[1]); // 0.793031
  //return x[0]*x[0] + x[1]*x[1] + 2*x[2]*x[2]; //1.33333
}


__global__
void setup_random( curandState * state )
{
   int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;  
   curand_init(1234, id, 0, &state[id]);
}


__global__
void MakeEdges( float * d_x, const float * xmin, const float * xmax, const int NBINS, const int NDIM )
{
  const int ibin  = blockIdx.x;
  const int idim  = threadIdx.x;
  const int iglob = idim * (NBINS+1) + ibin;

  if( idim > NDIM ) return;
  if( ibin >= (NBINS+1) ) return;

  d_x[iglob] = xmin[idim] + ( xmax[idim] - xmin[idim] ) * (float)ibin / (float)NBINS;
}


__host__ __device__
void DumpEdges( const float * x, const int NBINS, const int NDIM )
{
  printf( "DEBUG: %i bin edges in %i dimensions:\n", NBINS, NDIM );  
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    for( unsigned int b = 0 ; b <= NBINS ; ++b ) {
      printf( " %f", x[d*(NBINS+1)+b] );
    }
    printf( "\n" );
  }
}

void DumpBinVolumes( const float * binVol, const int NBINS, const int NDIM )
{
  cout << "DEBUG: bin volumes:" << endl;  

  const unsigned int NBINS_TOT = pow( NBINS, NDIM );
  for( unsigned int b = 0 ; b < NBINS_TOT ; ++b ) {
    cout << " [(" << b << ")" << binVol[b] << "] ";
    if( (b+1) % NBINS == 0 ) cout << endl;
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ __host__
void Refine( float * x_new, const float * x_old, const int * split_map, 
	     const int NBINS_OLD, const int NBINS_NEW, const int idim )
{
  int pos = 0;
  int offset = idim * ( NBINS_OLD + 1 );
  for( int i = 0 ; i < NBINS_OLD ; ++i ) {
    int Nsplit = (int)split_map[i];

    if( Nsplit < 2 ) {
      x_new[pos] = x_old[offset+i];
      //printf("%i : %f\n", pos, x_new[pos] );
      pos++;
    }
    else {
      const int Nbins = split_map[i];
      //printf("rebin (%i) -> %i\n", i, Nbins );

      const float deltax = x_old[offset+i+1] - x_old[offset+i];
      const float bw     = deltax / (float)Nbins;

      for( int p = 0 ; p < Nbins; ++p ) {
	x_new[pos] = x_old[offset+i] + p*bw;
	//printf("%i : %f\n", pos, x_new[pos] );
	pos++;
      }    
    }
  }
  x_new[NBINS_NEW] = x_old[offset+NBINS_OLD];
  
  /*
  for( int i = 0 ; i < NBINS_NEW ; ++i ) {
    float deltax = x_new[i+1] - x_new[i];
    printf( "  %2i) [ %f, %f ] = [ %f ] \n", i, x_new[i], x_new[i+1], deltax  );
  }
  */
  //printf("DEBUG: Refine done.\n" );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__device__ __host__
void Resize( float * x, const float * x_tmp, const float * h, const float tot_weight,
	     const int NBINS, const int NBINS_NEW, const int NDIM )
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
  
  /*
  for( int i = 0 ; i < NBINS ; ++i ) {
    const float deltax = x[i+1] - x[i];
    printf( "  %2i) [ %f, %f ] = [ %f ]\n", i, x[i], x[i+1], deltax );
  }
  */
  //printf("DEBUG: Resize done.\n" );
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


void RebinCPU( float * x, const float * f_abs, const int NBINS, const int NDIM )
{

  // const float xrange = x[NBINS] - x[0];
  float tot_weight = 0.;
  for( int i = 0 ; i < NBINS ; ++i ) {
    const float deltax = x[i+1] - x[i];
    //const float w = deltax / xrange;
    tot_weight += deltax * f_abs[i];
  }

  int NBINS_NEW = 0;
  int * split_map;
  cudaMallocManaged( &split_map, NBINS*sizeof(int) );

  for( int i = 0 ; i < NBINS ; ++i ) {
    split_map[i]  = 1 + floor( f_abs[i] / tot_weight + 0.5 );
    NBINS_NEW += split_map[i];
  }
  printf( "DEBUG: nbins new = %i\n", NBINS_NEW );

  float * x_tmp; //= new float[NBINS_NEW+1];
  cudaMallocManaged( &x_tmp, (NBINS+1) * sizeof(float) );

  Refine( x_tmp, x, split_map, NBINS, NBINS_NEW, NDIM );

  // check convergence here

  Resize( x, x_tmp, f_abs, tot_weight, NBINS, NBINS_NEW, NDIM );
  
  cudaFree( split_map );  
  cudaFree( x_tmp );
  
  printf("DEBUG: Rebin done\n" );
}


/////////////////////////////////////////////


__global__
void RebinGPU( float * x, 
	       const float * f_abs,
	       const float * binVol, const float totVol, 
	       const unsigned int NBINS, const unsigned int NDIM, const unsigned int NBINS_TOT )
{
  const int idim = blockIdx.x;
  if( idim > NDIM ) return;

  float tot_weight  = 0.;
  for( int i = 0 ; i < NBINS ; ++i ) {
    const unsigned int j = idim*(NBINS+1) + i;
    const float deltax = x[j+1] - x[j];
    tot_weight += deltax * f_abs[i];
  }
 

  int NBINS_NEW = 0;
  int   * split_map = new int[NBINS];
  float * x_tmp     = new float[NBINS_NEW];
  
  for( int i = 0 ; i < NBINS ; ++i ) {
    split_map[i]  = 1 + floor( f_abs[i] / tot_weight  + 0.5 );
    NBINS_NEW += split_map[i];
    // printf( "bin %i -> %i\n", i, split_map[i] );
  }
  // printf( "DEBUG: refined bins: %i\n", NBINS_NEW );

  Refine( x_tmp, x, split_map, NBINS, NBINS_NEW, idim );
  //DumpEdges( x_tmp, NBINS_NEW, NDIM );

  Resize( x, x_tmp, f_abs, tot_weight, NBINS, NBINS_NEW, idim );

  delete [] split_map;
  delete [] x_tmp;
}


/////////////////////////////////////////////



__global__
void CalcBinWidth( float * bw, const float * x, const int NBINS, const int NDIM )
{
  const int ibin  = blockIdx.x;
  const int idim  = threadIdx.x;
  const int iglob = idim * NBINS + ibin;
  const int iedge = idim * ( NBINS+1 ) + ibin;

  if( idim > NDIM ) return;
  if( ibin > NBINS ) return;

  bw[iglob] = x[iedge+1] - x[iedge];
}


void DumpBinWidths( const float * bw, const int NBINS, const int NDIM )
{
  cout << "DEBUG: bin widths:" << endl;  
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    for( unsigned int b = 0 ; b < NBINS ; ++b ) {
      cout << " " << bw[d*NBINS+b];
    }
    cout << endl;
  }
}

//~~~~~~~~~~~~~~~~~~~~~~


__global__ 
void CalcBinVolume( float * binVol, const float * bw, const int NBINS, const int NDIM, const int NBINS_TOT )
{
  //const unsigned int ibin  = threadIdx.x; 
  const unsigned int ibin  = blockIdx.x;

  //if( idim > NDIM ) return;
  if( ibin > NBINS_TOT ) return;

  float bv = 1.;

  // find bin indices
  unsigned int ig = ibin;
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    const unsigned int i     = ig % NBINS;
    const unsigned int iglob = d * NBINS + i;

    bv *= bw[iglob];

    ig /= NBINS;
  }

  binVol[ibin] = bv;
}


/////////////////////////////////////////////////


__global__
void DoSampling( 
		  float * sum_f, float * sum_f_sq, float * sum_f_abs,
		  curandState * state,
		  const float * x, const float * bw,
		  const int NBINS, const int NDIM, const int NBINS_TOT 
		 )
{
  /*
    N bins
    M threads per block (1 block = 1 bin)
    N * M threads in total
   */

  __shared__ float s_f[THREADS_PER_BLOCK];
  __shared__ float s_f_sq[THREADS_PER_BLOCK];
  __shared__ float s_f_abs[THREADS_PER_BLOCK];

  unsigned int iglob = blockIdx.x;
  unsigned int ibin  = blockIdx.x % NBINS;
  unsigned int idim  = ibin / NBINS;
  unsigned int tid   = threadIdx.x; 
  unsigned int I     = iglob*blockDim.x + tid;
  
  if( ibin > NBINS ) return;
  if( iglob >= NBINS_TOT ) return;
  if( idim > NDIM ) return;

  curandState localState = state[I];
  
  float f     = 0.;
  float f_sq  = 0.;
  float f_abs = 0.;
  
  float * x0 = new float[NDIM]; // N-dim point "x"

  // for each sampling (k) evaluate f() at a certain d-dimensional point x0

  const unsigned int NSAMPLINGS = POINTS_PER_BIN/THREADS_PER_BLOCK_SAMPLING ;
  for( unsigned int k = 0 ; k < NSAMPLINGS ; ++k ) {

    int ig = iglob;
    for( unsigned int d = 0 ; d < NDIM ; ++d ) {
      const unsigned int id = ig % NBINS;
      const unsigned int i  = d*(NBINS+1) + id;
      const unsigned int j  = d*(NBINS) + id;
      const float deltax = bw[j];

      x0[d] = x[i] + deltax * curand_uniform( &localState ); 
      ig /= NBINS;
    }

    const float y  = func( x0 );

    f     += y;
    f_sq  += y*y;
    f_abs += fabs(y);
  }
  delete x0;

  s_f[tid]     = f;
  s_f_sq[tid]  = f_sq;
  s_f_abs[tid] = f_abs;
  
  __syncthreads();

  // do reduction
  
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
    sum_f[iglob]     = s_f[0];
    sum_f_sq[iglob]  = s_f_sq[0];
    sum_f_abs[iglob] = s_f_abs[0];
  }
}


/////////////////////////////////////////////////


__global__ 
void Finalize( float * sum_f, 
	       const float * d_f, 
	       const float * binVol, const float totVol,
	       const unsigned int NBINS_TOT )
{
  extern __shared__ float s_f[];

  unsigned int tid = threadIdx.x;
  unsigned int   i = blockIdx.x*blockDim.x + threadIdx.x;

  if( tid >= NBINS_TOT ) return;

  const float w = binVol[i] / totVol;
  s_f[tid]    = w * d_f[i] / POINTS_PER_BIN;

  __syncthreads();

  for( unsigned int s = blockDim.x/2 ; s > 0 ; s >>= 1 ) {
    if( tid < s ) {
      s_f[tid]     += s_f[tid+s];
    }
    __syncthreads();
  }

  if( tid == 0 ) {
     sum_f[blockIdx.x] = s_f[0];
  }
 
}


////////////////////////////////////////////////

void printDevProp(cudaDeviceProp devProp)
{
  printf("*********************************\n" );
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
    printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %u\n",  devProp.totalConstMem);
    printf("Texture alignment:             %u\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
    printf("*********************************\n" );
    return;
}

int main( int argc, char ** argv )
{
  int success = 0;

  /*
  cudaDeviceProp devProp;
  cudaGetDeviceProperties( &devProp, 0 );
  printDevProp(devProp);
  */

  int NBINS = ( argc > 1 ) ? atoi(argv[1]) : 8;
  int NITER = ( argc > 2 ) ? atoi(argv[2]) : 4;

  const unsigned int NDIM = 2;

  const unsigned int NBINS_TOT = pow( NBINS, NDIM );

  float * xmin;
  float * xmax;
  float * d_x;
  float * d_f;
  float * d_f_sq;
  float * d_f_abs;
  float * d_bw;
  float * d_binVol;
  float * d_sum_f; 
  float * d_sum_f_sq;
  float * d_sum_f_abs;
    
  cudaMallocManaged( &xmin, NDIM*sizeof(float) );
  cudaMallocManaged( &xmax, NDIM*sizeof(float) );
  cudaMallocManaged( &d_x,        NDIM * (NBINS+1) * sizeof(float) );
  cudaMallocManaged( &d_bw,       NDIM * NBINS     * sizeof(float) );
  CudaSafeCall( cudaMallocManaged( &d_f,        NBINS_TOT        * sizeof(float) ) );
  cudaMallocManaged( &d_f_sq,        NBINS_TOT        * sizeof(float) );
  cudaMallocManaged( &d_f_abs,        NBINS_TOT        * sizeof(float) );
  cudaMallocManaged( &d_binVol,   NBINS_TOT        * sizeof(float) );
  cudaMallocManaged( &d_sum_f,     NBINS_TOT  * sizeof(float) );
  cudaMallocManaged( &d_sum_f_sq,  NBINS_TOT * sizeof(float) );
  cudaMallocManaged( &d_sum_f_abs, NBINS_TOT  * sizeof(float) );

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // setup random generator
  curandState *devStates;
  cudaMallocManaged( (void **)&devStates , NDIM * THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(curandState) );

  cudaEventRecord(start, 0);  
  setup_random<<< THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( devStates );
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup random numbers = " << time << " ms" << endl;
  CudaCheckError();

  // Initialize bin edges
  float totVol = 1.;
  for( int d = 0 ; d < NDIM ; ++d ) {
    xmin[d] = 0.0; xmax[d] = 1.0;
    totVol *= ( xmax[d] - xmin[d] );
  }
  
  cudaEventRecord(start, 0);  
  MakeEdges<<< (NBINS+1), NDIM >>>( d_x, xmin, xmax, NBINS, NDIM );
  cudaDeviceSynchronize();  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cout << "INFO: CUDA: Time for setup bin edges = " << time << " ms" << endl;
  DumpEdges( d_x, NBINS, NDIM );

  cout << "INFO: VEGAS will be executed on " << NBINS_TOT << " bins in total" << endl;

  float Ibest    = 0.;
  float * Iiter = new float[NITER];
  float * Viter = new float[NITER];
  float inv_var_tot  = 0.;
  float chi2     = 0.;
  
  for( int iIter = 0 ; iIter < NITER ; ++iIter ) {
    cout << "INFO: Iteration no. " << iIter << endl;

    cudaEventRecord(start, 0);  
    CalcBinWidth<<< NBINS, NDIM >>>( d_bw, d_x, NBINS, NDIM );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for bin widths = " << time << " ms" << endl;
    //DumpBinWidths( d_bw, NBINS, NDIM );
    CudaCheckError();

    cudaEventRecord(start, 0);  
    CalcBinVolume<<< NBINS_TOT, 1 >>>( d_binVol, d_bw, NBINS, NDIM, NBINS_TOT );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for bin volumes = " << time << " ms" << endl;
    //DumpBinVolumes( d_binVol, NBINS, NDIM );
    CudaCheckError();

    cudaEventRecord(start, 0);  
    DoSampling<<< NBINS_TOT, THREADS_PER_BLOCK_SAMPLING >>>( d_f, d_f_sq, d_f_abs, devStates, d_x, d_bw, NBINS, NDIM, NBINS_TOT );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for sampling = " << time << " ms" << endl;
    CudaCheckError();
    //    cudaDeviceSynchronize();  

    // final reduction
    //const size_t num_blocks = ( NBINS_TOT / THREADS_PER_BLOCK) + ( (NBINS_TOT % THREADS_PER_BLOCK) ? 1 : 0 );

    cudaEventRecord(start, 0);  
    Finalize<<< 1, NBINS_TOT, NBINS_TOT*sizeof(float) >>>( d_sum_f,     d_f,     d_binVol, totVol, NBINS_TOT );
    Finalize<<< 1, NBINS_TOT, NBINS_TOT*sizeof(float) >>>( d_sum_f_sq,  d_f_sq,  d_binVol, totVol, NBINS_TOT );
    Finalize<<< 1, NBINS_TOT, NBINS_TOT*sizeof(float) >>>( d_sum_f_abs, d_f_abs, d_binVol, totVol, NBINS_TOT );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for final reduction = " << time << " ms" << endl;

    float I = d_sum_f[0];
    float E = d_sum_f_sq[0];
    //float I = d_sum_f[num_blocks];
    //float E = d_sum_f_sq[num_blocks];
    //float Iabs = d_sum_f_abs[0];

    float var = ( E - I*I ) / ( NBINS_TOT * POINTS_PER_BIN-1);
    cout << "INFO: \\int{f(x)} = " <<  I << " \\pm " << sqrt(var) << endl;
  
    Iiter[iIter] = I;
    Viter[iIter] = E;

    Ibest       += I * (I*I/var);
    inv_var_tot += I * I / var ;
  
    if( NITER > 1 ) {
        cudaEventRecord(start, 0);  
	RebinGPU<<< NDIM, 1 >>>( d_x, d_f_abs, d_binVol, totVol, NBINS, NDIM, NBINS_TOT );
	cudaDeviceSynchronize();  
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	cout << "INFO: CUDA: Time for rebinning = " << time << " ms" << endl;

	//      RebinCPU( d_x, d_f_abs, NBINS, NDIM );
	DumpEdges( d_x, NBINS, NDIM );
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
  cudaFree( d_f );
  cudaFree( d_f_sq );
  cudaFree( d_f_abs );
  cudaFree( d_binVol );
  cudaFree( devStates );
  cudaFree( xmin );
  cudaFree( xmax );
  cudaFree( d_bw );
  cudaFree( d_sum_f );
  cudaFree( d_sum_f_sq );
  cudaFree( d_sum_f_abs );
  cout << "DEBUG: GPU memory freed" << endl;

  delete [] Iiter;
  delete [] Viter;

  return success;
}
