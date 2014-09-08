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
//#define THREADS_PER_BLOCK_SAMPLING 8
#define CALLS_PER_BOX   128
//( THREADS_PER_BLOCK * THREADS_PER_BLOCK )

using namespace std;

__device__ __host__
float func( const float * x )
{
  //return 1.;

  //return exp( -(x-0.5)*(x-0.5)/0.1 );  // 0.546292
  //return x[0]*x[0];

  //return x[0]*x[0] + x[1]*x[1]; //0.666667
  //return sin( x[0] ) * cos( x[1] ); // 0.386822
  // return x[0]*x[0] + sin(x[1]); // 0.793031
  return x[0]*x[0] + x[1]*x[1] + 2*x[2]*x[2]; //1.33333
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
  printf( "<<\n" );
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    printf( "[" );
    for( unsigned int b = 0 ; b <= NBINS ; ++b ) {
      printf( " %f", x[d*(NBINS+1)+b] );
    }
    printf( " ]\n" );
  }
  printf( ">>\n" );
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


__global__
void RefineGPU( float * x, const float * f, const float * binVolumes, const float totVolume, const float alpha, const int NBINS )
{
  // http://root.cern.ch/root/html/src/RooGrid.cxx.html#LazfmD

  // smooth this dimension's histogram of grid values and calculate the
  // new sum of the histogram contents as grid_tot_j

  const unsigned int idim = blockIdx.x;

  const unsigned int offset = idim * ( NBINS );
  const unsigned int xoffset = idim * ( NBINS + 1 );

  float * wf = new float[NBINS];
  for( unsigned int i = 0 ; i < NBINS ; ++i ) {
    const int j = offset + i;

    wf[i] = binVolumes[j] * f[j] / totVolume / CALLS_PER_BOX;
  }

  float oldg = wf[0];
  float newg = wf[1];
  wf[0] = 0.5 * ( oldg + newg );
  float grid_tot_j = wf[0];

  // this loop implements value(i,j) = ( value(i-1,j)+value(i,j)+value(i+1,j) ) / 3
  unsigned int i = 0;
  for( i = 1 ; i < (NBINS-1) ; ++i ) {
    float rc = oldg + newg;
    oldg = newg;
    newg = wf[i+1];
    wf[i] = ( rc + newg ) / 3.;
    grid_tot_j += wf[i];
  }
  wf[NBINS-1] = 0.5 * ( newg + oldg );
  grid_tot_j += f[NBINS-1];

  // calculate the weights for each bin of this dimension's histogram of values and their sum
  float * weights = new float[NBINS];
  float tot_weight = 0.;
  for( i = 0 ; i < NBINS ; ++i ) {
    weights[i] = 0.;
    if( wf[i] > 0. ) {
      oldg = grid_tot_j / wf[i];
      weights[i] = powf( ( (oldg-1.0)/oldg/log(oldg)), alpha );
    }
    tot_weight += weights[i];
  }

  float pts_per_bin = tot_weight / NBINS;

  float xold = 0.;
  float xnew = 0.;
  float dw   = 0.;
  float * x_new = new float[NBINS];

  unsigned int k = 0;
  i = 1;
  for( k = 0 ; k < NBINS ; ++k ) {
    dw += weights[k];
    xold = xnew;
    xnew = x[xoffset+k+1];

    while( dw > pts_per_bin ) { 
      dw -= pts_per_bin;
      x_new[i++] = xnew - (xnew - xold) * dw / weights[k];
    }
  }

  for( k = 1 ; k < NBINS ; ++k ) {
    x[xoffset+k] = x_new[k];
  }

  //x[xoffset+NBINS] = 1.0;

  delete [] wf;
  delete [] weights;
  delete [] x_new;
}

/////////////////////////////////////////////

/*
__device__ __host__
void ResizeGPU( float * xtarget, float * x_source,
		const unsigned int nbins_target, const unsigned int nbins_source, 
		const unsigned int idim )
{
  if( nbins_target == nbins_source ) return;

  const unsigned int xoffset = idim * ( NBINS + 1 );

  float pts_per_bin = (float)nbins_target / (float)nbins_source;
  
  float xold = 0.;
  float xnew = 0.;
  float dw   = 0.;
  unsigned int i = 0;
  unsigned int k = 0;
  for( k = 1 ; k <= nbins_source ; ++k ) {
    dw += 1.0;
    xold = xnew;
    while( dw > pts_per_bin ) {
      dw -= pts_per_bin;
      x_source[i++] = xnew - (xnew - xold) * dw;
    }
  }

  // copy new edges
  for(k = 1 ; k < nbins_target ; k++) {
    x[xoffset+k] = x_source[xoffset+k];
  }
}
*/

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

void DumpBoxContent( const float * bw, const int NBINS, const int NDIM )
{
  cout << "DEBUG: box content:" << endl;  
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    for( unsigned int b = 0 ; b < NBINS ; ++b ) {
      cout << " " << bw[d*NBINS+b];
    }
    cout << endl;
  }
}

//~~~~~~~~~~~~~~~~~~~~~~


__global__ 
void CalcBinVolume( float * binVol, const float * bw, const int NBINS, const int NDIM, const int NBOXES )
{
  //const unsigned int ibin  = threadIdx.x; 
  //const unsigned int ibin  = blockIdx.x;
  const unsigned int ibin  = blockIdx.x*blockDim.x + threadIdx.x;

  //if( idim > NDIM ) return;
  if( ibin > NBOXES ) return;

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
void DoSampling( float * d_f, float * d_f_sq, float * d_f_abs,
		  curandState * state,
		  const float * x, const float * bw,
		  const int NBINS, const int NDIM, const int NBOXES 
		 )
{
  unsigned int iglob = blockIdx.x*blockDim.x + threadIdx.x; 

  unsigned int ibin  = iglob % NBINS;
  unsigned int idim  = ibin / NBINS;

  if( ibin > NBINS ) return;
  if( iglob >= NBOXES ) return;
  if( idim > NDIM ) return;

  curandState localState = state[iglob];

  float f     = 0.;
  float f_sq  = 0.;
  float f_abs = 0.;
  
  float * x0 = new float[NDIM];

  for( unsigned int k = 0 ; k < CALLS_PER_BOX ; ++k ) {

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

  d_f[iglob]     = f;
  d_f_sq[iglob]  = f_sq;
  d_f_abs[iglob] = f_abs;

}


/////////////////////////////////////////////////

__global__
void Accumulate( float * output, const float * input,
		  const unsigned int NTHREADS )
{
  extern __shared__ float s_f[];
  
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  if( i >= NTHREADS ) return;
  
  s_f[i]    = input[i];

  __syncthreads();

  for( unsigned int s = blockDim.x/2 ; s > 0 ; s >>= 1 ) {
    if( i < s ) {
      s_f[i]     += s_f[i+s];
    }
    __syncthreads();
  }

  if( i == 0 ) {
     output[blockIdx.x] = s_f[0];
  }
}


__global__
void ReduceBoxes( float * output, const float * input,
		  const float * binVol, const float totVol,
		  const unsigned int NBOXES )
{
  extern __shared__ float s_f[];
  
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int   i = blockIdx.x*blockDim.x + threadIdx.x;

  if( i > NBOXES ) return;
  if( tid > THREADS_PER_BLOCK ) return;
  
  const float w = binVol[i] / totVol;
  s_f[tid]    = w * input[i] / ( CALLS_PER_BOX );

  __syncthreads();

  for( unsigned int s = blockDim.x/2 ; s > 0 ; s >>= 1 ) {
    if( tid < s ) {
      s_f[tid]     += s_f[tid+s];
    }
    __syncthreads();
  }

  if( tid == 0 ) {
     output[bid] = s_f[0];
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

  
  cudaDeviceProp devProp;
  cudaGetDeviceProperties( &devProp, 0 );
  printDevProp(devProp);
  

  const int NBINS = ( argc > 1 ) ? atoi(argv[1]) : 8;
  const int NITER = ( argc > 2 ) ? atoi(argv[2]) : 4;
  const float ALPHA = ( argc > 3 ) ? atoi(argv[3]) : 1.5;

  const unsigned int NDIM = 3;

  const unsigned int NBOXES     = pow( NBINS, NDIM );
  const unsigned int NUM_BLOCKS = ( NBOXES / THREADS_PER_BLOCK ) + ( (NBOXES % THREADS_PER_BLOCK) ? 1 : 0 );

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
  CudaSafeCall( cudaMallocManaged( &d_f,        NBOXES        * sizeof(float) ) );
  cudaMallocManaged( &d_f_sq,        NBOXES        * sizeof(float) );
  cudaMallocManaged( &d_f_abs,        NBOXES        * sizeof(float) );
  cudaMallocManaged( &d_binVol,   NBOXES        * sizeof(float) );
  cudaMallocManaged( &d_sum_f,     (NUM_BLOCKS) * sizeof(float) );
  cudaMallocManaged( &d_sum_f_sq,  (NUM_BLOCKS) * sizeof(float) );
  cudaMallocManaged( &d_sum_f_abs, (NUM_BLOCKS) * sizeof(float) );

  cudaEvent_t start, stop;
  float time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  // setup random generator
  curandState *randomStates;
  cudaMallocManaged( (void **)&randomStates , NDIM * THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(curandState) );

  cudaEventRecord(start, 0);  
  setup_random<<< THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( randomStates );
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

  cout << "INFO: VEGAS will be executed on " << NBOXES << " bins in total" << endl;
  cout << "DEBUG: number of parallel blocks in reduction step: " << NUM_BLOCKS << endl;

  float Ibest    = 0.;
  float * Iiter = new float[NITER];
  float * Viter = new float[NITER];
  float inv_var_tot  = 0.;
  float chi2     = 0.;

  //const float jacobian = totVol * pow( NBINS, NDIM ) / (NBINS * CALLS_PER_BOX );
  // const float jacobian = totVol / CALLS_PER_BOX;

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

    unsigned int bvdim = ( NBOXES / THREADS_PER_BLOCK ) + ( (NBOXES % THREADS_PER_BLOCK) ? 1 : 0 );
    cudaEventRecord(start, 0);  
    CalcBinVolume<<< bvdim, THREADS_PER_BLOCK >>>( d_binVol, d_bw, NBINS, NDIM, NBOXES );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for bin volumes = " << time << " ms" << endl;
    //DumpBinVolumes( d_binVol, NBINS, NDIM );
    CudaCheckError();

    cudaEventRecord(start, 0);  
    DoSampling<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>( d_f, d_f_sq, d_f_abs, randomStates, d_x, d_bw, NBINS, NDIM, NBOXES );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for sampling = " << time << " ms" << endl;
    CudaCheckError();
    //    cudaDeviceSynchronize();  
    //DumpBoxContent( d_f, NBINS, NDIM );

    // accumulate results - first pass
    cudaEventRecord(start, 0);  
    ReduceBoxes<<< NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(float) >>>( d_sum_f,     d_f,     d_binVol, totVol, NBOXES );
    ReduceBoxes<<< NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(float) >>>( d_sum_f_sq,  d_f_sq,  d_binVol, totVol, NBOXES );
    ReduceBoxes<<< NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(float) >>>( d_sum_f_abs, d_f_abs, d_binVol, totVol, NBOXES );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for accumulation (first step) = " << time << " ms" << endl;
    CudaCheckError();
    //DumpBoxContent( d_sum_f, NUM_BLOCKS, 1 );

    // accumulate results - second pass
    cudaEventRecord(start, 0);  
    Accumulate<<< 1, NUM_BLOCKS, NUM_BLOCKS*sizeof(float) >>>( d_sum_f    , d_sum_f,     NUM_BLOCKS );
    Accumulate<<< 1, NUM_BLOCKS, NUM_BLOCKS*sizeof(float) >>>( d_sum_f_sq , d_sum_f_sq,  NUM_BLOCKS );
    Accumulate<<< 1, NUM_BLOCKS, NUM_BLOCKS*sizeof(float) >>>( d_sum_f_abs, d_sum_f_abs, NUM_BLOCKS );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cout << "INFO: CUDA: Time for accumulation (second step) = " << time << " ms" << endl;
    CudaCheckError();

    float I = d_sum_f[0];
    float E = d_sum_f_sq[0];
    
    float var = ( E - I*I ) / ( NBOXES * CALLS_PER_BOX - 1 );
    cout << "INFO: \\int{f(x)} = " <<  I << " \\pm " << sqrt(var) << endl;
  
    Iiter[iIter] = I;
    Viter[iIter] = E;

    Ibest       += I * (I*I/var);
    inv_var_tot += I * I / var ;
  
    if( NITER > 1 ) {
        cudaEventRecord(start, 0);  
	RefineGPU<<< NDIM, 1 >>>( d_x, d_f_abs, d_binVol, totVol, ALPHA, NBINS );
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
  cudaFree( randomStates );
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
