// dear emacs, please treat this as -*- C++ -*- 

#include <iostream>
#include <stdio.h>
#include <argp.h>
#include <sstream>

#include <cuda.h>
#include <math_functions.h>
#include <cuda_runtime.h>
#include <vector_functions.h>
#include <curand_kernel.h>

#include "CudaErrorCheck.h"

//#include "kernel_functor.h"

#define _SEP_ ","
#define THREADS_PER_BLOCK 256
//#define THREADS_PER_BLOCK_SAMPLING 8
//#define CALLS_PER_BOX   512
//( THREADS_PER_BLOCK * THREADS_PER_BLOCK )

using namespace std;						

/*
DEFINE_KERNEL_FUNCTOR( 1, 0, 0.5,                                 0.5 )
DEFINE_KERNEL_FUNCTOR( 1, 1, x[0],                                1.0 )
DEFINE_KERNEL_FUNCTOR( 1, 2, x[0]*x[0],                           0.3333 )
DEFINE_KERNEL_FUNCTOR( 1, 3, exp( -(x[0]-0.5)*(x[0]-0.5)/0.1 ),   0.546292 )

DEFINE_KERNEL_FUNCTOR( 2, 0, x[0]*x[0] + x[1]*x[1],               0.666667 )
DEFINE_KERNEL_FUNCTOR( 2, 1, sin( x[0] ) * cos( x[1] ),           0.386822 )
DEFINE_KERNEL_FUNCTOR( 2, 2, x[0]*x[0] + sin(x[1]),               0.793031 )

DEFINE_KERNEL_FUNCTOR( 3, 0, x[0]*x[0] + x[1]*x[1] + 2*x[2]*x[2], 1.33333 )

CREATE_KERNEL_FUNCTOR_OBJECT(1,0)
CREATE_KERNEL_FUNCTOR_OBJECT(1,1)
CREATE_KERNEL_FUNCTOR_OBJECT(1,3)

IKernelFunctor * known_functors[3] = { &GET_FUNCTOR_OBJECT(1,0), &GET_FUNCTOR_OBJECT(1,1), &GET_FUNCTOR_OBJECT(1,3) };
//__constant__ pf_func function_table[2] = { GET_FUNCTOR_OBJECT(1,0), GET_FUNCTOR_OBJECT(1,1) }

#define ACTIVE_KERNEL_INDEX 2
*/

__device__ __host__
float integrand( const float * x )
{
  //return 1.;

  //return exp( -(x-0.5)*(x-0.5)/0.1 );  // 0.546292
  return x[0]*x[0];

  //return x[0]*x[0] + x[1]*x[1]; //0.666667
  //return sin( x[0] ) * cos( x[1] ); // 0.386822
//  return x[0]*x[0] + sin(x[1]); // 0.793031
  //return x[0]*x[0] + x[1]*x[1] + 2*x[2]*x[2]; //1.33333
}
static const unsigned int NDIM = 1;

//DECLARE_KERNEL_FUNCTION( 1, 0, x[0]*x[0] )
//DECLARE_KERNEL_FUNCTION( 1, 1, sin(x[0]) )

//__device__ pf_func d_pfunc = func;


//MAKE_KERNEL_NAME(1,1) myfunctor;
//__device__ pf_func d_pfunc = myfunctor;

//static const unsigned int NDIM = (known_functors[ACTIVE_KERNEL_INDEX])->GetNDim();


//__constant__ pf_func function_table[2] = { MAKE_KERNEL_NAME(1,0), MAKE_KERNEL_NAME(1,1) };

//static unsigned int NDIM = 1; // set this accordingly..
//static unsigned int ACTIVE_FUNCTION = 0;

__global__
void setup_random( curandState * state )
{
   int id = threadIdx.x + blockIdx.x * THREADS_PER_BLOCK;  
   curand_init(1234, id, 0, &state[id]);
}


__global__
void InitializeEdges( float * d_x, float * d_xl, float * d_xh, const float * xmin, const float * xmax, const int NBINS, const int NDIM )
{
  const int ibin  = blockIdx.x;
  const int idim  = threadIdx.x;
  const int i     = idim * NBINS + ibin;
  const int iglob = idim * (NBINS+1) + ibin;

  if( idim > NDIM ) return;
  if( ibin >= (NBINS+1) ) return;

  d_x[iglob] = xmin[idim] + ( xmax[idim] - xmin[idim] ) * (float)ibin / (float)NBINS;

  d_xl[i] = xmin[idim] + ( xmax[idim] - xmin[idim] ) * (float)ibin / (float)NBINS;
  d_xh[i] = xmin[idim] + ( xmax[idim] - xmin[idim] ) * (float)(ibin+1) / (float)NBINS;
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

void DumpBoxVolumes( const float * boxVol, const int NBINS, const int NDIM )
{
  cout << "DEBUG: box volumes:" << endl;  

  const unsigned int NBINS_TOT = pow( NBINS, NDIM );
  for( unsigned int b = 0 ; b < NBINS_TOT ; ++b ) {
    cout << " [(" << b << ")" << boxVol[b] << "] ";
    if( (b+1) % NBINS == 0 ) cout << endl;
  }
}


void DumpProjection( const float * f, const int NBINS, const int NDIM )
{
  cout << "DEBUG: integrand projections:" << endl;  

  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    for( unsigned int b = 0 ; b < NBINS ; ++b ) {
      const unsigned int j = d*NBINS + b;

      cout << " [(" << b << ")" << f[j] << "] ";
      if( (b+1) % NBINS == 0 ) cout << endl;
    }
  }
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


__global__
void RefineGPU( float * x, float * xl, float * xh,
		const float * f, const float * binWidths, 
		const float totVolume, const float alpha, 
		const int NBINS, const int NDIM, const int CALLS_PER_BOX )
{
  // http://root.cern.ch/root/html/src/RooGrid.cxx.html#LazfmD

  // smooth this dimension's histogram of grid values and calculate the
  // new sum of the histogram contents as grid_tot_j

  const unsigned int idim = blockIdx.x;

//  const unsigned int offset = idim * ( NBINS );
  const unsigned int offset = idim * NDIM;
  const unsigned int xoffset = idim * ( NBINS + 1 );

  float * wf = new float[NBINS];
  for( unsigned int i = 0 ; i < NBINS ; ++i ) {
    const int j = offset + i;

    wf[i] = binWidths[j] * f[j] / totVolume / CALLS_PER_BOX;
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

  for( k = 0 ; k < NBINS ; ++k ) {
    xl[offset+k] = x[xoffset+k];
    xh[offset+k] = x[xoffset+k+1];
  }

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
void CalcBinWidth( float * bw, const float * xl, const float * xh, const int NBINS, const int NDIM )
{
  const int ibin  = blockIdx.x;
  const int idim  = threadIdx.x;
  const int i = idim * NBINS + ibin;
  //const int iedge = idim * ( NBINS+1 ) + ibin;

  if( idim > NDIM ) return;
  if( ibin > NBINS ) return;

  //bw[iglob] = x[iedge+1] - x[iedge];
  float dx = xh[i] - xl[i];
  if( dx <= 0. ) printf( "ERROR: negative bin width [D=%i N=%i]: (%f, %f) = %f\n", idim, ibin, xl[i], xh[i], dx );

  bw[i] = xh[i] - xl[i];
}


void DumpBinWidths( const float * bw, const int NBINS, const int NDIM )
{
  cout << "DEBUG: bin widths:" << endl;  
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    for( unsigned int b = 0 ; b < NBINS ; ++b ) {
      cout << " [" << d << "," << b << "]" << bw[d*NBINS+b];
    }
    cout << endl;
  }
}

void DumpBoxContent( const float * box, const int NBOXES, const int NDIM )
{
  cout << "DEBUG: box content:" << endl;  
  for( unsigned int b = 0 ; b < NBOXES ; ++b ) {
    cout << " " << box[b];
  }
  cout << endl;
}

//~~~~~~~~~~~~~~~~~~~~~~


__global__ 
void CalcBoxVolume( float * boxVol, const float * bw, const int NBINS, const int NDIM, const int TOT_BOXES )
{
  //const unsigned int ibin  = threadIdx.x; 
  //const unsigned int ibin  = blockIdx.x;
  const unsigned int ibin  = blockIdx.x*blockDim.x + threadIdx.x;

  //if( idim > NDIM ) return;
  if( ibin > TOT_BOXES ) return;

  float bv = 1.;

  // find bin indices
  unsigned int ig = ibin;
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
    const unsigned int i     = ig % NBINS;
    const unsigned int iglob = d * NBINS + i;

    bv *= bw[iglob];

    ig /= NBINS;
  }

  boxVol[ibin] = bv;
}


/////////////////////////////////////////////////

//template< class FUNCTOR >
__global__
void DoSampling( /* pf_func fx, */ float * d_box, float * d_box_sq, float * d_f_abs,
		  curandState * state,
		  const float * xl, const float * bw, const float jacobian,
		  const int NBINS, const int NDIM, const int TOT_BOXES, const int CALLS_PER_BOX  
		 )
{
  unsigned int iglob = blockIdx.x*blockDim.x + threadIdx.x; 

  unsigned int ibin  = iglob % NBINS;
  unsigned int idim  = ibin / NBINS;

  if( ibin > NBINS ) return;
  if( iglob >= TOT_BOXES ) return;
  if( idim > NDIM ) return;

  curandState localState = state[iglob];

  float f     = 0.;
  float f_sq  = 0.;
  float f_abs = 0.;
  
  // find bin indices for each dimension for this box
  int * bin_index = new int[NDIM];
  float binVol = 1.;
  int ig = iglob;
  for( unsigned int d = 0 ; d < NDIM ; ++d ) {
      const unsigned int ib = ig % NBINS;
      const unsigned int j  = d * NBINS + ib;
      bin_index[d] = j;

      const float deltax = bw[j];

      binVol *= deltax;
      ig /= NBINS;
  }

  float * x0 = new float[NDIM];
  for( unsigned int k = 0 ; k < CALLS_PER_BOX ; ++k ) {

    // throw random (x0,..xN) position inside this N-dim box
    for( unsigned int d = 0 ; d < NDIM ; ++d ) {

      const int j        = bin_index[d];
      const float deltax = bw[j];

      x0[d] = xl[j] + deltax * curand_uniform( &localState ); 
    }
     
    const float fx  = integrand( x0 );
    //const float y  = jacobian * binVol * fx;

    f     += jacobian * binVol * fx;
    f_sq  += jacobian * binVol * (fx*fx);
    f_abs += jacobian * binVol * fabs( fx );  
  }

  //find a smarter way?
  for( int d = 0 ; d < NDIM ; ++d ) {
    const int j = bin_index[d];
    atomicAdd( &d_f_abs[j], f_abs ); 
  }

  delete x0;
  delete bin_index;

  d_box[iglob]     = f     ; //* CALLS_PER_BOX;
  d_box_sq[iglob]  = f_sq  ; //* CALLS_PER_BOX;
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
		  const float * boxVol, const float totVol,
		  const unsigned int TOT_BOXES, const unsigned int CALLS_PER_BOX )
{
  extern __shared__ float s_f[];
  
  unsigned int bid = blockIdx.x;
  unsigned int tid = threadIdx.x;
  unsigned int   i = blockIdx.x*blockDim.x + threadIdx.x;

  if( i > TOT_BOXES ) return;
  if( tid > THREADS_PER_BLOCK ) return;
  
  s_f[tid] = input[i];

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

/*
__global__
void ProjectOnAxis( float * d_output, const float * d_input, const int NBINS, const int NDIM, const int TOT_BOXES )
{
  const unsigned int ibin = blockIdx.x;
  const unsigned int idim = threadIdx.x;
  //const unsigned int iglob = idim * NBINS + ibin;

  if( ibin > NBINS ) return;
  if( idim > NDIM ) return;

  for( int b = 0 ; b < NBINS ; b++ ) {
    float sum = 0.;

    for( int d = 0 ; d < NDIM ; d++ ) {
      int j = d*NBINS + b;

      sum += d_input[j];
    }
  }

  d_output[i] = sum;
}
*/

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


///////////////////////////////////////////////////////////////


int main( int argc, char ** argv )
{
  int success = 0;

  stringstream log_overhead;

  int NITER = 1;
  int NBINS = 2;
  int CALLS_PER_BOX = 0;
  int CALLS = 1024;
  float ALPHA = 1.5;
  //int NDIM = -1;

  int c;
  while( (c = getopt( argc, argv, "a:b:c:i:" ) ) != -1) {
    switch (c)
      {
      case 'i':
	NITER = atoi(optarg);
	break;
      case 'b':
	NBINS = atoi(optarg);
	break;
      case 'c':
	CALLS_PER_BOX = atoi(optarg);
	break;
      case 'a':
	ALPHA = atof(optarg);
	break;
	// case 'd':
	//NDIM = atoi(optarg);
	//break;
      case 'x':
	cudaDeviceProp devProp;
	cudaGetDeviceProperties( &devProp, 0 );
	printDevProp(devProp);
	exit(success);
	break;
      default:
	abort();
      }
  }

  // calculate the largest number of equal subdivisions ("boxes") along each
  // axis that results in an average of no more than 2 integrand calls per cell

  /*
  const unsigned int min_calls_per_box = 128;
  unsigned int BOXES = (unsigned int)floor( pow(CALLS/min_calls_per_box, 1.0/NDIM ) );
  int bins = NBINS;
  int box_per_bin = 0;
  if( min_calls_per_box * BOXES >= NBINS ) {
    box_per_bin = ( BOXES > NBINS ) ? BOXES/NBINS : 1;
    bins = BOXES / box_per_bin;
    if( bins > NBINS ) bins = NBINS;
    BOXES = box_per_bin * bins;

    cout << "INFO: use stratified sampling with " << bins << " bins and " << box_per_bin << " box-per-bin" << endl;
  }
  else {
    cout << "INFO: use importance sampling with " << bins << " bins and " << box_per_bin << " box-per-bin" << endl;
  }
  
  unsigned int TOT_BOXES = pow( BOXES, NDIM );
  NBINS = bins;
  
   // increase the total number of calls to get at least min_calls_per_box calls per box, if necessary
   CALLS_PER_BOX = (unsigned int)(CALLS / TOT_BOXES );
   if( CALLS_PER_BOX < min_calls_per_box ) CALLS_PER_BOX = min_calls_per_box;
  */

  unsigned int BOXES = NBINS;

  unsigned int TOT_BOXES = pow( BOXES, NDIM );
  //CALLS_PER_BOX = (unsigned int)(CALLS / TOT_BOXES );
  CALLS = CALLS_PER_BOX * TOT_BOXES;

  const unsigned int NUM_BLOCKS = ( TOT_BOXES / THREADS_PER_BLOCK ) + ( (TOT_BOXES % THREADS_PER_BLOCK) ? 1 : 0 );

  cout << "INFO: No. of dimensions = " << NDIM << endl;
  cout << "INFO: Bins per axis = " << NBINS << endl;
  cout << "INFO: Boxes per axis = " << BOXES << endl;
  cout << "INFO: No. of " << NDIM << "D boxes = " << TOT_BOXES << endl;
  cout << "INFO: Calls per box = " << CALLS_PER_BOX << endl;
  cout << "INFO: Total no. of calls = " << CALLS << endl;
  cout << "INFO: No. of parallel blocks in reduction step: " << NUM_BLOCKS << endl;
  cout << "INFO: No. of iterations = " << NITER << endl;
  cout << "INFO: Alpha parameter = " << ALPHA << endl;

  log_overhead << NBINS << _SEP_ << NITER << _SEP_ << TOT_BOXES << _SEP_ << CALLS_PER_BOX << _SEP_ << CALLS
       	       << _SEP_ << NUM_BLOCKS << _SEP_ << ALPHA;

  cudaEvent_t start, stop;
  float gputime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float * xmin;
  float * xmax;
  float * d_x;
  float * d_xl;
  float * d_xh;
  float * d_box;
  float * d_box_sq;
  float * d_box_abs;
  float * d_bw;
  float * d_boxVol;

  float * d_f;
  float * d_f_sq;
  float * d_f_abs;

  float * d_sum_f; 
  float * d_sum_f_sq;
  float * d_sum_f_abs;
  
  cudaEventRecord(start, 0);  

  /*
    In order to avoid the number of histogram bins growing like K^d the probability distribution is approximated by a separable function: 
    g(x_1, x_2, ...) = g_1(x_1) g_2(x_2) ... 
    so that the number of bins required is only Kd. 
    This is equivalent to locating the peaks of the function from the projections of the integrand onto the coordinate axes. 

    See also: https://www.gnu.org/software/gsl/manual/html_node/VEGAS.html
  */

  CudaSafeCall( cudaMallocManaged( &xmin, NDIM*sizeof(float) )                          );
  CudaSafeCall( cudaMallocManaged( &xmax, NDIM*sizeof(float) )                          );

  CudaSafeCall( cudaMallocManaged( &d_x,        NDIM * NBINS * sizeof(float) )      );
  CudaSafeCall( cudaMallocManaged( &d_xl,       NDIM * NBINS * sizeof(float) )      );
  CudaSafeCall( cudaMallocManaged( &d_xh,       NDIM * NBINS * sizeof(float) )      );
  CudaSafeCall( cudaMallocManaged( &d_bw,       NDIM * NBINS     * sizeof(float) )      );

  CudaSafeCall( cudaMallocManaged( &d_box,      TOT_BOXES  * sizeof(float) )     );
  CudaSafeCall( cudaMallocManaged( &d_box_sq,   TOT_BOXES  * sizeof(float) )     );
  CudaSafeCall( cudaMallocManaged( &d_box_abs,  TOT_BOXES  * sizeof(float) )     );
  CudaSafeCall( cudaMallocManaged( &d_boxVol,   TOT_BOXES        * sizeof(float) )         );

  CudaSafeCall( cudaMallocManaged( &d_f,      NDIM * NBINS  * sizeof(float) )     );
  CudaSafeCall( cudaMallocManaged( &d_f_sq,   NDIM * NBINS  * sizeof(float) )     );
  CudaSafeCall( cudaMallocManaged( &d_f_abs,  NDIM * NBINS  * sizeof(float) )     );

  CudaSafeCall( cudaMallocManaged( &d_sum_f,     (NUM_BLOCKS) * sizeof(float) )         );
  CudaSafeCall( cudaMallocManaged( &d_sum_f_sq,  (NUM_BLOCKS) * sizeof(float) )         );
  CudaSafeCall( cudaMallocManaged( &d_sum_f_abs, (NUM_BLOCKS) * sizeof(float) )         );
  CudaSafeCall( cudaMemset( d_sum_f, 0,  NUM_BLOCKS * sizeof(float)   ) );
  CudaSafeCall( cudaMemset( d_sum_f_sq, 0, NUM_BLOCKS * sizeof(float) ) );
  CudaSafeCall( cudaMemset( d_sum_f_abs, 0, NUM_BLOCKS * sizeof(float) ) );

  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gputime, start, stop);
  cout << "INFO: CUDA: Time to allocate managed memory = " << gputime << " ms" << endl;
  CudaCheckError();
  log_overhead << _SEP_ << gputime;

  // setup random generator
  curandState *randomStates;
  cudaMallocManaged( (void **)&randomStates , THREADS_PER_BLOCK * THREADS_PER_BLOCK * sizeof(curandState) );

  cudaEventRecord(start, 0);  
  setup_random<<< THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>( randomStates );
  cudaDeviceSynchronize();
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gputime, start, stop);
  cout << "INFO: CUDA: Time for setup random numbers = " << gputime << " ms" << endl;
  CudaCheckError();
  log_overhead << _SEP_ << gputime;

  // Initialize bin edges
  float totVol = 1.;
  for( int d = 0 ; d < NDIM ; ++d ) {
    xmin[d] = 0.0; xmax[d] = 1.0;
    totVol *= ( xmax[d] - xmin[d] );
  }
  // calculate the Jacobean factor: volume/(avg # of calls/bin)
  //  const float jacobian = totVol * pow( (float)NBINS, (float)NDIM ) / CALLS; 
  //const float jacobian = totVol / CALLS_PER_BOX;
  const float jacobian = 1. / totVol / CALLS_PER_BOX;
  cout << "INFO: Jacobian factor = " << jacobian << endl;
  
  cudaEventRecord(start, 0);  
  InitializeEdges<<< (NBINS+1), NDIM >>>( d_x, d_xl, d_xh, xmin, xmax, NBINS, NDIM );
  cudaDeviceSynchronize();  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gputime, start, stop);
  cout << "INFO: CUDA: Time for setup bin edges = " << gputime << " ms" << endl;
  log_overhead << _SEP_ << gputime;
  DumpEdges( d_x, NBINS, NDIM );

  float Ibest    = 0.;
  float * Iiter = new float[NITER];
  float * Viter = new float[NITER];
  float inv_var_tot  = 0.;
  float chi2     = 0.;


  for( int iIter = 0 ; iIter < NITER ; ++iIter ) {
    cout << "INFO: Iteration no. " << iIter << endl;

    stringstream log;

    log << iIter << _SEP_ << log_overhead.str();

    cudaEventRecord(start, 0);  
    CalcBinWidth<<< NBINS, NDIM >>>( d_bw, d_xl, d_xh, NBINS, NDIM );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);
    cout << "INFO: CUDA: Time for bin widths = " << gputime << " ms" << endl;
    DumpBinWidths( d_bw, NBINS, NDIM );
    CudaCheckError();
    log << _SEP_ << gputime;

    unsigned int bvdim = ( TOT_BOXES / THREADS_PER_BLOCK ) + ( (TOT_BOXES % THREADS_PER_BLOCK) ? 1 : 0 );
    cudaEventRecord(start, 0);  
    CalcBoxVolume<<< bvdim, THREADS_PER_BLOCK >>>( d_boxVol, d_bw, NBINS, NDIM, TOT_BOXES );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);
    cout << "INFO: CUDA: Time for box volumes = " << gputime << " ms" << endl;
    // DumpBoxVolumes( d_boxVol, NBINS, NDIM );
    CudaCheckError();
    log << _SEP_ << gputime;

    cudaEventRecord(start, 0);  
    CudaSafeCall( cudaMemset( d_f_abs, 0, NDIM * NBINS * sizeof(float) ) );
    DoSampling<<< NUM_BLOCKS, THREADS_PER_BLOCK >>>( d_box, d_box_sq, d_f_abs, randomStates, d_xl, d_bw, jacobian,
						     NBINS, NDIM, TOT_BOXES, CALLS_PER_BOX );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);
    cout << "INFO: CUDA: Time for sampling = " << gputime << " ms" << endl;
    CudaCheckError();
    //DumpBoxContent( d_box, TOT_BOXES, NDIM );
    log << _SEP_ << gputime;

    /*
    // now project on axes (remember: p(x) is assumed to be factorizable)
    cudaEventRecord(start, 0);
    CudaSafeCall( cudaMemset( d_f,     0, NDIM * NBINS * sizeof(float)   ) );
    CudaSafeCall( cudaMemset( d_f_sq,  0, NDIM * NBINS * sizeof(float) ) );
    CudaSafeCall( cudaMemset( d_f_abs, 0, NDIM * NBINS * sizeof(float) ) );
    cudaDeviceSynchronize(); 

    ProjectOnAxis<<< NBINS, NDIM >>>( d_f,     d_box,     NBINS, NDIM, TOT_BOXES );
    ProjectOnAxis<<< NBINS, NDIM >>>( d_f_sq,  d_box_sq,  NBINS, NDIM, TOT_BOXES );
    ProjectOnAxis<<< NBINS, NDIM >>>( d_f_abs, d_box_abs, NBINS, NDIM, TOT_BOXES );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);
    cout << "INFO: CUDA: Time to project boxes on axes = " << gputime << " ms" << endl;
    CudaCheckError();
    DumpProjection( d_f, NBINS, NDIM );
    */

    // accumulate results - first pass
    cudaEventRecord(start, 0);  
    ReduceBoxes<<< NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(float) >>>( d_sum_f,     d_box,     d_boxVol, totVol, TOT_BOXES, CALLS_PER_BOX );
    ReduceBoxes<<< NUM_BLOCKS, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(float) >>>( d_sum_f_sq,  d_box_sq,  d_boxVol, totVol, TOT_BOXES, CALLS_PER_BOX );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);
    cout << "INFO: CUDA: Time for accumulation (first step) = " << gputime << " ms" << endl;
    CudaCheckError();
    //DumpBoxContent( d_sum_f_sq, NUM_BLOCKS, 1 );
    log << _SEP_ << gputime;

    // accumulate results - second pass
    cudaEventRecord(start, 0);  
    Accumulate<<< 1, NUM_BLOCKS, NUM_BLOCKS*sizeof(float) >>>( d_sum_f    , d_sum_f,     NUM_BLOCKS );
    Accumulate<<< 1, NUM_BLOCKS, NUM_BLOCKS*sizeof(float) >>>( d_sum_f_sq , d_sum_f_sq,  NUM_BLOCKS );
    cudaDeviceSynchronize();  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);
    cout << "INFO: CUDA: Time for accumulation (second step) = " << gputime << " ms" << endl;
    CudaCheckError();
    log << _SEP_ << gputime;

    float I = d_sum_f[0];
    float E = d_sum_f_sq[0];
    
    float var = ( E - I*I ) / ( CALLS - 1 );
    cout << "INFO: \\int{f(x)} = " <<  I << " \\pm " << sqrt(var) << endl;

    log << _SEP_ << I << _SEP_ << sqrt(var);
  
    Iiter[iIter] = I;
    Viter[iIter] = E;

    Ibest       += I * (I*I/var);
    inv_var_tot += I * I / var ;
  
    if( NITER > 1 ) {
        cudaEventRecord(start, 0);  
	RefineGPU<<< NDIM, 1 >>>( d_x, d_xl, d_xh, d_f_abs, d_bw, totVol, ALPHA, NBINS, NDIM, CALLS_PER_BOX );
	cudaDeviceSynchronize();  
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gputime, start, stop);
	cout << "INFO: CUDA: Time for rebinning = " << gputime << " ms" << endl;

        log << _SEP_ << gputime;
	//      RebinCPU( d_x, d_box_abs, NBINS, NDIM );
	DumpEdges( d_x, NBINS, NDIM );
    }
    else {
      log << _SEP_ << 0.0;
    }

    cout << "LOG: " << log.str() << endl;
  }

  DumpEdges( d_x, NBINS, NDIM );

  cout << "INFO: Final results" << endl;
  
  Ibest /= inv_var_tot;
  float sigma_best = Ibest / sqrt( inv_var_tot );
  
  printf( "INFO: I_best = %f \\pm %f\n", Ibest, sigma_best );

  for( int m = 0 ; m < NITER ; ++m ) {
    chi2 += ( Iiter[m] - Ibest ) * ( Iiter[m] - Ibest ) / Viter[m];
  }
  double chi2ndf = chi2 / (float)( NITER-1 );
  printf( "INFO: chi2 / ndf = %f / %i = %f\n", chi2, NITER, chi2ndf );
  
  //cout << "INFO: Finished. Flushing memory." << endl;
  // flush memory
  cudaFree( d_x );
  cudaFree( d_xl );
  cudaFree( d_xh );
  cudaFree( d_box );
  cudaFree( d_box_sq );
  cudaFree( d_box_abs );
  cudaFree( d_boxVol );
  cudaFree( randomStates );
  cudaFree( xmin );
  cudaFree( xmax );
  cudaFree( d_bw );
  cudaFree( d_f );
  cudaFree( d_f_sq );
  cudaFree( d_f_abs );
  cudaFree( d_sum_f );
  cudaFree( d_sum_f_sq );
  cudaFree( d_sum_f_abs );
  // cudaFree( function_table );
  // cout << "DEBUG: GPU memory freed" << endl;

  delete [] Iiter;
  delete [] Viter;

  cout << "INFO: Finished. GPU memory cleared." << endl;

  exit( success );
}
