#CUDA_DIR=/usr/local/cuda-5.5
CUDA_DIR=/usr/local/cuda-6.0
#CUDA_DIR=/opt/cuda
CUDA_SDK_DIR=/opt/cuda/sdk

#INCFLAG=-I$(CUDA_DIR)/include -I$(CUDA_SDK_DIR)/C/common/inc -I..
INCFLAG=-I$(CUDA_DIR)/include -I..

CULIBS = -L$(CUDA_DIR)/lib64 -lcudart

# CUDA code generation flags
GENCODE_SM10    := -gencode arch=compute_10,code=sm_10
GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30 
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_SM35    := -gencode arch=compute_35,code=sm_35
GENCODE_FLAGS   := $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM20) #$(GENCODE_SM30)

NVCC = nvcc
#NVCCFLAGS = -O3 $(INCFLAG) --ptxas-options=-v -use_fast_math $(GENCODE_FLAGS)
NVCCFLAGS = -g -G $(INCFLAG) $(GENCODE_SM30) #-DTHRUST_DEBUG
# --verbose
#NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM13) #use to compile on l2gpu with GTX285
#NVCCFLAGS = -O3 $(INCFLAG)  --use_fast_math $(GENCODE_SM30)


#LIBS=-L/home/wittich/src/cudpp_install_2.0/lib -lcudpp

%.o: %.c
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: simple vegas vegas_multidim

simple: simple.cu
	$(NVCC) $(NVCCFLAGS) $(CULIBS) $^ -o $@

vegas: vegas.cu
	$(NVCC) $(NVCCFLAGS) $(CULIBS) $^ -o $@

vegas_multidim: vegas_multidim.cu
	$(NVCC) $(NVCCFLAGS) $(CULIBS) $^ -o $@

clean:
	$(RM) *.o *~ simple vegas vegas_multidim

depend:
	makedepend -Y $(INCFLAG) *.cu *.cc *.h

# DO NOT DELETE

