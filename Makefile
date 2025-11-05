% : %.cu
	nvcc -arch sm_86 --generate-line-info -o $@ $@.cu

all: mma_m8n8k4 mma_m16n8k16
