#pragma once

#include <cuda_runtime.h>

#define GPU_KERNAL __global__
#define GPU_FUNCTION __device__
#define HOST_FUNCTION __host__
#define GPU_GLOBAL_VARIABLE __device__
#define GPU_GLOBAL_CONSTANT __constant__
#define GPU_BLOCK_VARIABLE __shared__

#define CUDA_PRINT_ERROR

#define cuda_barrier __syncthreads