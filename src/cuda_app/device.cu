#include "device.hpp"
#include "cuda_def.cuh"

#include <cassert>

#ifdef CUDA_PRINT_ERROR

#include <cstdio>
#include <cstring>

#endif


static void check_error(cudaError_t err, cstr label = "")
{
    if(err == cudaSuccess)
    {
        return;
    }

    #ifdef CUDA_PRINT_ERROR
    #ifndef	NDEBUG

    printf("\n*** CUDA ERROR ***\n\n");
    printf("%s", cudaGetErrorString(err));

    if(std::strlen(label))
    {
        printf("\n%s", label);
    }
    
    printf("\n\n******************\n\n");

    #endif
    #endif
}


namespace cuda
{
    bool device_malloc(DevicePointer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data);

        if(!n_bytes || buffer.data)
        {
            return false;
        }

        cudaError_t err = cudaMalloc((void**)&(buffer.data), n_bytes);
        check_error(err, "device_malloc");

        bool result = err == cudaSuccess;

        assert(result);

        return result;
    }


    bool unified_malloc(DevicePointer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data);

        if(!n_bytes || buffer.data)
        {
            return false;
        }

        cudaError_t err = cudaMallocManaged((void**)&(buffer.data), n_bytes);
        check_error(err, "unified_malloc");

        bool result = err == cudaSuccess;

        assert(result);

        return result;
    }


    bool free(void* data)
    {
        if(data)
        {
            return true;
        }

        cudaError_t err = cudaFree(data);
        check_error(err, "free");

        return err == cudaSuccess;
    }


    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes)
    {
        cudaError_t err = cudaMemcpy(device_dst, host_src, n_bytes, cudaMemcpyHostToDevice);
        check_error(err, "memcpy_to_device");

        bool result = err == cudaSuccess;

        assert(result);

        return result;
    }


    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes)
    {
        cudaError_t err = cudaMemcpy(host_dst, device_src, n_bytes, cudaMemcpyDeviceToHost);
        check_error(err, "memcpy_to_host");

        bool result = err == cudaSuccess;

        assert(result);

        return result;
    }


    bool no_errors(cstr label)
    {
        cudaError_t err = cudaGetLastError();
        check_error(err, label);

        return err == cudaSuccess;
    }


    bool launch_success(cstr label)
    {
        cudaError_t err = cudaDeviceSynchronize();
        check_error(err, label);

        return err == cudaSuccess;
    }
}