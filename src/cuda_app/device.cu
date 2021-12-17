#include "device.hpp"
#include "cuda_def.cuh"

#include <cassert>

#ifdef CUDA_PRINT_ERROR

#include <cstdio>

#endif


static void check_error(cudaError_t err)
{
    if(err == cudaSuccess)
    {
        return;
    }

    #ifdef CUDA_PRINT_ERROR

    printf("\n*** CUDA ERROR ***\n\n");
    printf("%s", cudaGetErrorString(err));
    printf("\n\n******************\n\n");

    #endif
}


static bool cuda_device_malloc(void** ptr, u32 n_bytes)
{
    cudaError_t err = cudaMalloc(ptr, n_bytes);
    check_error(err);
    
    return err == cudaSuccess;
}


static bool cuda_device_free(void* ptr)
{
    cudaError_t err = cudaFree(ptr);
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(device_dst, host_src, n_bytes, cudaMemcpyHostToDevice);
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes)
{
    cudaError_t err = cudaMemcpy(host_dst, device_src, n_bytes, cudaMemcpyDeviceToHost);
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_no_errors()
{
    cudaError_t err = cudaGetLastError();
    check_error(err);

    return err == cudaSuccess;
}


bool cuda_launch_success()
{
    cudaError_t err = cudaDeviceSynchronize();
    check_error(err);

    return err == cudaSuccess;
}


bool device_malloc(DeviceBuffer& buffer, size_t n_bytes)
{
    bool result = cuda_device_malloc((void**)&(buffer.data), n_bytes);
    if(result)
    {
        buffer.total_bytes = n_bytes;
    }

    return result;
}


bool device_free(DeviceBuffer& buffer)
{
    buffer.total_bytes = 0;
    buffer.offset = 0;
    return cuda_device_free(buffer.data);
}


bool make_image(DeviceImage& image, u32 width, u32 height, DeviceBuffer& buffer)
{
    assert(buffer.data);
    auto bytes = width * height * sizeof(pixel_t);

    bool result = buffer.total_bytes - buffer.offset >= bytes;
    if(result)
    {
        image.width = width;
        image.height = height;
        image.data = (pixel_t*)((u8*)buffer.data + buffer.offset);
        buffer.offset += bytes;
    }

    return result;
}


bool copy_to_device(image_t const& src, DeviceImage const& dst)
{
    assert(src.data);
    assert(src.width);
    assert(src.height);
    assert(dst.data);
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    auto bytes = src.width * src.height * sizeof(pixel_t);

    return cuda_memcpy_to_device(src.data, dst.data, bytes);
}


bool copy_to_host(DeviceImage const& src, image_t const& dst)
{
    assert(src.data);
    assert(src.width);
    assert(src.height);
    assert(dst.data);
    assert(dst.width == src.width);
    assert(dst.height == src.height);

    auto bytes = src.width * src.height * sizeof(pixel_t);

    return cuda_memcpy_to_host(src.data, dst.data, bytes);
}


bool make_matrix(DeviceMatrix& matrix, u32 width, u32 height, DeviceBuffer& buffer)
{
    assert(buffer.data);
    auto bytes = width * height * sizeof(u32);

    bool result = buffer.total_bytes - buffer.offset >= bytes;
    if(result)
    {
        matrix.width = width;
        matrix.height = height;
        matrix.data = (u32*)((u8*)buffer.data + buffer.offset);
        buffer.offset += bytes;
    }

    return result;
}
