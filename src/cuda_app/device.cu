#include "device.hpp"
#include "cuda_def.cuh"

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


namespace device
{
    bool malloc(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data);

        if(!n_bytes || buffer.data)
        {
            return false;
        }

        cudaError_t err = cudaMalloc((void**)&(buffer.data), n_bytes);
        check_error(err);

        bool result = err == cudaSuccess;

        if(result)
        {
            buffer.capacity = n_bytes;
        }
        
        return result;
    }


    bool unified_malloc(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(n_bytes);
        assert(!buffer.data);

        if(!n_bytes || buffer.data)
        {
            return false;
        }

        cudaError_t err = cudaMallocManaged((void**)&(buffer.data), n_bytes);
        check_error(err);

        bool result = err == cudaSuccess;

        if(result)
        {
            buffer.capacity = n_bytes;
        }
        
        return result;
    }


    bool free(MemoryBuffer& buffer)
    {
        buffer.capacity = 0;
        buffer.size = 0;

        if(buffer.data)
        {
            cudaError_t err = cudaFree(buffer.data);
            check_error(err);

            buffer.data = nullptr;

            return err == cudaSuccess;
        }

        return true;
    }


    u8* push_bytes(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(buffer.data);
        assert(buffer.capacity);
        assert(buffer.size < buffer.capacity);

        auto is_valid = 
            buffer.data &&
            buffer.capacity &&
            buffer.size < buffer.capacity;

        auto bytes_available = (buffer.capacity - buffer.size) >= n_bytes;
        assert(bytes_available);

        if(!is_valid || !bytes_available)
        {
            return nullptr;
        }

        auto data = buffer.data + buffer.size;

        buffer.size += n_bytes;

        return data;
    }


    bool pop_bytes(MemoryBuffer& buffer, size_t n_bytes)
    {
        assert(buffer.data);
        assert(buffer.capacity);
        assert(buffer.size <= buffer.capacity);
        assert(n_bytes <= buffer.capacity);
        assert(n_bytes <= buffer.size);

        auto is_valid = 
            buffer.data &&
            buffer.capacity &&
            buffer.size <= buffer.capacity &&
            n_bytes <= buffer.capacity &&
            n_bytes <= buffer.size;

        if(is_valid)
        {
            buffer.size -= n_bytes;
            return true;
        }

        return false;
    }


    bool push_device_image(MemoryBuffer& buffer, DeviceImage& image, u32 width, u32 height)
    {
        auto data = push_bytes(buffer, width * height * sizeof(Pixel));

        if(data)
        {
            image.width = width;
            image.height = height;
            image.data = (Pixel*)data;

            return true;
        }

        return false;
    }


    bool push_device_matrix(MemoryBuffer& buffer, DeviceMatrix& matrix, u32 width, u32 height)
    {
        auto bytes_per = width * height * sizeof(u32);
        auto src_data = push_bytes(buffer, bytes_per);

        if(!src_data)
        {
            return false;
        }

        auto dst_data = push_bytes(buffer, bytes_per);
        if(!dst_data)
        {
            pop_bytes(buffer, bytes_per);
            return false;
        }

        matrix.width = width;
        matrix.height = height;
        matrix.data_src = (u32*)src_data;
        matrix.data_dst = (u32*)dst_data;

        return true;
    }


    bool push_device_palette(MemoryBuffer& buffer, DeviceColorPalette& palette, u32 n_colors)
    {
        auto bytes_per_channel = sizeof(u8) * n_colors;
        size_t bytes_allocated = 0;

        for(u32 c = 0; c < RGB_CHANNELS; ++c)
        {
            auto data = push_bytes(buffer, bytes_per_channel);
            if(!data)
            {
                break;                
            }

            bytes_allocated += bytes_per_channel;
            palette.channels[c] = (u8*)data;
        }

        if(bytes_allocated == RGB_CHANNELS * bytes_per_channel)
        {
            palette.n_colors = n_colors;
            return true;
        }
        else if (bytes_allocated > 0)
        {
            pop_bytes(buffer, bytes_allocated);            
        }

        return false;
    }
}
