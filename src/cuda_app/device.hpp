#pragma once

#include "../utils/types.hpp"

#include <cstddef>



namespace cuda
{
    bool device_malloc(void** addr, size_t n_bytes);

    bool unified_malloc(void** addr, size_t n_bytes);

    bool free(void* data);


    bool memcpy_to_device(const void* host_src, void* device_dst, size_t n_bytes);

    bool memcpy_to_host(const void* device_src, void* host_dst, size_t n_bytes);


    bool no_errors(cstr label);

    bool launch_success(cstr label);
}


#define device_addr(ptr) (void**)&(ptr)