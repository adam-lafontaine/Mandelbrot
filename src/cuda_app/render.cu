#include "render.hpp"
#include "cuda_def.cuh"

constexpr int THREADS_PER_BLOCK = 1024;

/*
GPU_FUNCTION
static u8 rgb_grayscale_standard(u8 red, u8 green, u8 blue)
{
    return static_cast<u8>(0.299f * red + 0.587f * green + 0.114f * blue);
}
*/


GPU_KERNAL
static void gpu_set_color(pixel_t* dst, u32 n_pixels)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_pixels)
    {
        return;
    }

    pixel_t p = {};
    p.alpha = 255;
    p.red = 255;
    p.green = 255;
    p.blue = 255;

    dst[i] = p;
}


void render(AppState& state)
{
    auto& d_screen = state.device.pixels;
    u32 n_pixels = d_screen.width * d_screen.height;
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks = (n_pixels + threads_per_block - 1) / threads_per_block;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_set_color<<<blocks, threads_per_block>>>(
        d_screen.data,
        n_pixels
    );

    proc &= cuda_launch_success();
    assert(proc);

    auto& h_screen = state.screen_buffer;
    proc &= copy_to_host(d_screen, h_screen);
    assert(proc);
}