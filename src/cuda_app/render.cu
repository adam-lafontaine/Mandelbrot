#include "render.hpp"
#include "cuda_def.cuh"

constexpr int THREADS_PER_BLOCK = 1024;
/*
class MandelbrotProps
{
public:
    u32 max_iter;
    r64 min_re;
    r64 min_im;
    r64 re_step;
    r64 im_step;
    u32* iterations;
    u32 n_elements;
};


GPU_KERNAL
static void mandelbrot(MandelbrotProps* props)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= props->n_elements)
    {
        return;
    }


}
*/


GPU_KERNAL
static void gpu_set_color(pixel_t* dst, u32 width, u32 height)
{
    auto n_pixels = width * height;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_pixels)
    {
        return;
    }

    // i = y * width + x

    pixel_t p = {};
    p.alpha = 255;
    p.red = 0;
    p.green = 0;
    p.blue = 0;

    auto y = i / width;
    auto x = i - y * width;

    if(y < height / 3)
    {
        p.red = 255;
    }
    else if(y < height * 2 / 3)
    {
        p.green = 255;
    }
    else
    {
        p.blue = 255;
    }

    if(x < width / 3)
    {
        p.red = 255;
    }
    else if(x < width * 2 / 3)
    {
        p.green = 255;
    }
    else
    {
        p.blue = 255;
    }

    dst[i] = p;
}





void render(AppState& state)
{
    auto& d_screen = state.device.pixels;
    u32 n_pixels = d_screen.width * d_screen.height;
    int threads_per_block = THREADS_PER_BLOCK;
    int blocks = (n_pixels + threads_per_block - 1) / threads_per_block;
    /*
    MandelbrotProps m_props{};
    m_props.max_iter = state.max_iter;
	m_props.min_re = MBT_MIN_X + state.screen_pos.x;
	m_props.min_im = MBT_MIN_Y + state.screen_pos.y;
	m_props.re_step = screen_width(state) / width;
	m_props.im_step = screen_height(state) / height;
    */

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_set_color<<<blocks, threads_per_block>>>(
        d_screen.data,
        d_screen.width,
        d_screen.height
    );

    proc &= cuda_launch_success();
    assert(proc);

    auto& h_screen = state.screen_buffer;
    proc &= copy_to_host(d_screen, h_screen);
    assert(proc);
}