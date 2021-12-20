#include "render.hpp"
#include "cuda_def.cuh"

constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


class MandelbrotProps
{
public:
    u32 max_iter;
    r64 min_re;
    r64 min_im;
    r64 re_step;
    r64 im_step;
    DeviceMatrix iterations;
};


GPU_KERNAL
static void gpu_mandelbrot(MandelbrotProps props)
{
    auto const width = props.iterations.width;
    auto const height = props.iterations.height;
    auto n_elements = width * height;
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_elements)
    {
        return;
    }

    auto y = i / width;
    auto x = i - y * width;

    r64 const ci = props.min_im + y * props.im_step;
    u32 iter = 0;
    r64 const cr = props.min_re + x * props.re_step;

    r64 re = 0.0;
    r64 im = 0.0;
    r64 re2 = 0.0;
    r64 im2 = 0.0;

    while (iter < props.max_iter && re2 + im2 <= 4.0)
    {
        im = (re + re) * im + ci;
        re = re2 - im2 + cr;
        im2 = im * im;
        re2 = re * re;

        ++iter;
    }

    props.iterations.data[i] = props.iterations.data_mirror[i] = iter - 1;
}



GPU_KERNAL
static void gpu_set_color(DeviceImage image)
{
    auto const width = image.width;
    auto const height = image.height;
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

    image.data[i] = p;
}



GPU_FUNCTION
static u32 min_value(u32* sorted)
{
    return sorted[0];
}


GPU_FUNCTION
static u32 max_value(u32* sorted)
{
    return sorted[1];
}


GPU_KERNAL
static void gpu_sort_high_low(u32* values, u32 n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_elements)
    {
        return;
    }

    u32* low = values;
    u32* high = values + n_elements / 2;

    if(high[i] < low[i])
    {
        auto h = low[i];
        low[i] = high[i];
        high[i] = h;
    }    
}


GPU_KERNAL
static void gpu_reduce_min_max(u32* values, u32 n_elements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_elements)
    {
        return;
    }

    auto half = n_elements / 2;

    if(i < half)
    {
        values[i] = min(values[i], values[i + 1]);
    }
    else
    {
        values[i] = max(values[i + half], values[i + half + 1]);
    }
}


HOST_FUNCTION
static void sort_min_max(DeviceMatrix& mat)
{
    u32 n_elements = mat.width * mat.height;

    assert(n_elements % 2 == 0);

    auto values = mat.data_mirror;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_sort_high_low<<<calc_thread_blocks(n_elements), THREADS_PER_BLOCK>>>(values, n_elements);

    proc &= cuda_launch_success();
    assert(proc);

    for(u32 n = n_elements / 2; n > 1; n /= 2)
    {
        gpu_reduce_min_max<<<calc_thread_blocks(n), THREADS_PER_BLOCK>>>(values, n);

        proc &= cuda_launch_success();
        assert(proc);
    }
}


void render(AppState& state)
{
    auto& d_screen = state.device.pixels;
    u32 n_pixels = d_screen.width * d_screen.height;
    int blocks = calc_thread_blocks(n_pixels);
    
    MandelbrotProps m_props{};
    m_props.max_iter = state.max_iter;
	m_props.min_re = MBT_MIN_X + state.mbt_pos.x;
	m_props.min_im = MBT_MIN_Y + state.mbt_pos.y;
	m_props.re_step = state.mbt_screen_width / d_screen.width;
	m_props.im_step = state.mbt_screen_height / d_screen.height;
    m_props.iterations = state.device.iterations;    

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_mandelbrot<<<blocks, THREADS_PER_BLOCK>>>(m_props);

    proc &= cuda_launch_success();
    assert(proc);

    sort_min_max(state.device.iterations);

    gpu_set_color<<<blocks, THREADS_PER_BLOCK>>>(d_screen);

    proc &= cuda_launch_success();
    assert(proc);

    auto& h_screen = state.screen_buffer;
    proc &= copy_to_host(d_screen, h_screen);
    assert(proc);
}