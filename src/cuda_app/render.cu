#include "render.hpp"
#include "cuda_def.cuh"


static void set_rgb_channels(u32& cr, u32& cg, u32& cb, u32 rgb_option)
{
	switch (rgb_option)
	{
	case 1:
		cr = 0;
		cg = 1;
		cb = 2;
		break;
	case 2:
		cr = 0;
		cg = 2;
		cb = 1;
		break;
	case 3:
		cr = 1;
		cg = 0;
		cb = 2;
		break;
	case 4:
		cr = 1;
		cg = 2;
		cb = 0;
		break;
	case 5:
		cr = 2;
		cg = 0;
		cb = 1;
		break;
	case 6:
		cr = 2;
		cg = 1;
		cb = 0;
		break;
	}
}


constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


class MandelbrotProps
{
public:
    u32 iter_limit;
    r64 min_re;
    r64 min_im;
    r64 re_step;
    r64 im_step;
    DeviceMatrix iterations_dst;

    u32 n_threads;
};


class MinMaxProps
{
public:
    u32* values;
    u32 n_values;

    u32* thread_list_min;
    u32* thread_list_max;

    u32 n_threads;
};


class DrawProps
{
public:    
    DeviceMatrix iterations;

    u32 cr = 0;
	u32 cg = 0;
	u32 cb = 0;
    DeviceColorPalette palette;    

    u32* min_iter;
    u32* max_iter;

    DeviceImage pixels_dst;

    u32 n_threads;
};


namespace gpu
{
/**********/


GPU_FUNCTION
static void mandelbrot(MandelbrotProps const& props, u32 i)
{
    auto const width = props.iterations_dst.width;

    auto y = i / width;
    auto x = i - y * width;

    r64 const ci = props.min_im + y * props.im_step;
    u32 iter = 0;
    r64 const cr = props.min_re + x * props.re_step;

    r64 re = 0.0;
    r64 im = 0.0;
    r64 re2 = 0.0;
    r64 im2 = 0.0;

    while (iter < props.iter_limit && re2 + im2 <= 4.0)
    {
        im = (re + re) * im + ci;
        re = re2 - im2 + cr;
        im2 = im * im;
        re2 = re * re;

        ++iter;
    }

    props.iterations_dst.data[i] = iter - 1;
}


GPU_FUNCTION
static void draw(DrawProps const& props, u32 i)
{
    auto iter_min = *props.min_iter;
    auto iter_max = *props.max_iter;
    //auto diff = max - min;
    pixel_t p{};
    p.alpha = 255;
    
    auto iter = props.iterations.data[i];
    if(iter >= iter_max)
    {
        p.red = 0;
        p.green = 0;
        p.blue = 0;
    }
    else
    {
        auto n_colors = ((iter_max - iter_min) / 64 + 1) * 16;
        n_colors = min(n_colors, props.palette.n_colors);
        auto c = (iter - iter_min) % n_colors * props.palette.n_colors / n_colors;

        p.red = props.palette.channels[props.cr][c];
        p.green = props.palette.channels[props.cg][c];
        p.blue = props.palette.channels[props.cb][c];
    }

    props.pixels_dst.data[i] = p;
}


GPU_FUNCTION
static u32 find_min(u32* values, u32 n_elements)
{
    u32 min = values[0];

    for(u32 i = 1; i < n_elements; ++i)
    {
        if(values[i] < min)
        {
            min = values[i];
        }
    }

    return min;
}


GPU_FUNCTION
static u32 find_max(u32* values, u32 n_elements)
{
    u32 max = values[0];

    for(u32 i = 1; i < n_elements; ++i)
    {
        if(values[i] > max)
        {
            max = values[i];
        }
    }

    return max;
}


/**********/
}



GPU_KERNAL
static void gpu_mandelbrot(MandelbrotProps props)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= props.n_threads)
    {
        return;
    }

    u32 chunk_size = props.n_threads;
    u32 n_chunks = chunk_size / (props.iterations_dst.width * props.iterations_dst.height);

    for(u32 i = 0; i < n_chunks; ++i)
    {
        gpu::mandelbrot(props, t + i * chunk_size);
    }
}


GPU_KERNAL
static void gpu_draw(DrawProps props)
{    
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= props.n_threads)
    {
        return;
    }

    u32 chunk_size = props.n_threads;
    u32 n_chunks = chunk_size / (props.iterations.width * props.iterations.height);

    for(u32 i = 0; i < n_chunks; ++i)
    {
        gpu::draw(props, t + i * chunk_size);
    }    
}


GPU_KERNAL
static void gpu_find_min_max(MinMaxProps props)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= props.n_threads)
    {
        return;
    }
    
    auto chunk_size = (r32)props.n_values / props.n_threads;    

    auto i_begin = lround(t * chunk_size);
    auto i_end = min(lround((t + 1) * chunk_size), (long)props.n_values - 1);

    u32 min = props.values[i_begin];
    u32 max = props.values[i_end];
    u32 val = 0;
    for(u32 i = i_begin + 1; i < i_end; ++i)
    {
        val = props.values[i];
        if(val < min)
        {
            min = val;
        }
        else if(val > max)
        {
            max = val;
        }
    }

    props.thread_list_min[t] = min;
    props.thread_list_max[t] = max;

    cuda_barrier();

    switch(t)
    {
        case 0:
        props.thread_list_min[0] = gpu::find_min(props.thread_list_min, props.n_threads);
        break;

        case 1:
        props.thread_list_max[0] = gpu::find_max(props.thread_list_max, props.n_threads);
        break;

        default:
        break;
    }
}


static void mandelbrot(DeviceMatrix const& dst, AppState& state)
{
    u32 width = dst.width;
    u32 height = dst.height;
    u32 n_elements = width * height;

    u32 n_chunks = 1;
    u32 n_threads = n_elements / n_chunks;
    
    MandelbrotProps mb_props{};
    mb_props.iter_limit = state.iter_limit;
	mb_props.min_re = MBT_MIN_X + state.mbt_pos.x;
	mb_props.min_im = MBT_MIN_Y + state.mbt_pos.y;
	mb_props.re_step = state.mbt_screen_width / width;
	mb_props.im_step = state.mbt_screen_height / height;
    mb_props.iterations_dst = dst;
    mb_props.n_threads = n_threads;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_mandelbrot<<<calc_thread_blocks(mb_props.n_threads), THREADS_PER_BLOCK>>>(mb_props);

    proc &= cuda_launch_success();
    assert(proc);
}


static void find_min_max_iter(AppState& state)
{
    u32 width = state.device.iterations.width;
    u32 height = state.device.iterations.height;
    u32 n_elements = width * height;

    MinMaxProps mm_props{};
    mm_props.values = state.device.iterations.data;
    mm_props.n_values = n_elements;
    mm_props.thread_list_max = state.device.max_iters.data;
    mm_props.thread_list_min = state.device.min_iters.data;
    mm_props.n_threads = state.device.min_iters.n_elements;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_find_min_max<<<calc_thread_blocks(mm_props.n_threads), THREADS_PER_BLOCK>>>(mm_props);

    proc &= cuda_launch_success();
    assert(proc);
}


static void draw(image_t const& dst, AppState const& state)
{
    u32 width = dst.width;
    u32 height = dst.height;
    u32 n_elements = width * height;

    u32 n_chunks = 1;
    u32 n_threads = n_elements / n_chunks;

    DrawProps dr_props{};
    dr_props.iterations = state.device.iterations;
    dr_props.palette = state.device.palette;
    dr_props.pixels_dst = state.device.pixels;
    dr_props.n_threads = n_threads;
    dr_props.min_iter = state.device.min_iters.data;
    dr_props.max_iter = state.device.max_iters.data;
    set_rgb_channels(dr_props.cr, dr_props.cg, dr_props.cb, state.rgb_option);

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_draw<<<calc_thread_blocks(dr_props.n_threads), THREADS_PER_BLOCK>>>(dr_props);

    proc &= cuda_launch_success();
    assert(proc);
    
    proc &= copy_to_host(state.device.pixels, dst);
    assert(proc);
}


void render(AppState& state)
{
    if(state.render_new)
    {
        mandelbrot(state.device.iterations, state);
        find_min_max_iter(state);

        state.draw_new = true;
    }
    
    if(state.draw_new)
    {
        draw(state.screen_buffer, state);
    }
}