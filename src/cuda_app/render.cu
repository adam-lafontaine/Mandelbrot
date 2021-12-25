#include "render.hpp"
#include "cuda_def.cuh"

#include <cassert>


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

    u32* iterations_dst;
    u32 width;

    Range2Du32 r1;
    Range2Du32 r2;
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
    u32* iterations;

    u32 cr = 0;
	u32 cg = 0;
	u32 cb = 0;
    DeviceColorPalette palette;    

    u32* min_iter;
    u32* max_iter;

    pixel_t* pixels_dst;
};


class CopyProps
{
public:
    u32* src;
    u32* dst;

    Range2Du32 r_src;
    Range2Du32 r_dst;

    u32 width;
};


namespace gpu
{
/**********/


GPU_FUNCTION
static Point2Du32 get_position(Range2Du32 const& r, u32 width, u32 r_id)
{
    auto w1 = r.x_end - r.x_begin;

    assert(r_id < w1 * (r.y_end - r.y_begin));

    auto h = r_id / w1;

    Point2Du32 pt{};
    pt.x = r.x_begin + r_id - w1 * h;
    pt.y = r.y_begin + h;

    return pt;
}


GPU_FUNCTION
static Point2Du32 get_position(Range2Du32 const& r1, Range2Du32 const& r2, u32 width, u32 r_id)
{
    auto w1 = r1.x_end - r1.x_begin;
    auto h1 = r1.y_end - r1.y_begin;

    if(r_id < w1 * h1)
    {
        return get_position(r1, width, r_id);
    }
    
    return get_position(r2, width, r_id - w1 * h1);
}


GPU_FUNCTION
static u32 get_index(u32 x, u32 y, u32 width)
{
    return width * y + x;
}


GPU_FUNCTION
static u32 get_index(Point2Du32 const& pos, u32 width)
{
    return get_index(pos.x, pos.y, width);
}


GPU_FUNCTION
static void mandelbrot_by_range_id(MandelbrotProps const& props, u32 r_id)
{
    auto const width = props.width;
    auto pos = get_position(props.r1, props.r2, width, r_id);

    r64 const ci = props.min_im + pos.y * props.im_step;
    u32 iter = 0;
    r64 const cr = props.min_re + pos.x * props.re_step;

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

    auto i = gpu::get_index(pos, width);

    props.iterations_dst[i] = iter - 1;
}


GPU_FUNCTION
static void draw(DrawProps const& props, u32 i)
{
    auto iter_min = *props.min_iter;
    auto iter_max = *props.max_iter;
    //auto diff = max - min;
    pixel_t p{};
    p.alpha = 255;
    
    auto iter = props.iterations[i];
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

    props.pixels_dst[i] = p;
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
static void gpu_mandelbrot(MandelbrotProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    gpu::mandelbrot_by_range_id(props, t);
}


GPU_KERNAL
static void gpu_draw(DrawProps props, u32 n_threads)
{    
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    gpu::draw(props, t);
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


GPU_KERNAL
static void gpu_copy(CopyProps props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto src_pos = gpu::get_position(props.r_src, props.width, t);
    auto dst_pos = gpu::get_position(props.r_dst, props.width, t);

    auto src_i = gpu::get_index(src_pos, props.width);
    auto dst_i = gpu::get_index(dst_pos, props.width);

    props.dst[dst_i] = props.src[src_i];
}


static void copy(DeviceMatrix const& mat, Vec2Di32 const& direction)
{
    Range2Du32 r_src{};
    r_src.x_begin = 0;
    r_src.x_end = mat.width;
    r_src.y_begin = 0;
    r_src.y_end = mat.height;
    auto r_dst = r_src;

	auto const n_cols = static_cast<u32>(std::abs(direction.x));
	auto const n_rows = static_cast<u32>(std::abs(direction.y));

    if (n_cols == 0 && n_rows == 0)
	{
		return;
	}    

    auto right = direction.x > 0;
    auto left = direction.x < 0;
    auto up = direction.y < 0;
    auto down = direction.y > 0; 

    if(right)
    {
        r_src.x_end -= n_cols;
        r_dst.x_begin += n_cols;
    }
    
    if(left)
    {
        r_src.x_begin += n_cols;
        r_dst.x_end -= n_cols;
    }

    if(up)
    {
        r_src.y_begin += n_rows;
        r_dst.y_end -= n_rows;
    }

    if(down)
    {
        r_src.y_end -= n_rows;
        r_dst.y_begin += n_rows;
    }

    CopyProps props{};
    props.src = mat.data_src;
    props.dst = mat.data_dst;
    props.r_src = r_src;
    props.r_dst = r_dst;
    props.width = mat.width;

    auto n_threads = (r_src.x_end - r_src.x_begin) * (r_src.y_end - r_src.y_begin);

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_copy<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


static void mandelbrot(DeviceMatrix const& dst, AppState& state)
{
    u32 width = dst.width;
    u32 height = dst.height;

    auto& shift = state.pixel_shift;

	auto do_left = shift.x > 0;
	auto do_top = shift.y > 0;

    auto const n_cols = static_cast<u32>(std::abs(shift.x));
	auto const n_rows = static_cast<u32>(std::abs(shift.y));

	auto no_horizontal = n_cols == 0;
	auto no_vertical = n_rows == 0;

    Range2Du32 r1{};
    r1.x_begin = 0;
    r1.x_end = width;
    r1.y_begin = 0;
    r1.y_end = height;
    
    Range2Du32 r2{};
    r2.x_begin = 0;
    r2.x_end = 0;
    r2.y_begin = 0;
    r2.y_end = 0;

    if (no_horizontal && no_vertical)
	{

	}
	else if (no_horizontal)
	{
		if (do_top)
		{
			r1.y_end = n_rows;
		}
		else // if (do_bottom)
		{
			r1.y_begin = height - 1 - n_rows;
		}
	}
	else if (no_vertical)
	{
		if (do_left)
		{
			r1.x_end = n_cols;
		}
		else // if (do_right)
		{
			r1.x_begin = width - 1 - n_cols;
		}
	}
    else
    {
        r2 = r1;
        
        if (do_top)
        {
            r1.y_end = n_rows;
            r2.y_begin = n_rows;
        }
        else // if (do_bottom)
        {
            r1.y_begin = height - 1 - n_rows;
            r2.y_end = height - 1 - n_rows;
        }

        if (do_left)
        {
            r2.x_end = n_cols;
        }
        else // if (do_right)
        {
            r2.x_begin = width - 1 - n_cols;
        }
    }    

    u32 n_elements = (r1.x_end - r1.x_begin) * (r1.y_end - r1.y_begin) +
        (r2.x_end - r2.x_begin) * (r2.y_end - r2.y_begin);
    
    MandelbrotProps props{};
    props.iter_limit = state.iter_limit;
	props.min_re = MBT_MIN_X + state.mbt_pos.x;
	props.min_im = MBT_MIN_Y + state.mbt_pos.y;
	props.re_step = state.mbt_screen_width / width;
	props.im_step = state.mbt_screen_height / height;
    props.width = width;
    props.iterations_dst = dst.data_dst;
    props.r1 = r1;
    props.r2 = r2;

    auto n_threads = n_elements;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_mandelbrot<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
}


static void find_min_max_iter(AppState& state)
{
    u32 width = state.device.iterations.width;
    u32 height = state.device.iterations.height;
    u32 n_elements = width * height;

    MinMaxProps props{};
    props.values = state.device.iterations.data_dst;
    props.n_values = n_elements;
    props.thread_list_max = state.device.max_iters.data;
    props.thread_list_min = state.device.min_iters.data;
    props.n_threads = state.device.min_iters.n_elements;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_find_min_max<<<calc_thread_blocks(props.n_threads), THREADS_PER_BLOCK>>>(props);

    proc &= cuda_launch_success();
    assert(proc);
}


static void draw(image_t const& dst, AppState const& state)
{
    u32 width = dst.width;
    u32 height = dst.height;
    u32 n_elements = width * height;

    DrawProps props{};
    props.iterations = state.device.iterations.data_dst;
    props.palette = state.device.palette;
    props.pixels_dst = state.device.pixels.data;
    props.min_iter = state.device.min_iters.data;
    props.max_iter = state.device.max_iters.data;
    set_rgb_channels(props.cr, props.cg, props.cb, state.rgb_option);

    auto n_threads = n_elements;

    bool proc = cuda_no_errors();
    assert(proc);

    gpu_draw<<<calc_thread_blocks(n_threads), THREADS_PER_BLOCK>>>(props, n_threads);

    proc &= cuda_launch_success();
    assert(proc);
    
    proc &= copy_to_host(state.device.pixels, dst);
    assert(proc);
}


void render(AppState& state)
{
    if(state.render_new)
    {
        auto temp = state.device.iterations.data_dst;
        state.device.iterations.data_dst = state.device.iterations.data_src;
        state.device.iterations.data_src = temp;

        copy(state.device.iterations, state.pixel_shift);
        mandelbrot(state.device.iterations, state);
        
        find_min_max_iter(state);

        state.draw_new = true;
    }
    
    if(state.draw_new)
    {
        draw(state.screen_buffer, state);
    }
}