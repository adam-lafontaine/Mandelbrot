#include "render.hpp"
#include "../app/colors.hpp"
#include "cuda_def.cuh"

#include <cassert>

constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


class DrawProps
{
public:

    Mat2Di32 current_ids;

    Image screen_dst;

    u32 channel1;
    u32 channel2;
    u32 channel3;
};


static void set_rgb_channels(DrawProps& props, u32 rgb_option)
{
    auto& cr = props.channel1;
    auto& cg = props.channel2;
    auto& cb = props.channel3;

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





class RangeList
{
public:
	Range2Du32 copy_src;
	Range2Du32 copy_dst;
	Range2Du32 write_h;
	Range2Du32 write_v;
};


RangeList get_ranges(Range2Du32 const& full_range, Vec2Di32 const& direction)
{
	RangeList list{};

	auto no_horizontal = direction.x == 0;
	auto no_vertical = direction.y == 0;
	

	if(no_horizontal && no_vertical)
	{
		list.write_h = full_range;
		return list;
	}

	auto copy_right = direction.x > 0;
	auto copy_left = direction.x < 0;
	auto copy_down = direction.y > 0;
	auto copy_up = direction.y < 0;

	auto write_left = direction.x > 0;
	auto write_top = direction.y > 0;

	auto const n_cols = (u32)(std::abs(direction.x));
	auto const n_rows = (u32)(std::abs(direction.y));

	list.copy_src = full_range;
	list.copy_dst = full_range;

	if(copy_left)
	{
		list.copy_src.x_begin = n_cols;
		list.copy_dst.x_end -= n_cols;
	}
	else if(copy_right)
	{
		list.copy_dst.x_begin = n_cols;
		list.copy_src.x_end -= n_cols;
	}

	if(copy_up)
	{
		list.copy_src.y_begin = n_rows;
		list.copy_dst.y_end -= n_rows;
	}
	else if (copy_down)
	{
		list.copy_dst.y_begin = n_rows;
		list.copy_src.y_end -= n_rows;
	}

	list.write_h = full_range;


	if (no_horizontal && no_vertical)
	{

	}
	else if (no_horizontal)
	{
		if (write_top)
		{
			list.write_h.y_end = n_rows;
		}
		else // if (write_bottom)
		{
			list.write_h.y_begin = list.write_h.y_end - n_rows - 1;
		}
	}
	else if (no_vertical)
	{
		if (write_left)
		{
			list.write_h.x_end = n_cols;
		}
		else // if (wright_right)
		{
			list.write_h.x_begin = list.write_h.x_end - n_cols - 1;
		}
	}
    else
    {
		list.write_v = list.write_h;
        
        if (write_top)
        {
            list.write_h.y_end = n_rows;
            list.write_v.y_begin = n_rows;
        }
        else // if (write_bottom)
        {
            list.write_h.y_begin = full_range.y_end - n_rows - 1;
            list.write_v.y_end = full_range.y_end - n_rows - 1;
        }

        if (write_left)
        {
            list.write_v.x_end = n_cols;
        }
        else // if (write_right)
        {
            list.write_v.x_begin = full_range.x_end - n_cols - 1;
        }
    }    

	return list;
}


static Range2Du32 get_full_range(Mat2Di32 const& mat)
{
	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = mat.width;
	r.y_begin = 0;
	r.y_end = mat.height;

	return r;
}











class MbtProps
{
public:
	u32 iter_limit;
	r64 min_mx;
	r64 min_my;
	r64 mx_step;
	r64 my_step;

    Range2Du32 copy_src;
    Range2Du32 copy_dst;

    Mat2Di32 old_ids;
    
    DrawProps draw_props;
};





namespace gpu
{
/**********/


GPU_FUNCTION
static Pixel to_pixel(u8 r, u8 g, u8 b)
{
    Pixel p{};
    p.red = r;
    p.green = g;
    p.blue = b;
    p.alpha = 255;

    return p;
}


GPU_FUNCTION
static bool in_range(u32 x, u32 y, Range2Du32 const& range)
{
    return 
        range.x_begin <= x &&
        x > range.x_end &&
        range.y_begin <= y &&
        y < range.y_end;
}


GPU_FUNCTION
static void copy_xy(Mat2Di32 const& src, Mat2Di32 const& dst, Range2Du32 const& r_src, Range2Du32 const & r_dst, u32 src_x, u32 src_y)
{
    auto x_offset = src_x - r_src.x_begin;
    auto y_offset = src_y - r_src.y_begin;

    auto dst_x = r_dst.x_begin + x_offset;
    auto dst_y = r_dst.y_begin + y_offset;

    auto src_i = src_y * src.width + src_x;
    auto dst_i = dst_y * src.width + dst_x;    

    dst.data[dst_i] = src.data[src_i];
}


GPU_FUNCTION
static u32 mandelbrot_iter(r64 cx, r64 cy, u32 iter_limit)
{
	u32 iter = 0;

    r64 mx = 0.0;
    r64 my = 0.0;
    r64 mx2 = 0.0;
    r64 my2 = 0.0;

    while (iter < iter_limit && mx2 + my2 <= 4.0)
    {
		++iter;

        my = (mx + mx) * my + cy;
        mx = mx2 - my2 + cx;
        my2 = my * my;
        mx2 = mx * mx;
    }

	return iter;
}


GPU_FUNCTION
static i32 color_index(u32 iter, u32 iter_limit)
{
	if (iter >= iter_limit)
	{
		return -1;
	}

	constexpr std::array<u32, 6> iter_levels = { 50, 300, 600, 1000, 1500, 2500 };

	u32 min = 0;
	u32 max = 0;
	u32 n_colors = 8;

	for (auto i : iter_levels)
	{
		n_colors *= 2;
		min = max;
		max = i;

		if (iter < max)
		{
			return (iter - min) % n_colors * (N_COLORS / n_colors);
		}
	}	

	min = max;
	
	return (iter - min) % N_COLORS;
}


GPU_FUNCTION
static void mandelbrot_xy(Mat2Di32 const& dst, MbtProps const& props, u32 x, u32 y)
{
    r64 cy = props.min_my + y * props.my_step;
    r64 cx = props.min_mx + x * props.mx_step;

    auto iter = gpu::mandelbrot_iter(cx, cy, props.iter_limit);
    auto index = color_index(iter, props.iter_limit);

    auto dst_i = y * dst.width + x;
    dst.data[dst_i] = index;
}


GPU_FUNCTION
static void draw_pixel(DrawProps const& props, u32 pixel_index)
{
    auto& src = props.current_ids;
    auto& dst = props.screen_dst;

    auto color_id = src.data[pixel_index];

    if(color_id < 0)
    {
        dst.data[pixel_index] = gpu::to_pixel(0, 0, 0); // TODO: platform pixel
    }
    else
    {
        u8 color_map[] = { palettes[0][color_id], palettes[1][color_id], palettes[2][color_id] };
        dst.data[pixel_index] = to_pixel(color_map[props.channel1], color_map[props.channel2], color_map[props.channel3]);
    }
}



/**********/
}


GPU_KERNAL
static void gpu_process_and_draw(MbtProps const& props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& current_ids = props.draw_props.current_ids;
    auto& old_ids = props.old_ids;

    auto const width = old_ids.width;
    auto const height = old_ids.height;

    assert(n_threads == width * height);

    auto pixel_id = (u32)t;

    auto y = pixel_id / width;
    auto x = pixel_id - y * width;

    if(gpu::in_range(x, y, props.copy_src))
    {
        gpu::copy_xy(old_ids, current_ids, props.copy_src, props.copy_dst, x, y);
    }
    else
    {
        gpu::mandelbrot_xy(current_ids, props, x, y);
    }

    gpu::draw_pixel(props.draw_props, pixel_id);
}


GPU_KERNAL
static void gpu_draw(DrawProps const& props, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto const width = props.current_ids.width;
    auto const height = props.current_ids.height;

    assert(n_threads == width * height);

    auto pixel_id = (u32)t;

    gpu::draw_pixel(props, pixel_id);
}


void render(AppState& state)
{
    auto width = state.unified.screen_buffer.width;
    auto height = state.unified.screen_buffer.height;
    auto n_threads = width * height;
    auto n_blocks = calc_thread_blocks(n_threads);

    if(!state.render_new && !state.draw_new)
    {
        return;
    }

    state.ids_current = state.ids_old;
    state.ids_old = !state.ids_old;

    auto& current_ids = state.device.color_ids[state.ids_current];
    auto& old_ids = state.device.color_ids[state.ids_old];

    DrawProps draw_props{};
    draw_props.current_ids = current_ids;
    draw_props.screen_dst = state.unified.screen_buffer;

    set_rgb_channels(draw_props, state.rgb_option);

    bool result = cuda::no_errors("render");
    assert(result);

    if(state.render_new)
    {	
		auto ranges = get_ranges(get_full_range(current_ids), state.pixel_shift);        

        MbtProps props{};
		props.iter_limit = state.iter_limit;
		props.min_mx = MBT_MIN_X + state.mbt_pos.x;
		props.min_my = MBT_MIN_Y + state.mbt_pos.y;
		props.mx_step = state.mbt_screen_width / width;
		props.my_step = state.mbt_screen_height / height;

        props.copy_src = ranges.copy_src;
        props.copy_dst = ranges.copy_dst;

        props.old_ids = old_ids;
        
        props.draw_props = draw_props;        

        cuda_launch_kernel(gpu_process_and_draw, n_blocks, THREADS_PER_BLOCK, props, n_threads);

        result = cuda::launch_success("gpu_process_and_draw");
        assert(result);
    }    
    else if(state.draw_new)
    {
        cuda_launch_kernel(gpu_draw, n_blocks, THREADS_PER_BLOCK, draw_props, n_threads);

        result = cuda::launch_success("gpu_draw");
        assert(result);
    }
}