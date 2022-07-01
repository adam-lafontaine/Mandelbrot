#include "render.hpp"
#include "cuda_def.cuh"

#include <cassert>

constexpr int THREADS_PER_BLOCK = 1024;

constexpr int calc_thread_blocks(u32 n_threads)
{
    return (n_threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}


static void set_rgb_channels(ChannelOptions& options, u32 rgb_option)
{
	auto& c1 = options.channel1;
    auto& c2 = options.channel2;
    auto& c3 = options.channel3;

	switch (rgb_option)
	{
	case 1:
		c1 = 0;
		c2 = 1;
		c3 = 2;
		break;
	case 2:
		c1 = 0;
		c2 = 2;
		c3 = 1;
		break;
	case 3:
		c1 = 1;
		c2 = 0;
		c3 = 2;
		break;
	case 4:
		c1 = 1;
		c2 = 2;
		c3 = 0;
		break;
	case 5:
		c1 = 2;
		c2 = 0;
		c3 = 1;
		break;
	case 6:
		c1 = 2;
		c2 = 1;
		c3 = 0;
		break;
	}
}


class RangeList
{
public:
	Range2Du32 copy_src;
	Range2Du32 copy_dst;
};


RangeList get_ranges(Range2Du32 const& full_range, Vec2Di32 const& direction)
{
	RangeList list{};

	auto no_horizontal = direction.x == 0;
	auto no_vertical = direction.y == 0;	

	if(no_horizontal && no_vertical)
	{
		return list;
	}

	auto copy_right = direction.x > 0;
	auto copy_left = direction.x < 0;
	auto copy_down = direction.y > 0;
	auto copy_up = direction.y < 0;

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

	return list;
}


static Range2Du32 get_full_range(Image image)
{
	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = image.width;
	r.y_begin = 0;
	r.y_end = image.height;

	return r;
}


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
        x < range.x_end &&
        range.y_begin <= y &&
        y < range.y_end;
}


GPU_FUNCTION
static void copy_xy(Mat2Di32 const& src, Mat2Di32 const& dst, Range2Du32 const& r_src, Range2Du32 const & r_dst, u32 dst_x, u32 dst_y)
{
    auto x_offset = dst_x - r_dst.x_begin;
    auto y_offset = dst_y - r_dst.y_begin;

    auto src_x = r_src.x_begin + x_offset;
    auto src_y = r_src.y_begin + y_offset;

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
static i32 color_index(u32 iter, u32 iter_limit, u32 total_colors)
{
	if (iter >= iter_limit)
	{
		return -1;
	}

    u32 const n_iter_levels = 6;

	u32 iter_levels[] = { 50, 300, 600, 1000, 1500, 2500 };

	u32 min = 0;
	u32 max = 0;
	u32 n_colors = 8;

    for(u32 i = 0; i < n_iter_levels; ++i)
    {
        n_colors *= 2;
		min = max;
		max = iter_levels[i];

		if (iter < max)
		{
			return (iter - min) % n_colors * (total_colors / n_colors);
		}
    }

	min = max;
	
	return (iter - min) % total_colors;
}


GPU_FUNCTION
static void mandelbrot_xy(DeviceMemory const& device, UnifiedMemory const& unified, u32 x, u32 y)
{
    auto& dst = device.color_ids[unified.ids_current];

    r64 cy = unified.min_my + y * unified.my_step;
    r64 cx = unified.min_mx + x * unified.mx_step;

    auto iter = gpu::mandelbrot_iter(cx, cy, unified.iter_limit);
    auto index = gpu::color_index(iter, unified.iter_limit, device.color_palette.n_colors);

    auto dst_i = y * dst.width + x;
    dst.data[dst_i] = index;
}


GPU_FUNCTION
static void draw_pixel(DeviceMemory const& device, UnifiedMemory const& unified, u32 pixel_index)
{
    auto& src = device.color_ids[unified.ids_current];
    auto& dst = unified.screen_buffer;
    auto& options = unified.channel_options;

    auto color_id = src.data[pixel_index];

    if(color_id < 0)
    {
        dst.data[pixel_index] = gpu::to_pixel(0, 0, 0); // TODO: platform pixel
    }
    else
    {
        auto& colors = device.color_palette;
        u8 color_map[] = { colors.channel1[color_id], colors.channel2[color_id], colors.channel3[color_id] };
        dst.data[pixel_index] = to_pixel(color_map[options.channel1], color_map[options.channel2], color_map[options.channel3]);
    }
}



/**********/
}



GPU_KERNAL
static void gpu_process_and_draw(DeviceMemory* device, UnifiedMemory* unified, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& D = *device;
    auto& U = *unified;

    assert(n_threads == U.screen_buffer.width * U.screen_buffer.height);

    auto pixel_id = (u32)t;
    auto const width = U.screen_buffer.width;

    auto y = pixel_id / width;
    auto x = pixel_id - y * width;

    if(gpu::in_range(x, y, U.copy_dst))
    {
        auto& current_ids = D.color_ids[U.ids_current];
        auto& prev_ids = D.color_ids[U.ids_prev];

        gpu::copy_xy(prev_ids, current_ids, U.copy_src, U.copy_dst, x, y);
    }
    else
    {
        gpu::mandelbrot_xy(D, U, x, y);
    }


    gpu::draw_pixel(D, U, pixel_id);
}


GPU_KERNAL
static void gpu_draw(DeviceMemory* device, UnifiedMemory* unified, u32 n_threads)
{
    int t = blockDim.x * blockIdx.x + threadIdx.x;
    if (t >= n_threads)
    {
        return;
    }

    auto& D = *device;
    auto& U = *unified;

    assert(n_threads == U.screen_buffer.width * U.screen_buffer.height);

    auto pixel_id = (u32)t;

    gpu::draw_pixel(D, U, pixel_id);
}


void render(AppState& state)
{
    auto device_data = state.device.data;
    auto unified_data = state.unified.data;

    auto& unified = *unified_data;

    auto width = unified.screen_buffer.width;
    auto height = unified.screen_buffer.height;
    auto n_threads = width * height;
    auto n_blocks = calc_thread_blocks(n_threads);

    if(!state.render_new && !state.draw_new)
    {
        return;
    }

    if(state.render_new)
    {
        unified.ids_current = unified.ids_prev;
        unified.ids_prev = !unified.ids_prev;
    } 

    set_rgb_channels(unified.channel_options, state.rgb_option);

    bool result = cuda::no_errors("render");
    assert(result);

    if(state.render_new)
    {	
		auto ranges = get_ranges(get_full_range(unified.screen_buffer), state.pixel_shift);   
        
		unified.iter_limit = state.iter_limit;
		unified.min_mx = MBT_MIN_X + state.mbt_pos.x;
		unified.min_my = MBT_MIN_Y + state.mbt_pos.y;
		unified.mx_step = state.mbt_screen_width / width;
		unified.my_step = state.mbt_screen_height / height;

        unified.copy_src = ranges.copy_src;
        unified.copy_dst = ranges.copy_dst;

        cuda_launch_kernel(gpu_process_and_draw, n_blocks, THREADS_PER_BLOCK, device_data, unified_data, n_threads);
        result = cuda::launch_success("gpu_process_and_draw");
        assert(result);

        cuda_launch_kernel(gpu_draw, n_blocks, THREADS_PER_BLOCK, device_data, unified_data, n_threads);

        result = cuda::launch_success("gpu_draw 1");
        assert(result);

        state.draw_new = false;
    }    
    
    if(state.draw_new)
    {
        cuda_launch_kernel(gpu_draw, n_blocks, THREADS_PER_BLOCK, device_data, unified_data, n_threads);

        result = cuda::launch_success("gpu_draw");
        assert(result);
    }
}