#include "render.hpp"
#include "render_include.hpp"
#include "colors.hpp"
#include "../utils/index_range.hpp"
#include "range_list.hpp"

#include <cassert>
#include <algorithm>
#include <functional>
#include <cmath>

#ifdef NO_CPP17

static void for_each(UnsignedRange const& ids, std::function<void(u32)> const& func)
{
	std::for_each(ids.begin(), ids.end(), func);
}

#else

#include <execution>

static void for_each(UnsignedRange const& ids, std::function<void(u32)> const& func)
{
	std::for_each(std::execution::par, ids.begin(), ids.end(), func);
}


#endif // NO_CPP17


static Pixel to_pixel(u8 r, u8 g, u8 b)
{
    Pixel p{};
    p.red = r;
    p.green = g;
    p.blue = b;
    p.alpha = 255;

    return p;
}


static bool in_range(u32 x, u32 y, Range2Du32 const& range)
{
    return 
        range.x_begin <= x &&
        x < range.x_end &&
        range.y_begin <= y &&
        y < range.y_end;
}


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


static void draw_xy(Mat2Di32 const& src, Image const& dst, ChannelOptions const& options, u32 x, u32 y)
{
	auto pixel_index = y * dst.width + x;
	auto color_id = src.data[pixel_index];

	if(color_id < 0)
    {
        dst.data[pixel_index] = to_pixel(0, 0, 0); // TODO: platform pixel
    }
    else
    {
        u8 color_map[] = { palettes[0][color_id], palettes[1][color_id], palettes[2][color_id] };
        dst.data[pixel_index] = to_pixel(color_map[options.channel1], color_map[options.channel2], color_map[options.channel3]);
    }
}


static void process_and_draw(AppState const& state)
{
	auto& ids = state.color_ids[state.current_id];
	auto& prev_ids = state.color_ids[state.prev_id];
	auto& pixels = state.screen_buffer;

	auto const mbt_row = [&](u32 y) 
	{
		auto ids_row = ids.row_begin(y);
		r64 cy = std::fma((r64)y, state.my_step, state.min_my); // state.min_my + y * state.my_step;

		for(u32 x = 0; x < ids.width; ++x)
		{
			if(in_range(x, y, state.copy_dst))
			{
				copy_xy(prev_ids, ids, state.copy_src, state.copy_dst, x, y);				
			}
			else
			{
				r64 cx = std::fma((r64)x, state.mx_step, state.min_mx); // state.min_mx + x * state.mx_step;
				auto iter = mandelbrot_iter(cx, cy, state.app_input.iter_limit);
				ids_row[x] = color_index(iter, state.app_input.iter_limit);
			}			

			draw_xy(ids, pixels, state.channel_options, x, y);
		}
	};

	UnsignedRange rows(0u, ids.height);
	for_each(rows, mbt_row);
}


static void draw(AppState const& state)
{
	auto& ids = state.color_ids[state.current_id];
	auto& pixels = state.screen_buffer;

	auto const draw_row = [&](u32 y)
	{
		for(u32 x = 0; x < pixels.width; ++x)
		{
			draw_xy(ids, pixels, state.channel_options, x, y);
		}
	};

	UnsignedRange rows(0u, pixels.height);
	for_each(rows, draw_row);
}





void render(AppState& state)
{
	auto width = state.screen_buffer.width;
	auto height = state.screen_buffer.height;

	if(!state.app_input.render_new && !state.app_input.draw_new)
    {
        return;
    }

    set_rgb_channels(state.channel_options, state.app_input.rgb_option);

	if(state.app_input.render_new)
	{
		state.current_id = state.prev_id;
		state.prev_id = !state.prev_id;
		
		auto ranges = get_ranges(make_range(width, height), state.app_input.pixel_shift);
		
		state.min_mx = MBT_MIN_X + state.app_input.mbt_pos.x;
		state.min_my = MBT_MIN_Y + state.app_input.mbt_pos.y;
		state.mx_step = state.app_input.mbt_screen_width / width;
		state.my_step = state.app_input.mbt_screen_height / height;

		state.copy_src = ranges.copy_src;
        state.copy_dst = ranges.copy_dst;

		process_and_draw(state);

		state.app_input.draw_new = false;
	}
    
    if(state.app_input.draw_new)
    {       
		draw(state);
    }
}


u32 get_rgb_combo_qty()
{
	constexpr auto n = num_rgb_combinations();

	return n;
}