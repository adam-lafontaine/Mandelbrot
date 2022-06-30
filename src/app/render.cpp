#include "render.hpp"
#include "colors.hpp"
#include "../utils/index_range.hpp"

#include <cassert>
#include <algorithm>
#include <functional>

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


static void transform(Mat2Di32 const& src, Image const& dst, std::function<Pixel(i32)> const& func)
{
	std::transform(src.begin(), src.end(), dst.begin(), func);
}


class RangeList
{
public:
	Range2Du32 copy_src;
	Range2Du32 copy_dst;
	Range2Du32 write_h;
	Range2Du32 write_v;
};


class MbtProps
{
public:
	u32 iter_limit;
	r64 min_mx;
	r64 min_my;
	r64 mx_step;
	r64 my_step;
};


// platform dependent e.g. win32_main.cpp
u32 platform_to_color_32(u8 red, u8 green, u8 blue);


static pixel_t to_platform_pixel(u8 red, u8 green, u8 blue)
{
	pixel_t p = {};
	p.value = platform_to_color_32(red, green, blue);

	return p;
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


static void set_rgb_channels(u32& c1, u32& c2, u32& c3, u32 rgb_option)
{
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


static void draw(AppState const& state)
{
	auto& src = state.color_ids[state.ids_current];
	auto& dst = state.screen_buffer;

	u32 c1 = 0;
	u32 c2 = 0;
	u32 c3 = 0;
	set_rgb_channels(c1, c2, c3, state.rgb_option);

	auto const to_color = [&](i32 i)
	{
		if(i < 0)
		{
			return to_platform_pixel(0, 0, 0);
		}

		u8 color_map[] = { palettes[0][i], palettes[1][i], palettes[2][i] };
		return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
	};

	transform(src, dst, to_color);
}


static void copy(Mat2Di32 const& src, Mat2Di32 const& dst, Range2Du32 const& src_r, Range2Du32 const& dst_r)
{
	auto copy_width = src_r.x_end - src_r.x_begin;
	auto copy_height = src_r.y_end - src_r.y_begin;

	assert(dst_r.x_end - dst_r.x_begin == copy_width);
	assert(dst_r.y_end - dst_r.y_begin == copy_height);

	auto const copy_row = [&](u32 row) 
	{
		auto src_row = src.row_begin(src_r.y_begin + row);
		auto dst_row = dst.row_begin(dst_r.y_begin + row);

		for (u32 w = 0; w < copy_width; ++w)
		{
			dst_row[dst_r.x_begin + w] = src_row[src_r.x_begin + w];
		}
	};

	UnsignedRange rows(0u, copy_height);
	for_each(rows, copy_row);
}


static void mandelbrot(Mat2Di32 const& dst, Range2Du32 const& range, MbtProps const& props)
{
	auto const mbt_row = [&](u32 y) 
	{
		auto dst_row = dst.row_begin(y);
		r64 cy = props.min_my + y * props.my_step;

		for (u32 x = range.x_begin; x < range.x_end; ++x)
		{
			r64 cx = props.min_mx + x * props.mx_step;
			auto iter = mandelbrot_iter(cx, cy, props.iter_limit);
			auto index = color_index(iter, props.iter_limit);
			dst_row[x] = index;
		}
	};

	UnsignedRange rows(range.y_begin, range.y_end);
	for_each(rows, mbt_row);
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


void render(AppState& state)
{
	if(state.render_new)
	{
		state.ids_current = state.ids_old;
		state.ids_old = !state.ids_old;

		auto& current_ids = state.color_ids[state.ids_current];
		auto& old_ids = state.color_ids[state.ids_old];

		auto& ids = state.color_ids;
		auto ranges = get_ranges(get_full_range(current_ids), state.pixel_shift);

		copy(old_ids, current_ids, ranges.copy_src, ranges.copy_dst);

		MbtProps props{};
		props.iter_limit = state.iter_limit;
		props.min_mx = MBT_MIN_X + state.mbt_pos.x;
		props.min_my = MBT_MIN_Y + state.mbt_pos.y;
		props.mx_step = state.mbt_screen_width / old_ids.width;
		props.my_step = state.mbt_screen_height / old_ids.height;

		mandelbrot(current_ids, ranges.write_h, props);
		mandelbrot(current_ids, ranges.write_v, props);

		state.draw_new = true;
	}
    
    if(state.draw_new)
    {       
		draw(state);
    }
}


u32 get_rgb_combo_qty()
{
	constexpr auto n = num_rgb_combinations();

	return n;
}