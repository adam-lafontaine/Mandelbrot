#include "render.hpp"
#include "colors.hpp"
#include "../utils/index_range.hpp"

#include <cassert>
#include <algorithm>
#include <functional>


class MandelbrotProps
{
public:
    u32 iter_limit;
    r64 min_mx;
    r64 min_my;
    r64 mx_step;
    r64 my_step;

    u32* iterations_dst;
    u32 width;

	i16* color_indeces_dst;

    Range2Du32 range1;
    Range2Du32 range2;
};


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


static Point2Du32 get_position(Range2Du32 const& range1, Range2Du32 const& range2, u32 width, u32 r_id)
{
    auto w1 = range1.x_end - range1.x_begin;
    auto h1 = range1.y_end - range1.y_begin;

    if(r_id < w1 * h1)
    {
        return get_position(range1, width, r_id);
    }
    
    return get_position(range2, width, r_id - w1 * h1);
}


static u32 get_index(u32 x, u32 y, u32 width)
{
    return width * y + x;
}


static u32 get_index(Point2Du32 const& pos, u32 width)
{
    return get_index(pos.x, pos.y, width);
}


static u32 mandelbrot_iterations(r64 cx, r64 cy, u32 iter_limit)
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


static i16 mandelbrot_color_index(r64 cx, r64 cy, u32 iter_limit)
{
	u32 iter = 0;
	u8 index = 0; // 256 colors

    r64 mx = 0.0;
    r64 my = 0.0;
    r64 mx2 = 0.0;
    r64 my2 = 0.0;

    while (iter < iter_limit && mx2 + my2 <= 4.0)
    {
		++iter;
		++index;

        my = (mx + mx) * my + cy;
        mx = mx2 - my2 + cx;
        my2 = my * my;
        mx2 = mx * mx;
    }

	return iter == iter_limit ? -1 : index;
}


#ifdef NO_CPP_17


static void transform(u32* src_begin, u32* src_end, pixel_t* dst_begin, std::function<pixel_t(u32)> const& f)
{
	std::transform(src_begin, src_end, dst_begin, f);
}


static void copy(u32* src_begin, u32* src_end, u32* dst_begin)
{
	std::copy(src_begin, src_end, dst_begin);
}


static void mandelbrot(MandelbrotProps const& props, Range2Du32 const& range)
{
	auto const width = props.width;

	for(u32 y = range.y_begin; y < range.y_end; ++y)
	{		
		auto row_offset = width * y;
		r64 cy = props.min_my + y * props.my_step;

		for(u32 x = range.x_begin; x < range.x_end; ++x)
		{
			r64 cx = props.min_mx + x * props.mx_step;
			auto iter = mandelbrot_iterations(cx, cy, props.iter_limit);
			props.iterations_dst[row_offset + x] = iter;

			//auto index = mandelbrot_color_index(cx, cy, props.iter_limit);
			//props.color_indeces_dst[row_offset + x] = index;
		}
	}
}


static void mandelbrot(MandelbrotProps const& props)
{
	mandelbrot(props, props.range1);
	mandelbrot(props, props.range2);	
}

#else

#include <execution>


static void transform(u32* src_begin, u32* src_end, pixel_t* dst_begin, std::function<pixel_t(u32)> const& f)
{
	std::transform(std::execution::par, src_begin, src_end, dst_begin, f);
}


static void copy(u32* src_begin, u32* src_end, u32* dst_begin)
{
	std::copy(std::execution::par, src_begin, src_end, dst_begin);
}


static void mandelbrot_by_xy(MandelbrotProps const& props, Point2Du32 pos)
{
	auto const width = props.width;
    
    r64 const cx = props.min_mx + pos.x * props.mx_step;
    r64 const cy = props.min_my + pos.y * props.my_step;

    auto iter = mandelbrot_iterations(cx, cy, props.iter_limit);

	auto i = get_index(pos, width);

    props.iterations_dst[i] = iter;
}


static void mandelbrot(MandelbrotProps const& props)
{
	u32 n_elements = (props.range1.x_end - props.range1.x_begin) * (props.range1.y_end - props.range1.y_begin) +
        (props.range2.x_end - props.range2.x_begin) * (props.range2.y_end - props.range2.y_begin);

	auto r_ids = UnsignedRange(0u, n_elements);

	auto const do_mandelbrot = [&](u32 r_id)
	{
		auto const width = props.width;
		auto pos = get_position(props.range1, props.range2, width, r_id);
		
		mandelbrot_by_xy(props, pos);
	};
	
	std::for_each(std::execution::par, r_ids.begin(), r_ids.end(), do_mandelbrot);
}

#endif



// platform dependent e.g. win32_main.cpp
u32 platform_to_color_32(u8 red, u8 green, u8 blue);


static Range2Du32 get_range(Mat2Du32 const& mat)
{
	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = mat.width;
	r.y_begin = 0;
	r.y_end = mat.height;

	return r;
}


static pixel_t to_platform_pixel(u8 red, u8 green, u8 blue)
{
	pixel_t p = {};
	p.value = platform_to_color_32(red, green, blue);

	return p;
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


static pixel_t to_rgb_16(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 16;
	u8 color_map[] = { palettes16[0][i], palettes16[1][i], palettes16[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_32(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 32;
	u8 color_map[] = { palettes32[0][i], palettes32[1][i], palettes32[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_48(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 48;
	u8 color_map[] = { palettes48[0][i], palettes48[1][i], palettes48[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_64(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 64;
	u8 color_map[] = { palettes64[0][i], palettes64[1][i], palettes64[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_80(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 80;
	u8 color_map[] = { palettes80[0][i], palettes80[1][i], palettes80[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_96(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 96;
	u8 color_map[] = { palettes96[0][i], palettes96[1][i], palettes96[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_112(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 112;
	u8 color_map[] = { palettes112[0][i], palettes112[1][i], palettes112[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_128(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 128;
	u8 color_map[] = { palettes128[0][i], palettes128[1][i], palettes128[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_144(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 144;
	u8 color_map[] = { palettes144[0][i], palettes144[1][i], palettes144[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_160(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 160;
	u8 color_map[] = { palettes160[0][i], palettes160[1][i], palettes160[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_176(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 176;
	u8 color_map[] = { palettes176[0][i], palettes176[1][i], palettes176[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_192(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 192;
	u8 color_map[] = { palettes192[0][i], palettes192[1][i], palettes192[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_208(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 208;
	u8 color_map[] = { palettes208[0][i], palettes208[1][i], palettes208[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_224(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 224;
	u8 color_map[] = { palettes224[0][i], palettes224[1][i], palettes224[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_240(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 240;
	u8 color_map[] = { palettes240[0][i], palettes240[1][i], palettes240[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_256(u32 iterations, u32 c1, u32 c2, u32 c3)
{
	auto i = iterations % 256;
	u8 color_map[] = { palettes256[0][i], palettes256[1][i], palettes256[2][i] };
	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


std::function<pixel_t(u32, u32, u32, u32)> get_rgb_function(u32 max_iter)
{
	/*
	switch (max_iter / 128)
	{
	case 0:
		return to_rgb_16;
	case 1:
		return to_rgb_32;
	case 2:
		return to_rgb_48;
	case 3:
		return to_rgb_64;
	case 4:
		return to_rgb_80;
	case 5:
		return to_rgb_96;
	case 6:
		return to_rgb_112;
	case 7:
		return to_rgb_128;
	case 8:
		return to_rgb_144;
	case 9:
		return to_rgb_160;
	case 10:
		return to_rgb_176;
	case 11:
		return to_rgb_192;
	case 12:
		return to_rgb_208;
	case 13:
		return to_rgb_224;
	case 14:
		return to_rgb_240;
	default:
		return to_rgb_256;
	}
	*/

	return to_rgb_32;
}


static void for_each_row(Mat2Du32 const& mat, std::function<void(u32 y)> const& func)
{
	UnsignedRange y_ids(0u, mat.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	std::for_each(y_id_begin, y_id_end, func);
}


static void copy_left(Mat2Du32 const& mat, u32 n_cols)
{
	u32 const x_len = mat.width - n_cols;

	auto const copy_row_part = [&](u32 y)
	{
		auto row = mat.row_begin(y);
		for (u32 ix = 0; ix < x_len; ++ix)
		{
			auto src_x = ix + n_cols;
			auto dst_x = src_x - n_cols;
			row[dst_x] = row[src_x];
		}
	};

	for_each_row(mat, copy_row_part);
}


static void copy_right(Mat2Du32 const& mat, u32 n_cols)
{
	u32 const x_len = mat.width - n_cols;

	auto const copy_row_part = [&](u32 y)
	{
		auto row = mat.row_begin(y);
		for (u32 ix = 0; ix < x_len; ++ix)
		{
			auto src_x = x_len - 1 - ix;
			auto dst_x = src_x + n_cols;
			row[dst_x] = row[src_x];
		}
	};

	for_each_row(mat, copy_row_part);
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



static void copy(Mat2Du32 const& mat, Vec2Di32 const& direction)
{
	auto const n_cols = (u32)(std::abs(direction.x));
	auto const n_rows = (u32)(std::abs(direction.y));

	if (n_cols == 0 && n_rows == 0)
	{
		return;
	}

	auto right = direction.x > 0;

	if (n_rows == 0)
	{
		if (right)
		{
			copy_right(mat, n_cols);
		}
		else
		{
			copy_left(mat, n_cols);
		}

		return;
	}

	auto up = direction.y < 0;

	u32 const x_len = mat.width - n_cols;
	u32 const y_len = mat.height - n_rows;

	u32 src_x_begin = right ? 0 : n_cols;
	u32 dst_x_begin = src_x_begin + direction.x;

	u32 src_y_begin = 0;
	i32 dy = 0;
	if (up)
	{
		src_y_begin = n_rows;
		dy = 1;
	}
	else
	{
		src_y_begin = y_len - 1;
		dy = -1;
	}

	for (u32 iy = 0; iy < y_len; ++iy)
	{
		auto src_y = src_y_begin + dy * iy;
		auto dst_y = src_y + direction.y;

		auto src_begin = mat.row_begin(src_y) + src_x_begin;
		auto dst_begin = mat.row_begin(dst_y) + dst_x_begin;

		auto src_end = src_begin + x_len;

		copy(src_begin, src_end, dst_begin);
	}
}


static void mandelbrot(Mat2Du32 const& dst, AppState const& state)
{
	auto const width = dst.width;
	auto const height = dst.height;

	auto& shift = state.pixel_shift;

	auto do_left = shift.x > 0;
	auto do_top = shift.y > 0;

    auto const n_cols = (u32)(std::abs(shift.x));
	auto const n_rows = (u32)(std::abs(shift.y));

	auto no_horizontal = n_cols == 0;
	auto no_vertical = n_rows == 0;

    Range2Du32 range1{};
    range1.x_begin = 0;
    range1.x_end = width;
    range1.y_begin = 0;
    range1.y_end = height;
    
    Range2Du32 range2{};
    range2.x_begin = 0;
    range2.x_end = 0;
    range2.y_begin = 0;
    range2.y_end = 0;

    if (no_horizontal && no_vertical)
	{

	}
	else if (no_horizontal)
	{
		if (do_top)
		{
			range1.y_end = n_rows;
		}
		else // if (do_bottom)
		{
			range1.y_begin = height - 1 - n_rows;
		}
	}
	else if (no_vertical)
	{
		if (do_left)
		{
			range1.x_end = n_cols;
		}
		else // if (do_right)
		{
			range1.x_begin = width - 1 - n_cols;
		}
	}
    else
    {
        range2 = range1;
        
        if (do_top)
        {
            range1.y_end = n_rows;
            range2.y_begin = n_rows;
        }
        else // if (do_bottom)
        {
            range1.y_begin = height - 1 - n_rows;
            range2.y_end = height - 1 - n_rows;
        }

        if (do_left)
        {
            range2.x_end = n_cols;
        }
        else // if (do_right)
        {
            range2.x_begin = width - 1 - n_cols;
        }
    }    

	MandelbrotProps props{};
	props.iter_limit = state.iter_limit;
	props.min_mx = MBT_MIN_X + state.mbt_pos.x;
	props.min_my = MBT_MIN_Y + state.mbt_pos.y;
	props.mx_step = state.mbt_screen_width / width;
	props.my_step = state.mbt_screen_height / height;
    props.width = width;
    props.iterations_dst = dst.data;
	//props.color_indeces_dst = state.color_indeces.data;
    props.range1 = range1;
    props.range2 = range2;

	mandelbrot(props);
}


static void find_min_max_iter(AppState& state)
{
	auto& mat = state.iterations;
	auto [mat_min, mat_max] = std::minmax_element(mat.begin(), mat.end());
	state.iter_min = *mat_min;
	state.iter_max = *mat_max;
}


static void draw(Image const& dst, AppState const& state)
{
	auto const min = state.iter_min;
	auto const max = state.iter_max;	

	auto diff = max - min;

	u32 c1 = 0;
	u32 c2 = 0;
	u32 c3 = 0;
	set_rgb_channels(c1, c2, c3, state.rgb_option);

	auto const to_rgb = get_rgb_function(diff);

	auto const to_color = [&](u32 i)
	{
		if (i >= max)
		{
			return to_platform_pixel(0, 0, 0);
		}

		return to_rgb(i - min, c1, c2, c3);
	};

	auto& mat = state.iterations;

	transform(mat.begin(), mat.end(), dst.begin(), to_color);
}


static void draw(Mat2Di16 const& src, Image const& dst, u32 rgb_option)
{
	u32 c1 = 0;
	u32 c2 = 0;
	u32 c3 = 0;
	set_rgb_channels(c1, c2, c3, rgb_option);

	auto& palettes = palettes32;

	auto const to_color = [&](i16 i)
	{
		if(i < 0)
		{
			return to_platform_pixel(0, 0, 0);
		}

		u8 color_map[] = { palettes[0][i], palettes[1][i], palettes[2][i] };
		return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
	};

	std::transform(src.begin(), src.end(), dst.begin(), to_color);
}





static void copy(Mat2Di16 const& src, Mat2Di16 const& dst, Range2Du32 const& src_r, Range2Du32 const& dst_r)
{
	auto copy_width = src_r.x_end - src_r.x_begin;
	auto copy_height = src_r.y_end - src_r.y_begin;

	assert(dst_r.x_end - dst_r.x_begin == copy_width);
	assert(dst_r.y_end - dst_r.y_begin == copy_height);

	for(u32 h = 0; h < copy_height; ++h)
	{
		auto src_row = src.row_begin(src_r.y_begin + h);
		auto dst_row = dst.row_begin(dst_r.y_begin + h);

		for(u32 w = 0; w < copy_width; ++w)
		{
			dst_row[dst_r.x_begin + w] = src_row[src_r.x_begin + w];
		}
	}
}


class MbtProps
{
public:
    u32 iter_limit;
    r64 min_mx;
    r64 min_my;
    r64 mx_step;
    r64 my_step;
};


static void mandelbrot(Mat2Di16 const& dst, Range2Du32 const& range, MbtProps const& props)
{
	for(u32 y = range.y_begin; y < range.y_end; ++y)
	{		
		auto dst_row = dst.row_begin(y);
		r64 cy = props.min_my + y * props.my_step;

		for(u32 x = range.x_begin; x < range.x_end; ++x)
		{
			r64 cx = props.min_mx + x * props.mx_step;
			auto index = mandelbrot_color_index(cx, cy, props.iter_limit);
			dst_row[x] = index;
		}
	}
}


static Range2Du32 get_full_range(Mat2Di16 const& mat)
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

		auto& current_ids = state.color_indeces[state.ids_current];
		auto& old_ids = state.color_indeces[state.ids_old];

		auto& ids = state.color_indeces;
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


		//copy(state.iterations, state.pixel_shift);

		//mandelbrot(state.iterations, state);
		//find_min_max_iter(state);

		state.draw_new = true;
	}
    
    if(state.draw_new)
    {
        //draw(state.screen_buffer, state);
		draw(state.color_indeces[state.ids_current], state.screen_buffer, state.rgb_option);
    }
}