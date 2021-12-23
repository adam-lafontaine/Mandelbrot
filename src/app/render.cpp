#include "render.hpp"
#include "colors.hpp"
#include "../utils/index_range.hpp"

#include <cassert>
#include <algorithm>
#include <cmath>
#include <functional>


using ur_it = UnsignedRange::iterator;

#ifdef NO_CPP_17

static void for_each(ur_it const& begin, ur_it const& end, std::function<void(u32)> const& func)
{
	std::for_each(begin, end, func);
}


static void transform(u32* src_begin, u32* src_end, pixel_t* dst_begin, std::function<pixel_t(u32)> const& f)
{
	std::transform(src_begin, src_end, dst_begin, f);
}


static void copy(u32* src_begin, u32* src_end, u32* dst_begin)
{
	std::copy(src_begin, src_end, dst_begin);
}

#else

#include <execution>

static void for_each(ur_it const& begin, ur_it const& end, std::function<void(u32)> const& func)
{
	std::for_each(std::execution::par, begin, end, func);
}


static void transform(u32* src_begin, u32* src_end, pixel_t* dst_begin, std::function<pixel_t(u32)> const& f)
{
	std::transform(std::execution::par, src_begin, src_end, dst_begin, f);
}


static void copy(u32* src_begin, u32* src_end, u32* dst_begin)
{
	std::copy(std::execution::par, src_begin, src_end, dst_begin);
}

#endif



// platform dependent e.g. win32_main.cpp
u32 platform_to_color_32(u8 red, u8 green, u8 blue);


static Range2Du32 get_range(mat_u32_t const& mat)
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
}





static void for_each_row(mat_u32_t const& mat, std::function<void(u32 y)> const& func)
{
	UnsignedRange y_ids(0u, mat.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	for_each(y_id_begin, y_id_end, func);
}


static void copy_left(mat_u32_t const& mat, u32 n_cols)
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


static void copy_right(mat_u32_t const& mat, u32 n_cols)
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


static void copy(mat_u32_t const& mat, Vec2Di32 const& direction)
{
	auto up = direction.y < 0;
	auto right = direction.x > 0;

	auto const n_cols = static_cast<u32>(std::abs(direction.x));
	auto const n_rows = static_cast<u32>(std::abs(direction.y));

	if (n_cols == 0 && n_rows == 0)
	{
		return;
	}

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


static void mandelbrot(mat_u32_t const& dst, AppState const& state)
{
	auto const x_pos = state.mbt_pos.x;
	auto const y_pos = state.mbt_pos.y;

	auto const min_re = MBT_MIN_X + x_pos;
	auto const min_im = MBT_MIN_Y + y_pos;

	auto const re_step = state.mbt_screen_width / dst.width;
	auto const im_step = state.mbt_screen_height / dst.height;

	auto const do_mandelbrot = [&](Range2Du32 const& range)
	{
		auto x_ids = UnsignedRange(range.x_begin, range.x_end);
		auto y_ids = UnsignedRange(range.y_begin, range.y_end);

		auto y_id_begin = y_ids.begin();
		auto y_id_end = y_ids.end();
		auto x_id_begin = x_ids.begin();
		auto x_id_end = x_ids.end();

		auto const do_row = [&](u32 y)
		{
			r64 const ci = min_im + y * im_step;
			auto row = dst.row_begin(y);

			auto const do_x = [&](u32 x)
			{
				u32 iter = 0;
				r64 const cr = min_re + x * re_step;

				r64 re = 0.0;
				r64 im = 0.0;
				r64 re2 = 0.0;
				r64 im2 = 0.0;

				while (iter < state.iter_limit && re2 + im2 <= 4.0)
				{
					im = (re + re) * im + ci;
					re = re2 - im2 + cr;
					im2 = im * im;
					re2 = re * re;

					++iter;
				}

				--iter;

				row[x] = iter;
			};

			for_each(x_id_begin, x_id_end, do_x);
		};

		for_each(y_id_begin, y_id_end, do_row);
	};

	auto& shift = state.pixel_shift;

	auto do_left = shift.x > 0;
	auto do_top = shift.y > 0;
	//auto do_right = shift.x < 0;	
	//auto do_bottom = shift.y < 0;

	auto const n_cols = static_cast<u32>(std::abs(shift.x));
	auto const n_rows = static_cast<u32>(std::abs(shift.y));

	auto no_horizontal = n_cols == 0;
	auto no_vertical = n_rows == 0;

	auto r = get_range(dst);

	if (no_horizontal && no_vertical)
	{
		do_mandelbrot(r);
		return;
	}

	if (no_horizontal)
	{
		if (do_top)
		{
			r.y_end = n_rows;
		}
		else // if (do_bottom)
		{
			r.y_begin = dst.height - 1 - n_rows;
		}

		do_mandelbrot(r);
		return;
	}

	if (no_vertical)
	{
		if (do_left)
		{
			r.x_end = n_cols;
		}
		else // if (do_right)
		{
			r.x_begin = dst.width - 1 - n_cols;
		}

		do_mandelbrot(r);
		return;
	}

	auto r2 = r;

	if (do_top)
	{
		r.y_end = n_rows;
		r2.y_begin = n_rows;
	}
	else // if (do_bottom)
	{
		r.y_begin = dst.height - 1 - n_rows;
		r2.y_end = dst.height - 1 - n_rows;
	}

	if (do_left)
	{
		r2.x_end = n_cols;
	}
	else // if (do_right)
	{
		r2.x_begin = dst.width - 1 - n_cols;
	}

	do_mandelbrot(r);
	do_mandelbrot(r2);	
}


static void find_min_max_iter(AppState& state)
{
	auto& mat = state.iterations;
	//auto [mat_min, mat_max] = std::minmax_element(mat.begin(), mat.end());
	//auto min = *mat_min;
	//auto max = *mat_max;

	auto mat_min_max = std::minmax_element(mat.begin(), mat.end());
	state.iter_min = *mat_min_max.first;
	state.iter_max = *mat_min_max.second;
}


static void draw(image_t const& dst, AppState const& state)
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


void render(AppState& state)
{
	copy(state.iterations, state.pixel_shift);

	mandelbrot(state.iterations, state);
	find_min_max_iter(state);

	draw(state.screen_buffer, state);
}