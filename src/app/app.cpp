#include "app.hpp"
#include "../utils/index_range.hpp"
#include "../utils/image.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <execution>
#include <cmath>
#include <immintrin.h>


constexpr r64 MBT_MIN_X = -2.0;
constexpr r64 MBT_MAX_X = 0.7;
constexpr r64 MBT_MIN_Y = -1.2;
constexpr r64 MBT_MAX_Y = 1.2;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;

constexpr size_t MAX_ITERATIONS = 500;

class Point2Dr64
{
public:
	r64 x;
	r64 y;
};


class Point2Di32
{
public:
	i32 x;
	i32 y;
};


class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
};


class AppState
{
public:
	bool render_new;

	Point2Dr64 screen_pos;
	Point2Di32 pixel_shift;
	r64 zoom;
};





static r64 screen_width(AppState const& state)
{
	return MBT_WIDTH / state.zoom;
}


static r64 screen_height(AppState const& state)
{
	return MBT_HEIGHT / state.zoom;
}


static image_t make_buffer_image(app::PixelBuffer const& buffer)
{
	assert(buffer.bytes_per_pixel == RGBA_CHANNELS);

	image_t image{};

	image.width = buffer.width;
	image.height = buffer.height;
	image.data = (pixel_t*)buffer.memory;

	return image;
}


static void copy_rows_up(image_t const& img, u32 n_rows)
{
	auto const width = static_cast<u64>(img.width);
	auto const height = static_cast<u64>(img.height);
	
	auto const y_begin = static_cast<u64>(n_rows);
	auto const y_end = height;

	for (u64 y = y_begin; y < y_end; ++y)
	{
		auto src_begin = img.row_begin(y);
		auto dst_begin = img.row_begin(y - n_rows);
		
		auto src_end = src_begin + width;

		std::copy(std::execution::par, src_begin, src_end, dst_begin);
	}
}


static void copy_rows_down(image_t const& img, u32 n_rows)
{
	auto const width = static_cast<u64>(img.width);
	auto const height = static_cast<u64>(img.height);

	auto const y_rbegin = static_cast<u64>(img.height - n_rows - 1);

	for (u64 i = 0; i < height - n_rows; ++i)
	{
		auto y = y_rbegin - i;

		auto src_begin = img.row_begin(y);
		auto dst_begin = img.row_begin(y + n_rows);
		
		auto src_end = src_begin + width;

		std::copy(std::execution::par, src_begin, src_end, dst_begin);
	}
}


static void copy_columns_left(image_t const& img, u32 n_cols)
{
	UnsignedRange y_ids(0u, img.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	auto const x_begin = n_cols;
	auto const x_end = static_cast<u64>(img.width);

	auto const copy_row_part = [&](u64 y)
	{
		auto row_begin = img.row_begin(y);
		for (u64 x = x_begin; x < x_end; ++x)
		{
			row_begin[x - n_cols] = row_begin[x];
		}
	};

	std::for_each(std::execution::par, y_id_begin, y_id_end, copy_row_part);
}


static void copy_columns_right(image_t const& img, u32 n_cols)
{
	UnsignedRange y_ids(0u, img.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	u64 const x_rbegin = static_cast<u64>(img.width - n_cols - 1);

	auto const copy_row_part = [&](u64 y)
	{
		auto row_begin = img.row_begin(y);
		for (u64 i = 0; i < img.width - n_cols;  ++i)
		{
			auto x = x_rbegin - i;
			row_begin[x + n_cols] = row_begin[x];
		}
	};

	std::for_each(std::execution::par, y_id_begin, y_id_end, copy_row_part);
}


static pixel_t to_platform_pixel(u8 red, u8 green, u8 blue)
{
	pixel_t p;
	p.value = platform_to_color_32(red, green, blue);

	return p;
}


static pixel_t to_platform_pixel(u8 gray)
{
	pixel_t p;
	p.value = platform_to_color_32(gray, gray, gray);

	return p;
}


static pixel_t to_platform_pixel(pixel_t const& p)
{
	return to_platform_pixel(p.red, p.green, p.blue);
}


static pixel_t to_platform_pixel(fpixel_t const& p)
{
	auto const r = static_cast<u8>(p.red);
	auto const g = static_cast<u8>(p.green);
	auto const b = static_cast<u8>(p.blue);

	return to_platform_pixel(r, g, b);
}


constexpr fpixel_t to_fpixel(r32 gray)
{
	fpixel_t p{};
	p.red = p.green = p.blue = gray;

	return p;
}


constexpr fpixel_t to_fpixel(r32 r, r32 g, r32 b)
{
	fpixel_t p{};
	p.red = r;
	p.green = g;
	p.blue = b;

	return p;
}


static void fill(image_t const& dst, pixel_t const& p)
{
	auto const platform_p = to_platform_pixel(p);
	std::fill(dst.begin(), dst.end(), platform_p);
}




using gray_palette_t = std::array<r32, MAX_ITERATIONS>;


constexpr gray_palette_t make_gray_palette()
{
	gray_palette_t palette = {};

	for (u32 i = 0; i < MAX_ITERATIONS; ++i)
	{
		auto ratio = static_cast<r64>(i) / (MAX_ITERATIONS - 1);
		palette[i] = static_cast<r32>(255.0 * ratio);
	}

	return palette;
}


constexpr fpixel_t to_gray(size_t index)
{
	constexpr auto palette = make_gray_palette();

	if (index >= palette.size())
	{
		index = palette.size() - 1;
	}

	return to_fpixel(palette[index]);
}


constexpr fpixel_t to_rgb(r64 ratio)
{
	assert(ratio >= 0.0);
	assert(ratio <= 1.0);

	auto const p = ratio;
	auto const q = 1.0 - p;
	auto const p2 = p * p;
	auto const p3 = p2 * p;
	auto const q2 = q * q;
	auto const q3 = q2 * q;

	auto const c_max = 255.0;

	auto const c1 = static_cast<r32>(9.0 * q * p3 * c_max);
	auto const c2 = static_cast<r32>(15.0 * q2 * p2 * c_max);
	auto const c3 = static_cast<r32>(8.5 * q3 * p * c_max);

	auto const r = c1;
	auto const g = c2;
	auto const b = c3;

	return to_fpixel(r, g, b);
}


using color_palette_t = std::array<fpixel_t, MAX_ITERATIONS>;


constexpr color_palette_t make_rgb_palette()
{
	color_palette_t palette = {};

	for (u32 i = 0; i < MAX_ITERATIONS; ++i)
	{
		auto ratio = static_cast<r64>(i) / (MAX_ITERATIONS - 1);
		palette[i] = to_rgb(ratio);
	}

	return palette;
}


constexpr fpixel_t to_color(size_t index)
{
	constexpr auto palette = make_rgb_palette();

	if (index >= palette.size())
	{
		index = palette.size() - 1;
	}

	return palette[index];
}





static UnsignedRange copy_y_range(image_t const& image, AppState const& state)
{
	u32 y_begin = 0u;
	u32 y_end = image.height;

	if (state.pixel_shift.y < 0)
	{
		y_end += state.pixel_shift.y;
	}
	else if (state.pixel_shift.y > 0)
	{
		y_begin += state.pixel_shift.y;
	}

	UnsignedRange y_ids(y_begin, y_end);

	return y_ids;
}





static Range2Du32 render_range(image_t const& image, AppState const& state)
{
	Range2Du32 range{};
	range.x_begin = 0;
	range.x_end = image.width;
	range.y_begin = 0;
	range.y_end = image.height;

	if (state.pixel_shift.x < 0)
	{
		range.x_begin = range.x_end + state.pixel_shift.x;
	}
	else if (state.pixel_shift.x > 0)
	{
		range.x_end = range.x_begin + state.pixel_shift.x;
	}

	if (state.pixel_shift.y < 0)
	{
		range.y_begin = range.y_end + state.pixel_shift.y;
	}
	else if (state.pixel_shift.y > 0)
	{
		range.y_end = range.y_begin + state.pixel_shift.y;
	}

	return range;
}


static void mandelbrot(image_t const& dst, AppState const& state, Range2Du32 const& range)
{
	UnsignedRange x_ids(range.x_begin, range.x_end);
	UnsignedRange y_ids(range.y_begin, range.y_end);

	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();
	auto const x_id_begin = x_ids.begin();
	auto const x_id_end = x_ids.end();

	auto const width = dst.width;
	auto const height = dst.height;

	constexpr u64 max_iter = MAX_ITERATIONS;
	constexpr r64 limit = 4.0;

	auto const x_pos = state.screen_pos.x;
	auto const y_pos = state.screen_pos.y;

	auto const scr_width = screen_width(state);
	auto const scr_height = screen_height(state);

	auto const min_re = MBT_MIN_X + x_pos;
	auto const min_im = MBT_MIN_Y + y_pos;

	auto const re_step = scr_width / width;
	auto const im_step = scr_height / height;

	auto const do_row = [&](u64 y)
	{
		r64 const ci = min_im + y * im_step;
		auto row = dst.row_begin(y);

		auto const do_x = [&](u64 x)
		{
			u32 iter = 0;
			r64 const cr = min_re + x * re_step;

			r64 re = 0.0;
			r64 im = 0.0;
			r64 re2 = 0.0;
			r64 im2 = 0.0;

			while (iter < max_iter && re2 + im2 <= limit)
			{
				im = (re + re) * im + ci;
				re = re2 - im2 + cr;
				im2 = im * im;
				re2 = re * re;

				++iter;
			}

			row[x] = to_platform_pixel(to_color(iter - 1));
		};

		std::for_each(std::execution::par, x_id_begin, x_id_end, do_x);
	};

	std::for_each(y_id_begin, y_id_end, do_row);
}


static void mandelbrot(image_t const& dst, AppState const& state)
{	
	auto mbt = render_range(dst, state);

	if (mbt.x_begin > 0)
	{
		auto n_cols = dst.width - mbt.x_begin;
		copy_columns_left(dst, n_cols);
	}
	else if (mbt.x_end < dst.width)
	{
		auto n_cols = mbt.x_end;
		copy_columns_right(dst, n_cols);
	}
	else if (mbt.y_begin > 0)
	{
		auto n_rows = dst.height - mbt.y_begin;
		copy_rows_up(dst, n_rows);
	}
	else if (mbt.y_end < dst.height)
	{
		auto n_rows = mbt.y_end;
		copy_rows_down(dst, n_rows);
	}

	mandelbrot(dst, state, mbt);
}






namespace app
{
	static void initialize_memory(AppState& state, PixelBuffer const& buffer)
	{
		state.render_new = true;

		state.screen_pos.x = 0.0;
		state.screen_pos.y = 0.0;

		state.zoom = 1.0;

		auto const width = buffer.width;
		auto const height = buffer.height;
	}


	static void process_input(Input const& input, AppState& state)
	{
		state.pixel_shift = { 0, 0 };

		i32 const pixel_shift = static_cast<i32>(std::round(PIXELS_PER_SECOND * input.dt_frame));

		r64 const zoom_per_second = 0.5;
		r64 zoom = 1.0 + zoom_per_second * input.dt_frame;

		if (input.keyboard.w_key.is_down)
		{
			auto old_w = screen_width(state);
			auto old_h = screen_height(state);
			state.zoom *= zoom;
			auto new_w = screen_width(state);
			auto new_h = screen_height(state);

			state.screen_pos.x += 0.5 * (old_w - new_w);	
			state.screen_pos.y += 0.5 * (old_h - new_h);

			state.render_new = true;
		}

		if (input.keyboard.s_key.is_down)
		{
			auto old_w = screen_width(state);
			auto old_h = screen_height(state);
			state.zoom /= zoom;
			auto new_w = screen_width(state);
			auto new_h = screen_height(state);

			state.screen_pos.x += 0.5 * (old_w - new_w);
			state.screen_pos.y += 0.5 * (old_h - new_h);

			state.render_new = true;
		}

		if (input.keyboard.right_key.is_down)
		{
			auto distance_per_pixel = screen_width(state) / BUFFER_WIDTH;

			state.pixel_shift.x -= pixel_shift;
			state.screen_pos.x += pixel_shift * distance_per_pixel;

			state.render_new = true;
		}

		if (input.keyboard.left_key.is_down)
		{
			auto distance_per_pixel = screen_width(state) / BUFFER_WIDTH;

			state.pixel_shift.x += pixel_shift;
			state.screen_pos.x -= pixel_shift * distance_per_pixel;

			state.render_new = true;
		}

		if (input.keyboard.up_key.is_down)
		{
			auto distance_per_pixel = screen_height(state) / BUFFER_HEIGHT;

			state.pixel_shift.y += pixel_shift;
			state.screen_pos.y -= pixel_shift * distance_per_pixel;

			state.render_new = true;
		}

		if (input.keyboard.down_key.is_down)
		{
			auto distance_per_pixel = screen_height(state) / BUFFER_HEIGHT;

			state.pixel_shift.y -= pixel_shift;
			state.screen_pos.y += pixel_shift * distance_per_pixel;

			state.render_new = true;
		}
	}


	void update_and_render(AppMemory& memory, Input const& input, PixelBuffer const& buffer)
	{
		auto const state_sz = sizeof(AppState);

		auto const required_sz = state_sz;

		assert(required_sz <= memory.permanent_storage_size);

		auto& state = *(AppState*)memory.permanent_storage;		

		if (!memory.is_app_initialized)
		{
			initialize_memory(state, buffer);
			memory.is_app_initialized = true;
		}

		process_input(input, state);		

		if (state.render_new)
		{
			auto buffer_image = make_buffer_image(buffer);
			mandelbrot(buffer_image, state);
		}

		state.render_new = false;
	}


	void end_program()
	{
		
	}
}


using r64_4 = __m256d;

inline r64_4 make_zero4()
{
	return _mm256_setzero_pd();
}


inline r64_4 make4(const r64* val)
{
	return _mm256_broadcast_sd(val);
}


inline r64_4 load4(const r64* begin)
{
	return _mm256_load_pd(begin);
}


inline r64_4 add4(r64_4 lhs, r64_4 rhs)
{
	return _mm256_add_pd(lhs, rhs);
}


inline r64_4 sub4(r64_4 lhs, r64_4 rhs)
{
	return _mm256_sub_pd(lhs, rhs);
}


inline r64_4 mul4(r64_4 lhs, r64_4 rhs)
{
	return _mm256_mul_pd(lhs, rhs);
}


inline r64_4 div4(r64_4 lhs, r64_4 rhs)
{
	return _mm256_div_pd(lhs, rhs);
}


inline r64_4 cmp_lt4(r64_4 lhs, r64_4 rhs)
{
	return _mm256_cmp_pd(lhs, rhs, _CMP_LT_OQ);
}


inline r64_4 cmp_ge4(r64_4 lhs, r64_4 rhs)
{
	return _mm256_cmp_pd(lhs, rhs, _CMP_GE_OQ);
}


inline void store4(r64_4 src, r64* dst)
{
	_mm256_store_pd(dst, src);
}



static void mandelbrot_simd(image_t const& dst, AppState const& state)
{
	constexpr u8 x_step = 4;
	constexpr u64 max_iter = MAX_ITERATIONS;

	r64 const limit_r64 = 4.0;

	auto const limit = make4(&limit_r64);

	auto const width_r64 = static_cast<r64>(dst.width);
	auto const height_r64 = static_cast<r64>(dst.height);
	auto const scr_width_r64 = screen_width(state);
	auto const scr_height_r64 = screen_height(state);

	auto const min_re = add4(make4(&MBT_MIN_X), make4(&state.screen_pos.x));
	auto const min_im = add4(make4(&MBT_MIN_Y), make4(&state.screen_pos.y));

	auto const re_step = div4(make4(&scr_width_r64), make4(&width_r64));
	auto const im_step = div4(make4(&scr_height_r64), make4(&height_r64));

	UnsignedRange y_ids(0u, dst.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	UnsignedRange x_ids(0u, dst.width / x_step);
	auto const x_id_begin = x_ids.begin();
	auto const x_id_end = x_ids.end();

	auto const do_row = [&](u64 y)
	{
		auto const y_r64 = static_cast<r64>(y);
		auto const y_4 = make4(&y_r64);

		auto const ci = mul4(add4(min_im, y_4), im_step);

		auto row = dst.row_begin(y);

		auto const do_x = [&](u64 x)
		{
			auto const x_begin = static_cast<r64>(x);
			r64 x_4[] = { x_begin, x_begin + 1.0, x_begin + 2.0, x_begin + 3.0 };

			auto const cr = mul4(add4(min_re, load4(x_4)), re_step);

			u32 iter = 0;

			auto re = make_zero4();
			auto im = make_zero4();
			auto re2 = make_zero4();
			auto im2 = make_zero4();

			u64 iters[] = { 0, 0, 0, 0 };
			u8 flags[] = { 1, 1, 1, 1 };
			auto const flag_on = [&]() { return flags[0] || flags[1] || flags[2] || flags[3]; };

			r64 comps[4] = {};

			while (iter < max_iter && flag_on())
			{
				im = add4(mul4(add4(re, re), im), ci);
				re = add4(sub4(re2, im2), cr);
				im2 = mul4(im, im);
				re2 = mul4(re, re);

				++iter;

				store4(cmp_ge4(add4(re2, im2), limit), comps);
				for (u32 i = 0; i < 4; ++i)
				{
					if (!flags[i])
					{
						continue;
					}

					if (comps[i] > 0.0)
					{
						flags[i] = 0;
					}
					else
					{
						iters[i] = iter;
					}
				}
			}

			for (u32 i = 0; i < 4; ++i)
			{
				row[x + i] = to_platform_pixel(to_color(iters[i] - 1));
			}
		};

		auto const do_x_id = [&](u64 x_id) { do_x(x_id * x_step); };

		std::for_each(x_id_begin, x_id_end, do_x_id);

		do_x(dst.width - x_step);
	};

	std::for_each(y_id_begin, y_id_end, do_row);
}