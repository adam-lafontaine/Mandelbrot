#include "app.hpp"
#include "../utils/index_range.hpp"
#include "../utils/image.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <execution>
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


class AppState
{
public:
	bool render_new;

	Point2Dr64 screen_pos;
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


using r64_4 = __m256d;

r64_4 make_zero()
{
	return _mm256_setzero_pd();
}


r64_4 add(r64_4 lhs, r64_4 rhs)
{
	return _mm256_add_pd(lhs, rhs);
}


r64_4 sub(r64_4 lhs, r64_4 rhs)
{
	return _mm256_sub_pd(lhs, rhs);
}


r64_4 mul(r64_4 lhs, r64_4 rhs)
{
	return _mm256_mul_pd(lhs, rhs);
}


static void mandelbrot(image_t const& dst, AppState const& state)
{
	constexpr u64 max_iter = MAX_ITERATIONS;
	constexpr r64 limit = 4.0;

	auto const width = dst.width;
	auto const height = dst.height;

	auto const x_pos = state.screen_pos.x;
	auto const y_pos = state.screen_pos.y;

	auto const scr_width = screen_width(state);
	auto const scr_height = screen_height(state);

	auto const min_re = MBT_MIN_X + x_pos;
	auto const max_re = min_re + scr_width;
	auto const min_im = MBT_MIN_Y + y_pos;
	auto const max_im = min_im + scr_height;

	auto const re_step = (max_re - min_re) / width;
	auto const im_step = (max_im - min_im) / height;

	UnsignedRange y_ids(0u, height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	UnsignedRange x_ids(0u, width);
	auto const x_id_begin = x_ids.begin();
	auto const x_id_end = x_ids.end();	

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

			while (re2 + im2 <= limit && iter < max_iter)
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

	std::for_each(std::execution::par, y_id_begin, y_id_end, do_row);
}


static void mandelbrot_simd(image_t const& dst, AppState const& state)
{
	constexpr u8 x_step = 4;
	constexpr u64 max_iter = MAX_ITERATIONS;
	constexpr r64 limit = 4.0;

	auto const width = dst.width;
	auto const height = dst.height;

	auto const x_pos = state.screen_pos.x;
	auto const y_pos = state.screen_pos.y;

	auto const scr_width = screen_width(state);
	auto const scr_height = screen_height(state);

	auto const min_re = MBT_MIN_X + x_pos;
	auto const max_re = min_re + scr_width;
	auto const min_im = MBT_MIN_Y + y_pos;
	auto const max_im = min_im + scr_height;

	auto const re_step = (max_re - min_re) / width;
	auto const im_step = (max_im - min_im) / height;

	UnsignedRange y_ids(0u, height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	UnsignedRange x_ids(0u, width / x_step);
	auto const x_id_begin = x_ids.begin();
	auto const x_id_end = x_ids.end();

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

			while (re2 + im2 <= limit && iter < max_iter)
			{
				im = (re + re) * im + ci;
				re = re2 - im2 + cr;
				im2 = im * im;
				re2 = re * re;

				++iter;
			}

			row[x] = to_platform_pixel(to_color(iter - 1));
		};

		auto const do_x_id = [&](u64 x_id) { do_x(x_id * x_step); };

		std::for_each(std::execution::par, x_id_begin, x_id_end, do_x_id);

		do_x(width - x_step);
	};

	std::for_each(std::execution::par, y_id_begin, y_id_end, do_row);
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
		r64 const pixels_per_second = 0.4 * BUFFER_HEIGHT;
		r64 pixel_distance = pixels_per_second * input.dt_frame;

		r64 const zoom_per_second = 0.5;
		r64 zoom = 1.0 + zoom_per_second * input.dt_frame;

		auto& pos = state.screen_pos;

		if (input.keyboard.w_key.is_down)
		{
			auto old_w = screen_width(state);
			auto old_h = screen_height(state);
			state.zoom *= zoom;
			auto new_w = screen_width(state);
			auto new_h = screen_height(state);

			pos.x += 0.5 * (old_w - new_w);	
			pos.y += 0.5 * (old_h - new_h);

			state.render_new = true;
		}

		if (input.keyboard.s_key.is_down)
		{
			auto old_w = screen_width(state);
			auto old_h = screen_height(state);
			state.zoom /= zoom;
			auto new_w = screen_width(state);
			auto new_h = screen_height(state);

			pos.x += 0.5 * (old_w - new_w);
			pos.y += 0.5 * (old_h - new_h);

			state.render_new = true;
		}

		if (input.keyboard.right_key.is_down)
		{
			auto distance_per_pixel = screen_width(state) / BUFFER_WIDTH;
			pos.x += pixel_distance * distance_per_pixel;

			state.render_new = true;
		}

		if (input.keyboard.left_key.is_down)
		{
			auto distance_per_pixel = screen_width(state) / BUFFER_WIDTH;
			pos.x -= pixel_distance * distance_per_pixel;

			state.render_new = true;
		}

		if (input.keyboard.up_key.is_down)
		{
			auto distance_per_pixel = screen_height(state) / BUFFER_HEIGHT;
			pos.y -= pixel_distance * distance_per_pixel;

			state.render_new = true;
		}

		if (input.keyboard.down_key.is_down)
		{
			auto distance_per_pixel = screen_height(state) / BUFFER_HEIGHT;
			pos.y += pixel_distance * distance_per_pixel;

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
			mandelbrot_simd(buffer_image, state);
		}

		state.render_new = false;
	}


	void end_program()
	{
		
	}
}