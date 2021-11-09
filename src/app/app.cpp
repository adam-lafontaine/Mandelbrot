#include "app.hpp"
#include "../utils/index_range.hpp"
#include "../utils/image.hpp"
#include "../utils/fixed_point.hpp"

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

using r64fp = sg14::make_fixed<1, 62>;


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

	fimage_t screen;
};


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


void copy_to_platform(fimage_t const& src, image_t const& dst)
{
	auto const convert = [](fpixel_t const& p)
	{
		auto const r = static_cast<u8>(p.red);
		auto const g = static_cast<u8>(p.green);
		auto const b = static_cast<u8>(p.blue);

		return to_platform_pixel(r, g, b);
	};

	std::transform(std::execution::par, src.begin(), src.end(), dst.begin(), convert);
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


static inline r64fp fp_add(r64fp lhs, r64fp rhs)
{
	return sg14::add(lhs, rhs);
}


static inline r64fp fp_sub(r64fp lhs, r64fp rhs)
{
	return sg14::subtract(lhs, rhs);
}


static inline r64fp fp_div(r64fp lhs, r64fp rhs)
{
	return sg14::divide(lhs, rhs);
}


static inline r64fp fp_mul(r64fp lhs, r64fp rhs)
{
	return sg14::multiply(lhs, rhs);
}


template <typename T>
static inline r64fp to_fp(T val)
{
	return r64fp{ val };
}


static void mandelbrot_fp(image_t const& dst, AppState const& state)
{
	auto const width = to_fp(dst.width);
	auto const height = to_fp(dst.height);

	auto const x_pos = to_fp(state.screen_pos.x);
	auto const y_pos = to_fp(state.screen_pos.y);
	auto const zoom = to_fp(state.zoom);

	auto const screen_width = fp_div(to_fp(MBT_WIDTH), zoom);
	auto const screen_height = fp_div(to_fp(MBT_HEIGHT), zoom);

	auto const min_re = fp_add(to_fp(MBT_MIN_X), x_pos);
	auto const max_re = fp_add(min_re, screen_width);
	auto const min_im = fp_add(to_fp(MBT_MIN_Y), y_pos);
	auto const max_im = fp_add(min_im, screen_height);

	auto const re_step = fp_div(fp_sub(max_re, min_re), width);
	auto const im_step = fp_div(fp_sub(max_im, min_im), height);

	UnsignedRange y_ids(0u, dst.height);
	UnsignedRange x_ids(0u, dst.width);

	u64 const max_iter = MAX_ITERATIONS;

	auto const limit = to_fp(4.0);

	auto const do_row = [&](u64 y)
	{
		auto const yfp = to_fp(y);
		auto const ci = fp_add(min_im, fp_mul(yfp, im_step));

		auto row = state.screen.row_begin(y);

		auto const do_x = [&](u64 x)
		{
			auto const xfp = to_fp(x);
			auto const cr = fp_add(min_re, fp_mul(x, re_step));

			auto re = to_fp(0.0);
			auto im = to_fp(0.0);
			auto re2 = to_fp(0.0);
			auto im2 = to_fp(0.0);

			u64 iter = 0;			

			while (fp_add(re2, im2) <= limit && iter < max_iter)
			{
				im = fp_add(fp_mul(fp_add(re, re), im), ci);
				re = re2 - im2 + cr;
				im2 = im * im;
				re2 = re * re;

				++iter;
			}

			//row[x] = to_gray(iter - 1);
			row[x] = to_color(iter - 1);
		};

		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), do_x);
	};

	std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), do_row);

	copy_to_platform(state.screen, dst);
}


static void mandelbrot(image_t const& dst, AppState const& state)
{
	auto const width = dst.width;
	auto const height = dst.height;

	auto const x_pos = state.screen_pos.x;
	auto const y_pos = state.screen_pos.y;
	auto const zoom = state.zoom;

	auto const screen_width = MBT_WIDTH / zoom;
	auto const screen_height = MBT_HEIGHT / zoom;

	auto const min_re = MBT_MIN_X + x_pos;
	auto const max_re = min_re + screen_width;
	auto const min_im = MBT_MIN_Y + y_pos;
	auto const max_im = min_im + screen_height;

	auto const re_step = (max_re - min_re) / width;
	auto const im_step = (max_im - min_im) / height;

	UnsignedRange y_ids(0u, height);
	UnsignedRange x_ids(0u, width);

	u64 const max_iter = MAX_ITERATIONS;
	r64 const limit = 4.0;

	auto const do_row = [&](u64 y)
	{
		r64 const ci = min_im + y * im_step;
		auto row = state.screen.row_begin(y);

		auto const do_x = [&](u64 x)
		{
			r64 const cr = min_re + x * re_step;

			r64 re = 0.0;
			r64 im = 0.0;
			r64 re2 = 0.0;
			r64 im2 = 0.0;

			u64 iter = 0;

			while (re2 + im2 <= limit && iter < max_iter)
			{
				im = (re + re) * im + ci;
				re = re2 - im2 + cr;
				im2 = im * im;
				re2 = re * re;

				++iter;
			}

			//row[x] = to_gray(iter - 1);
			row[x] = to_color(iter - 1);
		};

		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), do_x);
	};

	std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), do_row);

	copy_to_platform(state.screen, dst);
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

		state.screen.width = width;
		state.screen.height = height;
		state.screen.data = (fpixel_t*)((u8*)(&state) + sizeof(AppState));

		assert(state.screen.data);
	}


	static r64 screen_width(AppState const& state)
	{
		return MBT_WIDTH / state.zoom;
	}


	static r64 screen_height(AppState const& state)
	{
		return MBT_HEIGHT / state.zoom;
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
		auto const screen_data_sz = sizeof(fpixel_t) * buffer.width * buffer.height;

		auto const required_sz = state_sz + screen_data_sz;

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