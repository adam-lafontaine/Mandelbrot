#include "app.hpp"
#include "../utils/libimage/rgba.hpp"
#include "../utils/index_range.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <execution>
#include <immintrin.h>


namespace img = libimage;


constexpr std::array<img::pixel_t, 3> COLOR_OPTIONS = 
{
	img::to_pixel(255, 0, 0),
	img::to_pixel(0, 255, 0),
	img::to_pixel(0, 0, 255)
};


constexpr u32 next_option(u32 option)
{
	switch (option)
	{
	case 0:
	case 1:
		return option + 1;
	}

	return 0;
}


constexpr r64 MBT_MIN_X = -2.0;
constexpr r64 MBT_MAX_X = 1.0;
constexpr r64 MBT_MIN_Y = -1.0;
constexpr r64 MBT_MAX_Y = 1.0;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;


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


static img::image_t make_buffer_image(app::PixelBuffer const& buffer)
{
	assert(buffer.bytes_per_pixel == img::RGBA_CHANNELS);

	img::image_t image;

	image.width = buffer.width;
	image.height = buffer.height;
	image.data = (img::pixel_t*)buffer.memory;

	return image;
}


static img::pixel_t to_platform_pixel(u8 red, u8 green, u8 blue)
{
	img::pixel_t p;
	p.value = platform_to_color_32(red, green, blue);

	return p;
}


static img::pixel_t to_platform_pixel(u8 gray)
{
	img::pixel_t p;
	p.value = platform_to_color_32(gray, gray, gray);

	return p;
}


static img::pixel_t to_platform_pixel(img::pixel_t const& p)
{
	return to_platform_pixel(p.red, p.green, p.blue);
}


static void fill(img::image_t const& dst, img::pixel_t const& p)
{
	auto const platform_p = to_platform_pixel(p);
	std::fill(dst.begin(), dst.end(), platform_p);
}


constexpr size_t MAX_ITERATIONS = 500;

using gray_palette_t = std::array<u8, MAX_ITERATIONS>;

constexpr gray_palette_t make_gray_palette()
{
	std::array<u8, MAX_ITERATIONS> palette = {};
	for (u32 i = 0; i < MAX_ITERATIONS; ++i)
	{
		auto ratio = static_cast<double>(i) / (MAX_ITERATIONS - 1);
		palette[i] = static_cast<u8>(255 * ratio);
	}

	return palette;
}


constexpr u8 gray_palette(size_t index)
{
	constexpr auto palette = make_gray_palette();

	if (index >= palette.size())
	{
		index = palette.size() - 1;
	}

	return palette[index];
}


static void mandelbrot(img::image_t const& dst, AppState& state)
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

	auto const ci_step = (max_im - min_im) / height;
	auto const cr_step = (max_re - min_re) / width;
	
	UnsignedRange y_ids(0u, height);
	UnsignedRange x_ids(0u, width);

	u64 const max_iter = MAX_ITERATIONS;


	auto const do_row = [&](u64 y) 
	{
		r64 ci = min_im + y * ci_step;
		auto row = dst.row_begin(y);

		auto const do_simd = [&]() 
		{
			
		};

		auto const do_x = [&](u64 x) 
		{
			r64 cr = min_re + x * cr_step;

			r64 re = 0.0;
			r64 im = 0.0;

			u32 iter = 0;

			for (u32 i = 0; i < max_iter && re * re + im * im <= 4.0; ++i)
			{
				r64 tr = re * re - im * im + cr;
				im = 2 * re * im + ci;
				re = tr;

				++iter;
			}

			row[x] = to_platform_pixel(gray_palette(iter - 1));
		};
		
		std::for_each(std::execution::par, x_ids.begin(), x_ids.end(), do_x);
	};

	std::for_each(std::execution::par, y_ids.begin(), y_ids.end(), do_row);
}



namespace app
{
	static void initialize_memory(AppState& state)
	{
		state.render_new = true;

		state.screen_pos.x = 0.0;
		state.screen_pos.y = 0.0;

		state.zoom = 1.0;
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
		assert(sizeof(AppState) <= memory.permanent_storage_size);

		auto& state = *(AppState*)memory.permanent_storage;

		if (!memory.is_app_initialized)
		{
			initialize_memory(state);
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