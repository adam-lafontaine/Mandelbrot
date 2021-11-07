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
constexpr r64 MBT_MAX_X = 1.0;
constexpr r64 MBT_MIN_Y = -1.0;
constexpr r64 MBT_MAX_Y = 1.0;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;

constexpr size_t MAX_ITERATIONS = 1000;


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


static fpixel_t to_fpixel(r32 gray)
{
	fpixel_t p;
	p.red = p.green = p.blue = gray;

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




using gray_palette_t = std::array<u8, MAX_ITERATIONS>;

constexpr gray_palette_t make_gray_palette()
{
	gray_palette_t palette = {};
	for (u32 i = 0; i < MAX_ITERATIONS; ++i)
	{
		auto ratio = static_cast<r64>(i) / (MAX_ITERATIONS - 1);
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


constexpr std::array<r32, MAX_ITERATIONS> make_gray_palette_r32()
{
	std::array<r32, MAX_ITERATIONS> palette = {};

	for (u32 i = 0; i < MAX_ITERATIONS; ++i)
	{
		auto ratio = static_cast<r64>(i) / (MAX_ITERATIONS - 1);
		palette[i] = static_cast<r32>(255.0 * ratio);
	}

	return palette;
}

constexpr r32 gray_palette_r32(size_t index)
{
	constexpr auto palette = make_gray_palette_r32();

	if (index >= palette.size())
	{
		index = palette.size() - 1;
	}

	return palette[index];
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

	auto const ci_step = (max_im - min_im) / height;
	auto const cr_step = (max_re - min_re) / width;

	UnsignedRange y_ids(0u, height);
	UnsignedRange x_ids(0u, width);

	u64 const max_iter = MAX_ITERATIONS;

	auto const do_row = [&](u64 y)
	{
		r64 ci = min_im + y * ci_step;
		auto row = state.screen.row_begin(y);

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

			row[x] = to_fpixel(gray_palette_r32(iter - 1));
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