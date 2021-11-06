#include "app.hpp"
#include "../utils/libimage/rgba.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>


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


constexpr r64 MBT_MIN_X = -2.5;
constexpr r64 MBT_MAX_X = 1.0;
constexpr r64 MBT_MIN_Y = -1.0;
constexpr r64 MBT_MAX_Y = 1.0;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;


class Point3Dr64
{
public:
	r64 x;
	r64 y;
	r64 z;
};


class AppState
{
public:
	bool render_new;

	double min_re;
	double max_re;
	double min_im;
	double max_im;

	Point3Dr64 screen_pos;
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


using uP = u32;
constexpr auto uP_max = std::numeric_limits<uP>::max();


constexpr size_t MAX_ITERATIONS = 100;

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


constexpr long double dec = 0.00000000000001;
constexpr auto smallest = std::numeric_limits<double>::epsilon();

constexpr auto incr = smallest * uP_max;

constexpr u8 gray_palette(u32 index)
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
	auto const z_pos = state.screen_pos.z;

	auto const screen_width = MBT_WIDTH - 2.0 * z_pos;
	auto const screen_height = MBT_HEIGHT - 2.0 * z_pos;

	auto const min_re = MBT_MIN_X + x_pos;
	auto const max_re = min_re + screen_width;
	auto const min_im = MBT_MIN_Y + y_pos;
	auto const max_im = min_im + screen_height;

	double min_value = 5.0;

	for (u32 y = 0; y < height; ++y)
	{
		double ci = min_im + (max_im - min_im) * y / height;

		auto row = dst.row_begin(y);
		for (u32 x = 0; x < width; ++x)
		{
			double cr = min_re + (max_re - min_re) * x / width;

			double re = 0.0;
			double im = 0.0;

			u32 iter = 0;
			
			for (u32 i = 0; i < MAX_ITERATIONS && re * re + im * im <= 4.0; ++i)
			{
				double tr = re * re - im * im + cr;
				im = 2 * re * im + ci;
				re = tr;

				++iter;
			}

			row[x] = to_platform_pixel(gray_palette(iter - 1));
		}
	}
}



namespace app
{
	static void initialize_memory(AppState& state)
	{
		state.render_new = true;

		state.screen_pos.x = 0.5;
		state.screen_pos.y = 0.1;
		state.screen_pos.z = 0.0;
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

		auto buffer_image = make_buffer_image(buffer);

		if (input.keyboard.space_key.pressed || input.mouse.left.pressed)
		{
			state.render_new = true;
		}

		if (state.render_new)
		{
			mandelbrot(buffer_image, state);
			state.render_new = false;
		}
	}


	void end_program()
	{
		
	}
}