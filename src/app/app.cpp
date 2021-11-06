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


class AppState
{
public:
	u32 current_option;

	double min_re;
	double max_re;
	double min_im;
	double max_im;
	u32 max_iter;
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


constexpr size_t maxymax = std::numeric_limits<size_t>::max();


static void mandelbrot(img::image_t const& dst, AppState& state)
{
	auto const width = dst.width;
	auto const height = dst.height;
	auto const min_re = state.min_re;
	auto const max_re = state.max_re;
	auto const min_im = state.min_im;
	auto const max_im = state.max_im;
	auto const max_iter = state.max_iter;

	for (u32 y = 0; y < height; ++y)
	{
		double ci = min_im + (max_im - min_im) * y / height;
		auto row = dst.row_begin(y);
		for (u32 x = 0; x < width; ++x)
		{
			/*double cr = min_re + (max_re - min_re) * x / width;			
			double re = 0, im = 0;

			u32 iter = 0;

			for (iter = 0; iter < max_iter && re * re + im * im > 4.0; iter++)
			{
				double tr = re * re - im * im + cr;
				im = 2 * re * im + ci;
				re = tr;
			}*/

			double cr = min_re + (max_re - min_re) * x / width;
			double ci = min_im + (max_im - min_im) * y / height;
			double re = 0, im = 0;
			u32 iter;
			for (iter = 0; iter < max_iter; iter++)
			{
				double tr = re * re - im * im + cr;
				im = 2 * re * im + ci;
				re = tr;
				if (re * re + im * im > 4.0) break;
			}

			auto ratio = static_cast<double>(max_iter - iter) / max_iter;

			auto shade = static_cast<u8>(ratio * 255);

			row[x] = to_platform_pixel(shade);
		}
	}
}



namespace app
{
	static void initialize_memory(AppState& state)
	{
		state.current_option = 1;

		state.max_iter = 100;
		state.min_re = -2.5;
		state.max_re = 1.0;
		state.min_im = -1.0;
		state.max_im = 1.0;
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

		/*if (input.keyboard.space_key.pressed || input.mouse.left.pressed)
		{
			state.current_option = next_option(state.current_option); 
		}

		fill(buffer_image, COLOR_OPTIONS[state.current_option]);*/

		mandelbrot(buffer_image, state);

	}


	void end_program()
	{
		
	}
}