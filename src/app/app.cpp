#include "app.hpp"
#include "../utils/libimage/rgba.hpp"

#include <cassert>
#include <algorithm>
#include <array>


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


static img::pixel_t to_platform_pixel(img::pixel_t const& p)
{
	return to_platform_pixel(p.red, p.green, p.blue);
}


static void fill(img::image_t const& dst, img::pixel_t const& p)
{
	auto const platform_p = to_platform_pixel(p);
	std::fill(dst.begin(), dst.end(), platform_p);
}



namespace app
{
	static void initialize_memory(AppMemory& memory, AppState& state)
	{
		state.current_option = 1;
	}


	void update_and_render(AppMemory& memory, Input const& input, PixelBuffer const& buffer)
	{
		assert(sizeof(AppState) <= memory.permanent_storage_size);

		auto& state = *(AppState*)memory.permanent_storage;

		if (!memory.is_app_initialized)
		{
			initialize_memory(memory, state);
			memory.is_app_initialized = true;
		}

		auto buffer_image = make_buffer_image(buffer);

		if (input.keyboard.space_key.pressed || input.mouse.left.pressed)
		{
			state.current_option = next_option(state.current_option); 
		}

		fill(buffer_image, COLOR_OPTIONS[state.current_option]);

	}


	void end_program()
	{
		
	}
}