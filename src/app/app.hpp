#pragma once

#include "../input/input.hpp"

#include <functional>

namespace app
{
	// allocate memory
	constexpr u32 BUFFER_HEIGHT = 800;
	constexpr u32 BUFFER_WIDTH = BUFFER_HEIGHT * 16 / 9;


	using to_color32_f = std::function<u32(u8 red, u8 green, u8 blue)>;


	class AppMemory
	{
	public:
		b32 is_app_initialized;
		size_t permanent_storage_size;
		void* permanent_storage; // required to be zero at startup

		size_t transient_storage_size;
		void* transient_storage; // required to be zero at startup

	};


	class  PixelBuffer
	{
	public:
		void* memory;
		u32 width;
		u32 height;
		u32 bytes_per_pixel;

		//to_color32_f to_color32;
	};
	

	// app.cpp
	void update_and_render(AppMemory& memory, Input const& input, PixelBuffer const& buffer);

	// app.cpp
	void end_program();
}


// platform dependent e.g. win32_main.cpp
u32 platform_to_color_32(u8 red, u8 green, u8 blue);