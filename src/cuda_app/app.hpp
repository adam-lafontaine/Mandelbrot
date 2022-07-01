#pragma once

#include "../input/input.hpp"

namespace app
{
	constexpr auto APP_TITLE = "CUDA Mandelbrot";

	// allocate memory
	constexpr u32 BUFFER_HEIGHT = 800;
	constexpr u32 BUFFER_WIDTH = BUFFER_HEIGHT * 9 / 8;
	constexpr u32 PIXELS_PER_SECOND = (u32)(0.4 * BUFFER_HEIGHT);


	class AppMemory
	{
	public:
		bool is_app_initialized;
		size_t permanent_storage_size;
		void* permanent_storage;
	};


	class  ScreenBuffer
	{
	public:
		void* memory;
		u32 width;
		u32 height;
		u32 bytes_per_pixel;
	};


	// app.cpp
	bool initialize_memory(AppMemory& memory, ScreenBuffer& buffer);

	// app.cpp
	void update_and_render(AppMemory& memory, Input const& input);

	// app.cpp
	void end_program(AppMemory& memory);
}


void platform_signal_stop();