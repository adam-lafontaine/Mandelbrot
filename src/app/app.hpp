#pragma once

#include "../input/input.hpp"

namespace app
{
	constexpr auto APP_TITLE = "Mandelbrot";
	constexpr auto VERSION = "1.1.1";

	#ifdef SDL2_WASM

	constexpr u32 BUFFER_WIDTH = 640;
	constexpr u32 BUFFER_HEIGHT = BUFFER_WIDTH * 8 / 9;

	#else

	// allocate memory
	constexpr u32 BUFFER_HEIGHT = 800;
	constexpr u32 BUFFER_WIDTH = BUFFER_HEIGHT * 9 / 8;

	#endif

	constexpr u32 PIXELS_PER_SECOND = (u32)(0.2 * BUFFER_HEIGHT);


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


	class DebugInfo
	{
	public:
		u32 max_iter;
		r64 zoom;
	};


	// app.cpp
	bool initialize_memory(AppMemory& memory, ScreenBuffer& buffer);	

	// app.cpp
	void update_and_render(AppMemory& memory, Input const& input, DebugInfo& dbg);

	// app.cpp
	void end_program(AppMemory& memory);
}


// platform dependent e.g. win32_main.cpp
//u32 platform_to_color_32(u8 red, u8 green, u8 blue);

void platform_signal_stop();