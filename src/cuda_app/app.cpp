#include "../app/app.hpp"
#include "render.hpp"
#include "../app/app_input.hpp"
#include "../app/colors.hpp"


#include <cmath>

#ifndef NDEBUG
#define PRINT_APP_ERROR
#endif

#ifdef PRINT_APP_ERROR
#include <cstdio>
#endif

static void print_error(cstr msg)
{
#ifdef PRINT_APP_ERROR
	printf("\n*** APP ERROR ***\n\n");
	printf("%s", msg);
	printf("\n\n******************\n\n");
#endif
}


namespace app
{
    static AppState& get_state(AppMemory& memory)
    {
		assert(sizeof(AppState) <= memory.permanent_storage_size);

        return *(AppState*)memory.permanent_storage;
    }


	static bool init_unified_memory(AppState& state, ScreenBuffer& buffer)
	{
		assert(sizeof(pixel_t) == buffer.bytes_per_pixel);

		UnifiedMemory unified{};
		auto& screen = unified.screen_buffer;

		auto const width = buffer.width;
		auto const height = buffer.height;

        auto const n_pixels = width * height;

		if(!cuda::unified_malloc(state.unified_pixel, n_pixels))
		{
			print_error("unified_pixel");
			return false;
		}

		assert(state.unified_pixel.data);

		screen.data = cuda::push_elements(state.unified_pixel, n_pixels);
		if(!screen.data)
		{
			print_error("screen data");
			return false;
		}

		screen.width = width;
		screen.height = height;

		buffer.memory = screen.data;

		if(!cuda::unified_malloc(state.unified, 1))
		{
			print_error("state.unified");
			return false;
		}

		state.unified.data[0] = unified;

		return true;
	}


    static bool init_device_memory(AppState& state, ScreenBuffer const& buffer)
    {
		DeviceMemory device{};

        auto const width = buffer.width;
		auto const height = buffer.height;

		auto const n_id_matrices = 2;
        auto const n_pixels = width * height;

		if(!cuda::device_malloc(state.device_i32, n_id_matrices * n_pixels))
		{
			print_error("device_i32");
			return false;
		}

		for(u32 i = 0; i < n_id_matrices; ++i)
		{
			auto& id_matrix = device.color_ids[i];
			id_matrix.data = cuda::push_elements(state.device_i32, n_pixels);
			if(!id_matrix.data)
			{
				print_error("color_ids");
				return false;
			}

			id_matrix.width = width;
			id_matrix.height = height;
		}

		auto& palette = device.color_palette;	
		
		if(!cuda::device_malloc(state.device_u8, 3 * N_COLORS))
		{
			print_error("device_u8");
			return false;
		}

		palette.channel1 = cuda::push_elements(state.device_u8, N_COLORS);
		if(!palette.channel1)
		{
			print_error("channel1");
			return false;
		}

		palette.channel2 = cuda::push_elements(state.device_u8, N_COLORS);
		if(!palette.channel2)
		{
			print_error("channel2");
			return false;
		}

		palette.channel3 = cuda::push_elements(state.device_u8, N_COLORS);
		if(!palette.channel3)
		{
			print_error("channel3");
			return false;
		}

		palette.n_colors = N_COLORS;

		auto const palette_channel_sz = sizeof(u8) * N_COLORS;

		if(!cuda::memcpy_to_device(palettes[0].data(), palette.channel1, palette_channel_sz))
		{
			print_error("memcpy channel1");
			return false;
		}

		if(!cuda::memcpy_to_device(palettes[1].data(), palette.channel2, palette_channel_sz))
		{
			print_error("memcpy channel2");
			return false;
		}

		if(!cuda::memcpy_to_device(palettes[2].data(), palette.channel3, palette_channel_sz))
		{
			print_error("memcpy channel3");
			return false;
		}

		if(!cuda::device_malloc(state.device, 1))
		{
			print_error("state.device");
			return false;
		}

		if(!cuda::memcpy_to_device(&device, state.device.data, sizeof(DeviceMemory)))
		{
			print_error("memcpy device");
			return false;
		}

        return true;
    }


	bool initialize_memory(AppMemory& memory, ScreenBuffer& buffer)
	{
		auto& state = get_state(memory);

		if(!init_unified_memory(state, buffer))
		{
			print_error("unified memory");
			return false;
		}

		if(!init_device_memory(state, buffer))
        {
			print_error("device memory");
            return false;
        }

		init_app_input(state.app_input);		

		memory.is_app_initialized = true;
        return true;
	}


	void update_and_render(AppMemory& memory, Input const& input, DebugInfo& dbg)
	{
		if (!memory.is_app_initialized)
		{
			return;
		}

		auto& state = get_state(memory);

		process_input(input, state.app_input);

		dbg.max_iter = state.app_input.iter_limit;
		dbg.zoom = state.app_input.zoom_level;

		render(state);

		state.app_input.render_new = false;
		state.app_input.draw_new = false;
	}


	void end_program(AppMemory& memory)
	{
		auto& state = get_state(memory);

		cuda::free(state.device_i32);
		cuda::free(state.device_u8);
		cuda::free(state.unified_pixel);
	}
}