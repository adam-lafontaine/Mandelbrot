#include "app.hpp"
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

		auto const width = buffer.width;
		auto const height = buffer.height;

		assert(width == screen_buffer_width());
		assert(height == screen_buffer_height());

		UnifiedState unified{};		

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
		DeviceState device{};

        auto const width = buffer.width;
		auto const height = buffer.height;

		assert(width == screen_buffer_width());
		assert(height == screen_buffer_height());

		auto const n_id_matrices = 2;
        auto const n_pixels = width * height;

		if(!cuda::device_malloc(state.device_pixel_buffer, n_pixels))
		{
			print_error("device pixel_buffer");
			return false;
		}

		auto screen_data = cuda::push_elements(state.device_pixel_buffer, n_pixels);
		if(!screen_data)
		{
			print_error("screen data");
			return false;
		}

		device.screen_pixels.data = screen_data;
		device.screen_pixels.width = width;
		device.screen_pixels.height = height;

		state.device_pixels = device.screen_pixels;

		state.screen_pixels.data = (Pixel*)buffer.memory;
		state.screen_pixels.width = width;
		state.screen_pixels.height = height;

		if(!cuda::device_malloc(state.device_i32_buffer, n_id_matrices * n_pixels))
		{
			print_error("device_i32_buffer");
			return false;
		}

		for(u32 i = 0; i < n_id_matrices; ++i)
		{
			auto& id_matrix = device.color_ids[i];
			id_matrix.data = cuda::push_elements(state.device_i32_buffer, n_pixels);
			if(!id_matrix.data)
			{
				print_error("color_ids");
				return false;
			}

			id_matrix.width = width;
			id_matrix.height = height;
		}

		auto& palette = device.color_palette;	
		
		if(!cuda::device_malloc(state.device_u8_buffer, 3 * N_COLORS))
		{
			print_error("device_u8_buffer");
			return false;
		}

		palette.channel1 = cuda::push_elements(state.device_u8_buffer, N_COLORS);
		if(!palette.channel1)
		{
			print_error("channel1");
			return false;
		}

		palette.channel2 = cuda::push_elements(state.device_u8_buffer, N_COLORS);
		if(!palette.channel2)
		{
			print_error("channel2");
			return false;
		}

		palette.channel3 = cuda::push_elements(state.device_u8_buffer, N_COLORS);
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

		if(!cuda::memcpy_to_device(&device, state.device.data, sizeof(DeviceState)))
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

		auto src = state.device_pixels.data;
        auto dst = state.screen_pixels.data;
        auto size = state.device_pixels.width * state.device_pixels.height * sizeof(Pixel);

        cuda::memcpy_to_host(src, dst, size);
	}


	void end_program(AppMemory& memory)
	{
		auto& state = get_state(memory);

		cuda::free(state.device_i32_buffer);
		cuda::free(state.device_u8_buffer);
		cuda::free(state.device_pixel_buffer);

		cuda::free(state.device);
		cuda::free(state.unified);
	}


	u32 screen_buffer_width()
    {
        return SCREEN_WIDTH_PX;
    }

	
	u32 screen_buffer_height()
    {
        return SCREEN_HEIGHT_PX;
    }
}