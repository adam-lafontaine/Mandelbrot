#include "app.hpp"
#include "render.hpp"
#include "app_input.hpp"


namespace app
{
	static Image make_buffer_image(ScreenBuffer const& buffer)
	{
		assert(buffer.bytes_per_pixel == RGBA_CHANNELS);

		Image image{};

		image.width = buffer.width;
		image.height = buffer.height;
		image.data = (pixel_t*)buffer.memory;

		return image;
	}


	static AppState& get_state(AppMemory& memory)
	{
		return *(AppState*)memory.permanent_storage;
	}


	bool initialize_memory(AppMemory& memory, ScreenBuffer& buffer)
	{
		auto const width = buffer.width;
		auto const height = buffer.height;

		auto const state_sz = sizeof(AppState);
		auto const iter_sz = sizeof(u32) * width * height;

		auto const required_sz = state_sz + 2 * iter_sz;

		assert(required_sz <= memory.permanent_storage_size);

		auto& state = get_state(memory);		

		auto begin = (u8*)(&state);
		size_t offset = state_sz;

		for(u32 i = 0; i < 2; ++i)
		{
			state.iterations[i].width = width;
			state.iterations[i].height = height;

			state.iterations[i].data = (u32*)(begin + offset);
			offset += iter_sz;
		}

		state.screen_buffer = make_buffer_image(buffer);

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
		
	}
}