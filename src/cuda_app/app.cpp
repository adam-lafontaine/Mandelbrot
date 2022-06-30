#include "app.hpp"
#include "../app/input_controls.hpp"
#include "../app/colors.hpp"
#include "render.hpp"

#include <cmath>


constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = 50;
constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = 5000;
constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
constexpr r64 ZOOM_SPEED_LOWER_LIMIT = 1.0;


static void process_input(Input const& input, AppState& state)
{
	constexpr r64 zoom_speed_factor_per_second = 0.1;
	constexpr r64 iteration_adjustment_factor = 0.02;

	auto zoom_speed_factor = 1.0 + zoom_speed_factor_per_second * input.dt_frame;

	state.pixel_shift = { 0, 0 };

	i32 const pixel_shift = (i32)(std::round(app::PIXELS_PER_SECOND * input.dt_frame));

	r64 const zoom_per_second = 0.5;
	auto const zoom = [&]() { return state.zoom_speed * (1.0 + zoom_per_second * input.dt_frame); };

	auto direction = false;
    
	if (pan_right(input))
	{
		auto distance_per_pixel = state.mbt_screen_width / app::BUFFER_WIDTH;

		state.pixel_shift.x -= pixel_shift;
		state.mbt_pos.x += pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
	if (pan_left(input))
	{
		auto distance_per_pixel = state.mbt_screen_width / app::BUFFER_WIDTH;

		state.pixel_shift.x += pixel_shift;
		state.mbt_pos.x -= pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
	if (pan_up(input))
	{
		auto distance_per_pixel = state.mbt_screen_height / app::BUFFER_HEIGHT;

		state.pixel_shift.y += pixel_shift;
		state.mbt_pos.y -= pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
	if (pan_down(input))
	{
		auto distance_per_pixel = state.mbt_screen_height / app::BUFFER_HEIGHT;

		state.pixel_shift.y -= pixel_shift;
		state.mbt_pos.y += pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
    
	if (increase_zoom_speed(input))
	{
		state.zoom_speed *= zoom_speed_factor;
		state.render_new = true;
	}
	if (state.zoom_speed > ZOOM_SPEED_LOWER_LIMIT && decrease_zoom_speed(input))
	{
		state.zoom_speed = std::max(state.zoom_speed / zoom_speed_factor, ZOOM_SPEED_LOWER_LIMIT);

		state.render_new = true;
	}    

	if (zoom_in(input) && !direction)
	{
		auto old_w = state.mbt_screen_width;
		auto old_h = state.mbt_screen_height;

		state.zoom_level *= zoom();

		state.mbt_screen_width = mbt_screen_width(state.zoom_level);
		state.mbt_screen_height = mbt_screen_height(state.zoom_level);

		state.mbt_pos.x += 0.5 * (old_w - state.mbt_screen_width);
		state.mbt_pos.y += 0.5 * (old_h - state.mbt_screen_height);

		state.render_new = true;
	}
	if (zoom_out(input) && !direction)
	{
		auto old_w = state.mbt_screen_width;
		auto old_h = state.mbt_screen_height;

		state.zoom_level /= zoom();

		state.mbt_screen_width = mbt_screen_width(state.zoom_level);
		state.mbt_screen_height = mbt_screen_height(state.zoom_level);

		state.mbt_pos.x += 0.5 * (old_w - state.mbt_screen_width);
		state.mbt_pos.y += 0.5 * (old_h - state.mbt_screen_height);

		state.render_new = true;
	}
    
	if (state.iter_limit < MAX_ITERATIONS_UPPER_LIMIT && increase_resolution(input))
	{
		u32 adj = (u32)(iteration_adjustment_factor * state.iter_limit);
		adj = std::max(adj, 5u);

		state.iter_limit = std::min(state.iter_limit + adj, MAX_ITERATIONS_UPPER_LIMIT);
		state.render_new = true;
	}
	if (state.iter_limit > MAX_ITERTAIONS_LOWER_LIMIT && decrease_resolution(input))
	{
		u32 adj = (u32)(iteration_adjustment_factor * state.iter_limit);
		adj = std::max(adj, 5u);

		state.iter_limit = std::max(state.iter_limit - adj, MAX_ITERTAIONS_LOWER_LIMIT);
		state.render_new = true;
	}

	u32 qty = num_rgb_combinations();
    if(cycle_color_scheme_right(input))
    {
        ++state.rgb_option;
        if(state.rgb_option > qty)
        {
            state.rgb_option = 1;
        }

        state.draw_new = true;
    }
    if(cycle_color_scheme_left(input))
    {
        --state.rgb_option;
        if(state.rgb_option < 1)
        {
            state.rgb_option = qty;
        }

        state.draw_new = true;
    }

    if(stop_application(input))
    {
        platform_signal_stop();
    }
}


namespace app
{
    static AppState& get_state(AppMemory& memory)
    {
		assert(sizeof(AppState) <= memory.permanent_storage_size);

        return *(AppState*)memory.permanent_storage;
    }


	static bool init_unified_memory(UnifiedMemory& unified, ScreenBuffer& buffer)
	{
		assert(sizeof(pixel_t) == buffer.bytes_per_pixel);

		auto const width = buffer.width;
		auto const height = buffer.height;

        auto const n_pixels = width * height;

		auto const screen_sz = sizeof(pixel_t) * n_pixels;

		auto unified_sz = screen_sz;

		DevicePointer screen_ptr{};
		if(!cuda::unified_malloc(screen_ptr, screen_sz))
		{
			return false;
		}

		auto& screen = unified.screen_buffer;

		screen.data = (Pixel*)screen_ptr.data;
		screen.width = width;
		screen.height = height;

		return true;
	}


    static bool init_device_memory(DeviceMemory& device, ScreenBuffer const& buffer)
    {
        auto const width = buffer.width;
		auto const height = buffer.height;

        auto const n_pixels = width * height;
        
		auto const ids_sz = sizeof(i32) * n_pixels;

		DevicePointer ids_ptr0{};
		if(!cuda::device_malloc(ids_ptr0, ids_sz));
		{
			return false;
		}

		auto& ids0 = device.color_ids[0];
		ids0.data = (i32*)ids_ptr0.data;
		ids0.width = width;
		ids0.height = height;

		DevicePointer ids_ptr1{};
		if(!cuda::device_malloc(ids_ptr1, ids_sz));
		{
			return false;
		}

		auto& ids1 = device.color_ids[1];
		ids1.data = (i32*)ids_ptr1.data;
		ids1.width = width;
		ids1.height = height;

		auto const color_palette_channel_sz = sizeof(u8) * N_COLORS;

		auto& palette = device.color_palette;		

		DevicePointer ch_ptr1{};
		if(!cuda::device_malloc(ch_ptr1, color_palette_channel_sz))
		{
			return false;
		}

		palette.channel1 = (u8*)ch_ptr1.data;

		DevicePointer ch_ptr2{};
		if(!cuda::device_malloc(ch_ptr2, color_palette_channel_sz))
		{
			return false;
		}

		palette.channel2 = (u8*)ch_ptr2.data;

		DevicePointer ch_ptr3{};
		if(!cuda::device_malloc(ch_ptr3, color_palette_channel_sz))
		{
			return false;
		}

		palette.channel3 = (u8*)ch_ptr3.data;

		palette.n_colors = N_COLORS;

		if(!cuda::memcpy_to_device(palettes[0].data(), palette.channel1, color_palette_channel_sz))
		{
			return false;
		}

		if(!cuda::memcpy_to_device(palettes[1].data(), palette.channel2, color_palette_channel_sz))
		{
			return false;
		}

		if(!cuda::memcpy_to_device(palettes[2].data(), palette.channel3, color_palette_channel_sz))
		{
			return false;
		}

        return true;
    }


	bool initialize_memory(AppMemory& memory, ScreenBuffer& buffer)
	{
		auto& state = get_state(memory);

		if(!init_unified_memory(state.unified, buffer))
		{
			return false;
		}

		if(!init_device_memory(state.device, buffer))
        {
            return false;
        }

		state.render_new = true;

		state.mbt_pos.x = 0.0;
		state.mbt_pos.y = 0.0;

		state.zoom_level = 1.0;
		state.zoom_speed = ZOOM_SPEED_LOWER_LIMIT;

        state.mbt_screen_width = mbt_screen_width(state.zoom_level);
		state.mbt_screen_height = mbt_screen_height(state.zoom_level);

		state.iter_limit = MAX_ITERATIONS_START;

		state.rgb_option = 1;		

		memory.is_app_initialized = true;
        return true;
	}


	void update_and_render(AppMemory& memory, Input const& input)
	{
		if (!memory.is_app_initialized)
		{
			return;
		}

		auto& state = get_state(memory);

		process_input(input, state);

		render(state);

		state.render_new = false;
		state.draw_new = false;
	}


	void end_program(AppMemory& memory)
	{
		auto& state = get_state(memory);

		cuda::free((void*)(state.device.color_ids[0].data));
		cuda::free((void*)(state.device.color_ids[1].data));

		cuda::free((void*)(state.unified.screen_buffer.data));
	}
}