#include "app.hpp"
#include "render.hpp"
#include "input_controls.hpp"

#include <algorithm>
#include <cmath>


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

    if(cycle_color_scheme_up(input))
    {
        ++state.rgb_option;
        if(state.rgb_option > 6)
        {
            state.rgb_option = 1;
        }

        state.draw_new = true;
    }
    if(cycle_color_scheme_down(input))
    {
        --state.rgb_option;
        if(state.rgb_option < 1)
        {
            state.rgb_option = 6;
        }

        state.draw_new = true;
    }

	// color scheme with 1 - 6
	if (input.keyboard.one_key.is_down && state.rgb_option != 1)
	{
		state.rgb_option = 1;
		state.draw_new = true;
	}
	if (input.keyboard.two_key.is_down && state.rgb_option != 2)
	{
		state.rgb_option = 2;
		state.draw_new = true;
	}
	if (input.keyboard.three_key.is_down && state.rgb_option != 3)
	{
		state.rgb_option = 3;
		state.draw_new = true;
	}
	if (input.keyboard.four_key.is_down && state.rgb_option != 4)
	{
		state.rgb_option = 4;
		state.draw_new = true;
	}
	if (input.keyboard.five_key.is_down && state.rgb_option != 5)
	{
		state.rgb_option = 5;
		state.draw_new = true;
	}
	if (input.keyboard.six_key.is_down && state.rgb_option != 6)
	{
		state.rgb_option = 6;
		state.draw_new = true;
	}

    if(stop_application(input))
    {
        platform_signal_stop();
    }
}


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


	static AppState& get_state(AppMemory& memory, ScreenBuffer const& buffer)
	{
		auto const state_sz = sizeof(AppState);
		auto const iter_sz = sizeof(u32) * buffer.width * buffer.height;

		auto const required_sz = state_sz + iter_sz;

		assert(required_sz <= memory.permanent_storage_size);

		auto& state = *(AppState*)memory.permanent_storage;

		return state;
	}


	bool initialize_memory(AppMemory& memory, ScreenBuffer const& buffer)
	{
		auto& state = get_state(memory, buffer);

		state.render_new = true;

		state.screen_buffer = make_buffer_image(buffer);

		state.mbt_pos.x = 0.0;
		state.mbt_pos.y = 0.0;

		state.zoom_level = 1.0;
		state.zoom_speed = ZOOM_SPEED_LOWER_LIMIT;

		state.iter_limit = MAX_ITERATIONS_START;

        state.mbt_screen_width = mbt_screen_width(state.zoom_level);
		state.mbt_screen_height = mbt_screen_height(state.zoom_level);

		state.rgb_option = 1;

		auto const width = buffer.width;
		auto const height = buffer.height;

		state.iterations.width = width;
		state.iterations.height = height;
		state.iterations.data = (u32*)((u8*)(&state) + sizeof(u32) * width * height);

		memory.is_app_initialized = true;

		return true;
	}


	void update_and_render(AppMemory& memory, Input const& input, ScreenBuffer const& buffer)
	{
		if (!memory.is_app_initialized)
		{
			return;
		}

		auto& state = get_state(memory, buffer);

		process_input(input, state);

		render(state);

		state.render_new = false;
		state.draw_new = false;
	}


	void end_program(AppMemory& memory)
	{
		
	}
}