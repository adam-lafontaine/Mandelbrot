#include "app.hpp"
#include "render.hpp"
#include "input_controls.hpp"

#include <algorithm>
#include <cmath>


constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = 50;
constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = 5000;
constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
constexpr r64 ZOOM_SPEED_LOWER_LIMIT = 1.0;

static void init_app_input(AppInput& state)
{
	state.render_new = true;		

	state.mbt_pos.x = 0.0;
	state.mbt_pos.y = 0.0;

	state.zoom_level = 1.0;
	state.zoom_speed = ZOOM_SPEED_LOWER_LIMIT;

	state.iter_limit = MAX_ITERATIONS_START;

	state.mbt_screen_width = mbt_screen_width(state.zoom_level);
	state.mbt_screen_height = mbt_screen_height(state.zoom_level);

	state.rgb_option = 1;
}


static void process_input(Input const& input, AppInput& state)
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

	u32 qty = get_rgb_combo_qty();
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


	bool initialize_memory(AppMemory& memory, ScreenBuffer const& buffer)
	{
		auto const width = buffer.width;
		auto const height = buffer.height;

		auto const state_sz = sizeof(AppState);
		auto const color_sz = sizeof(i32) * width * height;

		auto const required_sz = state_sz + 2 *color_sz;

		assert(required_sz <= memory.permanent_storage_size);

		auto& state = get_state(memory);		

		auto begin = (u8*)(&state);
		size_t offset = state_sz;

		state.color_ids[0].width = width;
		state.color_ids[0].height = height;
		state.color_ids[0].data = (i32*)(begin + offset);

		offset += color_sz;

		state.color_ids[1].width = width;
		state.color_ids[1].height = height;
		state.color_ids[1].data = (i32*)(begin + offset);

		state.screen_buffer = make_buffer_image(buffer);

		init_app_input(state.app_input);

		memory.is_app_initialized = true;

		return true;
	}


	void update_and_render(AppMemory& memory, Input const& input, ScreenBuffer const& buffer)
	{
		if (!memory.is_app_initialized)
		{
			return;
		}

		auto& state = get_state(memory);

		process_input(input, state.app_input);

		render(state);

		state.app_input.render_new = false;
		state.app_input.draw_new = false;
	}


	void end_program(AppMemory& memory)
	{
		
	}
}