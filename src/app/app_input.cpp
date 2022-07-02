#include "app_input.hpp"
#include "app.hpp"
#include "render_include.hpp"

#include <cmath>


constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = 50;
constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = 1000;
constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
constexpr r64 ZOOM_SPEED_LOWER_LIMIT = 1.0;


static bool pan_right(Input const& input)
{
    return 
        input.keyboard.d_key.is_down || 
        input.keyboard.np_six_key.is_down || 
        input.controllers[0].stick_right_x.end >= 0.5f;
}


static bool pan_left(Input const& input)
{
    return 
        input.keyboard.a_key.is_down || 
        input.keyboard.np_four_key.is_down || 
        input.controllers[0].stick_right_x.end <= -0.5f;
}


static bool pan_up(Input const& input)
{
    return 
        input.keyboard.w_key.is_down || 
        input.keyboard.np_eight_key.is_down || 
        input.controllers[0].stick_right_y.end >= 0.5f;
}


static bool pan_down(Input const& input)
{
    return 
        input.keyboard.s_key.is_down || 
        input.keyboard.np_two_key.is_down || 
        input.controllers[0].stick_right_y.end <= -0.5f;
}


static bool increase_zoom_speed(Input const& input)
{
    return 
        input.keyboard.mult_key.is_down || 
        input.controllers[0].trigger_right.end >= 0.5f;
}


static bool decrease_zoom_speed(Input const& input)
{
    return 
        input.keyboard.div_key.is_down || 
        input.controllers[0].trigger_left.end >= 0.5f;
}


static bool zoom_in(Input const& input)
{
    return 
        input.keyboard.plus_key.is_down || 
        input.controllers[0].stick_left_y.end >= 0.5f;
}


static bool zoom_out(Input const& input)
{
    return 
        input.keyboard.minus_key.is_down || 
        input.controllers[0].stick_left_y.end <= -0.5f;
}


static bool increase_resolution(Input const& input)
{
    return 
        input.keyboard.up_key.is_down ||
        input.controllers[0].dpad_up.is_down;
}


static bool decrease_resolution(Input const& input)
{
    return 
        input.keyboard.down_key.is_down ||
        input.controllers[0].dpad_down.is_down;
}


static bool cycle_color_scheme_right(Input const& input)
{
    return 
        input.keyboard.right_key.pressed || 
        input.controllers[0].dpad_right.pressed ||
        input.controllers[0].shoulder_right.pressed;
}


static bool cycle_color_scheme_left(Input const& input)
{
    return 
        input.keyboard.left_key.pressed || 
        input.controllers[0].dpad_left.pressed ||
        input.controllers[0].shoulder_left.pressed;
}


static bool stop_application(Input const& input)
{
    return 
        input.keyboard.escape_key.pressed || 
        input.controllers[0].button_b.pressed;
}


void set_rgb_channels(ChannelOptions& options, u32 rgb_option)
{
	auto& c1 = options.channel1;
    auto& c2 = options.channel2;
    auto& c3 = options.channel3;

	switch (rgb_option)
	{
	case 1:
		c1 = 0;
		c2 = 1;
		c3 = 2;
		break;
	case 2:
		c1 = 0;
		c2 = 2;
		c3 = 1;
		break;
	case 3:
		c1 = 1;
		c2 = 0;
		c3 = 2;
		break;
	case 4:
		c1 = 1;
		c2 = 2;
		c3 = 0;
		break;
	case 5:
		c1 = 2;
		c2 = 0;
		c3 = 1;
		break;
	case 6:
		c1 = 2;
		c2 = 1;
		c3 = 0;
		break;
	}
}


void init_app_input(AppInput& state)
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


void process_input(Input const& input, AppInput& state)
{
	constexpr r64 zoom_speed_factor_per_second = 0.1;
	constexpr r64 iteration_adjustment_factor = 0.005;

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

	u32 qty = 6;
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