#pragma once

#include "../input/input.hpp"


inline bool pan_right(Input const& input)
{
    return input.keyboard.right_key.is_down || input.controllers[0].stick_right_x.end >= 0.5f;
}


inline bool pan_left(Input const& input)
{
    return input.keyboard.left_key.is_down || input.controllers[0].stick_right_x.end <= -0.5f;
}


inline bool pan_up(Input const& input)
{
    return input.keyboard.up_key.is_down || input.controllers[0].stick_right_y.end >= 0.5f;
}


inline bool pan_down(Input const& input)
{
    return input.keyboard.down_key.is_down || input.controllers[0].stick_right_y.end <= -0.5f;
}


inline bool increase_zoom_speed(Input const& input)
{
    return input.keyboard.mult_key.is_down || input.controllers[0].trigger_right.end >= 0.5f;
}


inline bool decrease_zoom_speed(Input const& input)
{
    return input.keyboard.div_key.is_down || input.controllers[0].trigger_left.end >= 0.5f;
}


inline bool zoom_in(Input const& input)
{
    return input.keyboard.plus_key.is_down || input.controllers[0].stick_left_y.end >= 0.5f;
}


inline bool zoom_out(Input const& input)
{
    return input.keyboard.minus_key.is_down || input.controllers[0].stick_left_y.end <= -0.5f;
}


inline bool increase_resolution(Input const& input)
{
    return input.keyboard.f_key.is_down || input.controllers[0].shoulder_right.is_down;
}


inline bool decrease_resolution(Input const& input)
{
    return input.keyboard.d_key.is_down || input.controllers[0].shoulder_left.is_down;
}


inline bool stop_application(Input const& input)
{
    return input.controllers[0].button_b.pressed;
}


inline bool cycle_color_scheme_up(Input const& input)
{
    return input.controllers[0].dpad_up.pressed || input.controllers[0].dpad_right.pressed;
}

inline bool cycle_color_scheme_down(Input const& input)
{
    return input.controllers[0].dpad_down.pressed || input.controllers[0].dpad_left.pressed;
}