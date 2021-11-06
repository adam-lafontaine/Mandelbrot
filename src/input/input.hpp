#pragma once

#include "keyboard.hpp"
#include "mouse.hpp"


typedef struct input_t
{
	KeyboardInput keyboard;
	MouseInput mouse;

	r32 dt_frame;

} Input;


inline void reset_button_state(ButtonState& state)
{
	for (u32 i = 0; i < ArrayCount(state.states); ++i)
	{
		state.states[i] = false;
	}
}



inline void copy_button_state(ButtonState const& src, ButtonState& dst)
{
	for (u32 i = 0; i < ArrayCount(src.states); ++i)
	{
		dst.states[i] = src.states[i];
	}
}


inline void copy_keyboard_state(KeyboardInput const& src, KeyboardInput& dst)
{
	for (u32 i = 0; i < ArrayCount(src.keys); ++i)
	{
		copy_button_state(src.keys[i], dst.keys[i]);
	}
}


inline void reset_mouse(MouseInput& mouse)
{
	mouse.mouse_x = 0.0f;
	mouse.mouse_y = 0.0f;
	mouse.mouse_z = 0.0f;

	for (u32 i = 0; i < ArrayCount(mouse.buttons); ++i)
	{
		reset_button_state(mouse.buttons[i]);
	}
}
