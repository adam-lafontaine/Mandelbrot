#pragma once

#include "keyboard.hpp"
#include "mouse.hpp"


typedef struct input_t
{
	KeyboardInput keyboard;
	MouseInput mouse;

} Input;



inline void reset_button_state(ButtonState& state)
{
	for (u32 i = 0; i < ArrayCount(state.states); ++i)
	{
		state.states[i] = false;
	}
}


inline void reset_keyboard(KeyboardInput& keyboard)
{
	for (u32 i = 0; i < ArrayCount(keyboard.keys); ++i)
	{
		reset_button_state(keyboard.keys[i]);
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
