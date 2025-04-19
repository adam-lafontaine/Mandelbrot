#pragma once

#include "input.hpp"


/* helpers */

namespace input
{
	inline void record_button_input(ButtonState const& old_state, ButtonState& new_state, b32 is_down)
	{
		new_state.pressed = !old_state.is_down && is_down;
		new_state.is_down = is_down;
		new_state.raised = old_state.is_down && !is_down;
	}


	inline void record_button_input(ButtonState& state, b32 is_down)
	{
		state.pressed = !state.is_down && is_down;
		state.raised = state.is_down && !is_down;
		state.is_down = is_down;		
	}


	inline void copy_button_state(ButtonState const& src, ButtonState& dst)
	{
		dst.is_down = src.is_down;
		dst.pressed = 0;
		dst.raised = 0;
	}


	inline void reset_button_state(ButtonState& btn)
	{
		btn.is_down = 0;
		btn.pressed = 0;
		btn.raised = 0;
	}


	template <typename T>
	inline void copy_vec_2d(Vec2D<T> const& src, Vec2D<T>& dst)
	{
		dst.x = src.x;
		dst.y = src.y;
	}


	template <typename T>
	inline void copy_vector_state(VectorState<T> const& src, VectorState<T>& dst)
	{
		copy_vec_2d(src.vec, dst.vec);
		dst.magnitude = src.magnitude;
		copy_vec_2d(src.unit_direction, dst.unit_direction);		
	}


	template <typename T>
	inline void reset_vec_2d(Vec2D<T>& vec)
	{
		vec.x = (T)0;
		vec.y = (T)0;
	}


	template <typename T>
	inline void reset_vector_state(VectorState<T>& vs)
	{
		reset_vec_2d(vs.vec);
		vs.magnitude = 0.0f;
		reset_vec_2d(vs.unit_direction);
	}
	
}


/* keyboard */

namespace input
{
	inline void set_is_active(KeyboardInput& kbd)
	{
		kbd.is_active = false;

		for (u32 i = 0; i < N_KEYBOARD_KEYS; ++i)
		{
			kbd.is_active |= kbd.keys[i].is_down;
		}
	}


	inline void copy_keyboard_state(KeyboardInput const& src, KeyboardInput& dst)
	{
		for (u32 i = 0; i < N_KEYBOARD_KEYS; ++i)
		{
			copy_button_state(src.keys[i], dst.keys[i]);
		}

		set_is_active(dst);
	}


	inline void reset_keyboard_state(KeyboardInput& kbd)
	{
		for (u32 i = 0; i < N_KEYBOARD_KEYS; i++)
		{
			reset_button_state(kbd.keys[i]);
		}

		kbd.is_active = false;
	}
}


/* mouse */

namespace input
{
	inline void set_is_active(MouseInput& mouse)
	{
		// .is_active is initially set in event listener

	#if MOUSE_POSITION

	#endif
	#if MOUSE_WHEEL

	#endif

		for (u32 i = 0; i < N_MOUSE_BUTTONS; ++i)
		{
			mouse.is_active |= mouse.buttons[i].is_down;
		}
	}

	
	inline void copy_mouse_position(MouseInput const& src, MouseInput& dst)
	{
	#if MOUSE_POSITION
		copy_vec_2d(src.window_pos, dst.window_pos);
	#endif
	}


	inline void reset_mouse_position(MouseInput& mouse)
	{
	#if MOUSE_POSITION
		reset_vec_2d(mouse.window_pos);
	#endif
	}


	inline void reset_mouse_wheel(MouseInput& mouse)
	{
	#if MOUSE_WHEEL
		reset_vec_2d(mouse.wheel);
	#endif
	}	


	inline void copy_mouse_state(MouseInput const& src, MouseInput& dst)
	{
		for (u32 i = 0; i < N_MOUSE_BUTTONS; ++i)
		{
			copy_button_state(src.buttons[i], dst.buttons[i]);
		}

		copy_mouse_position(src, dst);
		reset_mouse_wheel(dst);

		dst.is_active = false;
		set_is_active(dst);
	}


	inline void reset_mouse_state(MouseInput& mouse)
	{
		for (u32 i = 0; i < N_MOUSE_BUTTONS; ++i)
		{
			reset_button_state(mouse.buttons[i]);
		}

		reset_mouse_position(mouse);
		reset_mouse_wheel(mouse);
		mouse.is_active = false;
	}

	
}


/* controller */

namespace input
{
	inline void set_is_active(ControllerInput& controller)
    {
        controller.is_active = false ||

	#if CONTROLLER_TRIGGER_LEFT
        controller.trigger_left > 0.0f ||
	#endif
	#if CONTROLLER_TRIGGER_RIGHT
        controller.trigger_right > 0.0f ||
	#endif

	#if CONTROLLER_AXIS_STICK_LEFT
        controller.stick_left.magnitude > 0.0f ||
	#endif
	#if CONTROLLER_AXIS_STICK_RIGHT
        controller.stick_right.magnitude > 0.0f ||
	#endif
        false;

		if (!controller.is_active)
		{
			for (u32 i = 0; i < N_CONTROLLER_BUTTONS; i++)
			{
				controller.is_active |= controller.buttons[i].is_down;
			}
		}
    }


	inline void copy_controller_axes(ControllerInput const& src, ControllerInput& dst)
	{
	#if CONTROLLER_AXIS_STICK_LEFT
		copy_vector_state(src.stick_left, dst.stick_left);
	#endif

	#if CONTROLLER_AXIS_STICK_RIGHT
		copy_vector_state(src.stick_right, dst.stick_right);
	#endif
	}


	inline void reset_controller_axes(ControllerInput& controller)
	{
	#if CONTROLLER_AXIS_STICK_LEFT
		reset_vector_state(controller.stick_left);
	#endif

	#if CONTROLLER_AXIS_STICK_RIGHT
		reset_vector_state(controller.stick_right);
	#endif
	}


	inline void copy_controller_triggers(ControllerInput const& src, ControllerInput& dst)
	{
	#if CONTROLLER_TRIGGER_LEFT
		dst.trigger_left = src.trigger_left;
	#endif

	#if CONTROLLER_TRIGGER_RIGHT
		dst.trigger_right = src.trigger_right;
	#endif
	}


	inline void reset_controller_triggers(ControllerInput& controller)
	{
	#if CONTROLLER_TRIGGER_LEFT
		controller.trigger_left = 0.0f;
	#endif

	#if CONTROLLER_TRIGGER_RIGHT
		controller.trigger_right = 0.0f;
	#endif
	}


	inline void copy_controller_state(ControllerInput const& src, ControllerInput& dst)
	{
		for (u32 i = 0; i < N_CONTROLLER_BUTTONS; ++i)
		{
			copy_button_state(src.buttons[i], dst.buttons[i]);
		}

		copy_controller_axes(src, dst);
		copy_controller_triggers(src, dst);
		dst.is_active = false;
	}


	inline void reset_controller_state(ControllerInput& controller)
	{
		for (u32 i = 0; i < N_CONTROLLER_BUTTONS; ++i)
		{
			reset_button_state(controller.buttons[i]);
		}

		reset_controller_axes(controller);
		reset_controller_triggers(controller);

		controller.is_active = false;
		set_is_active(controller);
	}
	
}


/* joystic */

namespace input
{
	inline void set_is_active(JoystickInput& jsk)
	{
		jsk.is_active = false;

		for (u32 i = 0; i < N_JOYSTICK_BUTTONS; ++i)
		{
			jsk.is_active |= jsk.buttons[i].is_down;
		}

		jsk.is_active |= jsk.vec_axis.vec.x;
		jsk.is_active |= jsk.vec_axis.vec.y;
	}


	inline void copy_joystick_state(JoystickInput const& src, JoystickInput& dst)
	{
		for (u32 i = 0; i < N_JOYSTICK_BUTTONS; ++i)
		{
			copy_button_state(src.buttons[i], dst.buttons[i]);
		}

		copy_vector_state(src.vec_axis, dst.vec_axis);

		set_is_active(dst);
	}


	inline void reset_joystick_state(JoystickInput& jsk)
	{
		for (u32 i = 0; i < N_JOYSTICK_BUTTONS; ++i)
		{
			reset_button_state(jsk.buttons[i]);
		}

		reset_vector_state(jsk.vec_axis);

		jsk.is_active = false;
	}
}


/* copy/reset input */

namespace input
{
	inline void copy_input_state(Input const& src, Input& dst)
	{
		copy_keyboard_state(src.keyboard, dst.keyboard);
		copy_mouse_state(src.mouse, dst.mouse);

		for (u32 i = 0; i < MAX_CONTROLLERS; i++)
		{
			copy_controller_state(src.controllers[i], dst.controllers[i]);
		}

		for (u32 i = 0; i < MAX_JOYSTICKS; i++)
		{
			copy_joystick_state(src.joysticks[i], dst.joysticks[i]);
		}
	}


	inline void reset_input_state(Input& input)
	{
		input.dt_frame = 0.0f;
		input.frame = (u64)0 - 1;
		input.window_size_changed = 0;
		
		reset_keyboard_state(input.keyboard);
		reset_mouse_state(input.mouse);

		for (u32 i = 0; i < MAX_CONTROLLERS; i++)
		{
			reset_controller_state(input.controllers[i]);
		}

		for (u32 i = 0; i < MAX_JOYSTICKS; i++)
		{
			reset_joystick_state(input.joysticks[i]);
		}
	}
}