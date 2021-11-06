#pragma once

#include "button_state.hpp"


// activate buttons to accept input
#define MOUSE_LEFT 1
#define MOUSE_RIGHT 1
#define MOUSE_MIDDLE 0
#define MOUSE_X1 0
#define MOUSE_X2 0

// track mouse position
#define MOUSE_POSITION 1


constexpr size_t MOUSE_BUTTONS =
MOUSE_LEFT
+ MOUSE_RIGHT
+ MOUSE_MIDDLE
+ MOUSE_X1
+ MOUSE_X2;



typedef struct mouse_input_t
{
#if MOUSE_POSITION

	r64 mouse_x;
	r64 mouse_y;
	r64 mouse_z;

#endif	

	union
	{
		ButtonState buttons[MOUSE_BUTTONS];
		struct
		{
#if MOUSE_LEFT
			ButtonState left;
#endif
#if MOUSE_RIGHT
			ButtonState right;
#endif
#if MOUSE_MIDDLE
			ButtonState middle;
#endif
#if MOUSE_X1
			ButtonState x1;
#endif
#if MOUSE_X2
			ButtonState x2;
#endif
		};
	};

} MouseInput;