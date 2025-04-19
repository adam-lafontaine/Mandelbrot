#pragma once

//#define NO_MOUSE

// activate buttons to accept input
#define MOUSE_LEFT 1
#define MOUSE_RIGHT 1
#define MOUSE_MIDDLE 1
#define MOUSE_X1 0
#define MOUSE_X2 0

// track mouse position
#define MOUSE_POSITION 1

#define MOUSE_WHEEL 1


namespace input
{
	constexpr unsigned N_MOUSE_BUTTONS =
	MOUSE_LEFT + 
	MOUSE_RIGHT + 
	MOUSE_MIDDLE + 
	MOUSE_X1 + 
	MOUSE_X2;
}
