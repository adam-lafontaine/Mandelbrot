#pragma once

#include "../utils/types.hpp"

typedef union button_state_t
{
	bool states[3];
	struct
	{
		bool pressed;
		bool is_down;
		bool raised;
	};


} ButtonState;
