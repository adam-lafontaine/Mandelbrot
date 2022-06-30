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


/* KEYBOARD */

// activate keys to accept input from
#define KEYBOARD_A 1
#define KEYBOARD_B 0
#define KEYBOARD_C 0
#define KEYBOARD_D 1
#define KEYBOARD_E 0
#define KEYBOARD_F 0
#define KEYBOARD_G 0
#define KEYBOARD_H 0
#define KEYBOARD_I 0
#define KEYBOARD_J 0
#define KEYBOARD_K 0
#define KEYBOARD_L 0
#define KEYBOARD_M 0
#define KEYBOARD_N 0
#define KEYBOARD_O 0
#define KEYBOARD_P 0
#define KEYBOARD_Q 0
#define KEYBOARD_R 0
#define KEYBOARD_S 1
#define KEYBOARD_T 0
#define KEYBOARD_U 0
#define KEYBOARD_V 0
#define KEYBOARD_W 1
#define KEYBOARD_X 0
#define KEYBOARD_Y 0
#define KEYBOARD_Z 0
#define KEYBOARD_0 0
#define KEYBOARD_1 0
#define KEYBOARD_2 0
#define KEYBOARD_3 0
#define KEYBOARD_4 0
#define KEYBOARD_5 0
#define KEYBOARD_6 0
#define KEYBOARD_7 0
#define KEYBOARD_8 0
#define KEYBOARD_9 0
#define KEYBOARD_UP 1
#define KEYBOARD_DOWN 1
#define KEYBOARD_LEFT 1
#define KEYBOARD_RIGHT 1
#define KEYBOARD_RETURN 0
#define KEYBOARD_ESCAPE 1
#define KEYBOARD_SPACE 1
#define KEYBOARD_SHIFT 1
#define KEYBOARD_PLUS 1
#define KEYBOARD_MINUS 1
#define KEYBOARD_MULTIPLY 1
#define KEYBOARD_DIVIDE 1
#define KEYBOARD_NUMPAD_0 0
#define KEYBOARD_NUMPAD_1 0
#define KEYBOARD_NUMPAD_2 1
#define KEYBOARD_NUMPAD_3 0
#define KEYBOARD_NUMPAD_4 1
#define KEYBOARD_NUMPAD_5 0
#define KEYBOARD_NUMPAD_6 1
#define KEYBOARD_NUMPAD_7 0
#define KEYBOARD_NUMPAD_8 1
#define KEYBOARD_NUMPAD_9 0


constexpr size_t KEYBOARD_KEYS = 
KEYBOARD_A 
+ KEYBOARD_B 
+ KEYBOARD_C
+ KEYBOARD_D
+ KEYBOARD_E
+ KEYBOARD_F
+ KEYBOARD_G
+ KEYBOARD_H
+ KEYBOARD_I
+ KEYBOARD_J
+ KEYBOARD_K
+ KEYBOARD_L
+ KEYBOARD_M
+ KEYBOARD_N
+ KEYBOARD_O
+ KEYBOARD_P
+ KEYBOARD_Q
+ KEYBOARD_R
+ KEYBOARD_S
+ KEYBOARD_T
+ KEYBOARD_U
+ KEYBOARD_V
+ KEYBOARD_W
+ KEYBOARD_X
+ KEYBOARD_Y
+ KEYBOARD_Z
+ KEYBOARD_0
+ KEYBOARD_1
+ KEYBOARD_2
+ KEYBOARD_3
+ KEYBOARD_4
+ KEYBOARD_5
+ KEYBOARD_6
+ KEYBOARD_7
+ KEYBOARD_8
+ KEYBOARD_9
+ KEYBOARD_UP
+ KEYBOARD_DOWN
+ KEYBOARD_LEFT
+ KEYBOARD_RIGHT
+ KEYBOARD_RETURN
+ KEYBOARD_ESCAPE
+ KEYBOARD_SPACE
+ KEYBOARD_PLUS
+ KEYBOARD_MINUS
+ KEYBOARD_MULTIPLY
+ KEYBOARD_DIVIDE
+ KEYBOARD_NUMPAD_0
+ KEYBOARD_NUMPAD_1
+ KEYBOARD_NUMPAD_2
+ KEYBOARD_NUMPAD_3
+ KEYBOARD_NUMPAD_4
+ KEYBOARD_NUMPAD_5
+ KEYBOARD_NUMPAD_6
+ KEYBOARD_NUMPAD_7
+ KEYBOARD_NUMPAD_8
+ KEYBOARD_NUMPAD_9
;


typedef union keyboard_input_t
{
	ButtonState keys[KEYBOARD_KEYS];
	struct
	{

#if KEYBOARD_A
		ButtonState a_key;
#endif
#if KEYBOARD_B
		ButtonState b_key;
#endif
#if KEYBOARD_C
		ButtonState c_key;
#endif
#if KEYBOARD_D
		ButtonState d_key;
#endif
#if KEYBOARD_E
		ButtonState e_key;
#endif
#if KEYBOARD_F
		ButtonState f_key;
#endif
#if KEYBOARD_G
		ButtonState g_key;
#endif
#if KEYBOARD_H
		ButtonState h_key;
#endif
#if KEYBOARD_I
		ButtonState i_key;
#endif
#if KEYBOARD_J
		ButtonState j_key;
#endif
#if KEYBOARD_K
		ButtonState k_key;
#endif
#if KEYBOARD_L
		ButtonState l_key;
#endif
#if KEYBOARD_M
		ButtonState m_key;
#endif
#if KEYBOARD_N
		ButtonState n_key;
#endif
#if KEYBOARD_O
		ButtonState o_key;
#endif
#if KEYBOARD_P
		ButtonState p_key;
#endif
#if KEYBOARD_Q
		ButtonState q_key;
#endif
#if KEYBOARD_R
		ButtonState r_key;
#endif
#if KEYBOARD_S
		ButtonState s_key;
#endif
#if KEYBOARD_T
		ButtonState t_key;
#endif
#if KEYBOARD_U
		ButtonState u_key;
#endif
#if KEYBOARD_V
		ButtonState v_key;
#endif
#if KEYBOARD_W
		ButtonState w_key;
#endif
#if KEYBOARD_X
		ButtonState x_key;
#endif
#if KEYBOARD_Y
		ButtonState y_key;
#endif
#if KEYBOARD_Z
		ButtonState z_key;
#endif
#if KEYBOARD_0
		ButtonState zero_key;
#endif
#if KEYBOARD_1
		ButtonState one_key;
#endif
#if KEYBOARD_2
		ButtonState two_key;
#endif
#if KEYBOARD_3
		ButtonState three_key;
#endif
#if KEYBOARD_4
		ButtonState four_key;
#endif
#if KEYBOARD_5
		ButtonState five_key;
#endif
#if KEYBOARD_6
		ButtonState six_key;
#endif
#if KEYBOARD_7
		ButtonState seven_key;
#endif
#if KEYBOARD_8
		ButtonState eight_key;
#endif
#if KEYBOARD_9
		ButtonState nine_key;
#endif
#if KEYBOARD_UP
		ButtonState up_key;
#endif
#if KEYBOARD_DOWN
		ButtonState down_key;
#endif
#if KEYBOARD_LEFT
		ButtonState left_key;
#endif
#if KEYBOARD_RIGHT
		ButtonState right_key;
#endif
#if KEYBOARD_RETURN
		ButtonState return_key;
#endif
#if KEYBOARD_ESCAPE
		ButtonState escape_key;
#endif
#if KEYBOARD_SPACE
		ButtonState space_key;
#endif
#if KEYBOARD_SHIFT
		ButtonState shift_key;
#endif
#if KEYBOARD_PLUS
		ButtonState plus_key;
#endif
#if KEYBOARD_MINUS
		ButtonState minus_key;
#endif
#if KEYBOARD_MULTIPLY
		ButtonState mult_key;
#endif
#if KEYBOARD_DIVIDE
		ButtonState div_key;
#endif
#if KEYBOARD_NUMPAD_0
		ButtonState np_zero_key;
#endif
#if KEYBOARD_NUMPAD_1
		ButtonState np_one_key;
#endif
#if KEYBOARD_NUMPAD_2
		ButtonState np_two_key;
#endif
#if KEYBOARD_NUMPAD_3
		ButtonState np_three_key;
#endif
#if KEYBOARD_NUMPAD_4
		ButtonState np_four_key;
#endif
#if KEYBOARD_NUMPAD_5
		ButtonState np_five_key;
#endif
#if KEYBOARD_NUMPAD_6
		ButtonState np_six_key;
#endif
#if KEYBOARD_NUMPAD_7
		ButtonState np_seven_key;
#endif
#if KEYBOARD_NUMPAD_8
		ButtonState np_eight_key;
#endif
#if KEYBOARD_NUMPAD_9
		ButtonState np_nine_key;
#endif

	};

} KeyboardInput;


/* MOUSE */

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

	Point2Di32 win_pos;

#endif	

	union
	{
		ButtonState buttons[MOUSE_BUTTONS];
		struct
		{
#if MOUSE_LEFT
			ButtonState button_left;
#endif
#if MOUSE_RIGHT
			ButtonState button_right;
#endif
#if MOUSE_MIDDLE
			ButtonState button_middle;
#endif
#if MOUSE_X1
			ButtonState button_x1;
#endif
#if MOUSE_X2
			ButtonState button_x2;
#endif
		};
	};

} MouseInput;


/* CONTROLLER */

#define CONTROLLER_UP 1
#define CONTROLLER_DOWN 1
#define CONTROLLER_LEFT 1
#define CONTROLLER_RIGHT 1
#define CONTROLLER_START 1
#define CONTROLLER_BACK 1
#define CONTROLLER_LEFT_SHOULDER 1
#define CONTROLLER_RIGHT_SHOULDER 1
#define CONTROLLER_A 1
#define CONTROLLER_B 1
#define CONTROLLER_X 1
#define CONTROLLER_Y 1
#define CONTROLLER_STICK_LEFT 1
#define CONTROLLER_STICK_RIGHT 1
#define CONTROLLER_TRIGGER_LEFT 1
#define CONTROLLER_TRIGGER_RIGHT 1

constexpr size_t CONTROLLER_BUTTONS = 
CONTROLLER_UP +
CONTROLLER_DOWN +
CONTROLLER_LEFT +
CONTROLLER_RIGHT +
CONTROLLER_START +
CONTROLLER_BACK +
CONTROLLER_LEFT_SHOULDER +
CONTROLLER_RIGHT_SHOULDER +
CONTROLLER_A +
CONTROLLER_B +
CONTROLLER_X +
CONTROLLER_Y;


class AxisState
{
public:
    r32 start;
    r32 end;
    
    r32 min;
    r32 max;
};


typedef union controller_input_t
{
    ButtonState buttons[CONTROLLER_BUTTONS];
    struct 
    {
#if CONTROLLER_UP
        ButtonState dpad_up;
#endif
#if CONTROLLER_DOWN
        ButtonState dpad_down;
#endif
#if CONTROLLER_LEFT
        ButtonState dpad_left;
#endif
#if CONTROLLER_RIGHT
        ButtonState dpad_right;
#endif
#if CONTROLLER_START
        ButtonState button_start;
#endif
#if CONTROLLER_BACK
        ButtonState button_back;
#endif
#if CONTROLLER_LEFT_SHOULDER
        ButtonState shoulder_left;
#endif
#if CONTROLLER_RIGHT_SHOULDER
        ButtonState shoulder_right;
#endif
#if CONTROLLER_A
        ButtonState button_a;
#endif
#if CONTROLLER_B
        ButtonState button_b;
#endif
#if CONTROLLER_X
        ButtonState button_x;
#endif
#if CONTROLLER_Y
        ButtonState  button_y;
#endif
#if CONTROLLER_STICK_LEFT
        AxisState stick_left_x;
        AxisState stick_left_y;
#endif
#if CONTROLLER_STICK_RIGHT
        AxisState stick_right_x;
        AxisState stick_right_y;
#endif
#if CONTROLLER_TRIGGER_LEFT
        AxisState trigger_left;
#endif
#if CONTROLLER_TRIGGER_LEFT
        AxisState trigger_right;
#endif
    };    

} ControllerInput;




constexpr u32 MAX_CONTROLLERS = 1;


typedef struct input_t
{
	KeyboardInput keyboard;
	MouseInput mouse;

	ControllerInput controllers[MAX_CONTROLLERS];
	u32 num_controllers;

	r32 dt_frame;

} Input;
