#include "win32_main.hpp"
#include "../input/input_state.hpp"


namespace win32
{
	void process_mouse_input(HWND window, MouseInput const& old_input, MouseInput& new_input)
	{

#ifdef MOUSE_POSITION

        reset_mouse(new_input);

        POINT mouse_pos;
        GetCursorPos(&mouse_pos);
        ScreenToClient(window, &mouse_pos);

		/*
        RECT client_rect;
        GetClientRect(window, &client_rect);

        int window_width = client_rect.right - client_rect.left;
        int window_height = client_rect.bottom - client_rect.top;
		*/

		new_input.win_pos.x = mouse_pos.x;
		new_input.win_pos.y = mouse_pos.y;

#endif

        auto const button_is_down = [](int btn) { return GetKeyState(btn) & (1u << 15); };


#if MOUSE_LEFT
        record_input(old_input.button_left, new_input.button_left, button_is_down(VK_LBUTTON));
#endif
#if MOUSE_RIGHT
        record_input(old_input.button_right, new_input.button_right, button_is_down(VK_RBUTTON));
#endif
#if MOUSE_MIDDLE
        record_input(old_input.button_middle, new_input.button_middle, button_is_down(VK_MBUTTON));
#endif
#if MOUSE_X1
        record_input(old_input.button_x1, new_input.button_x1, button_is_down(VK_XBUTTON1));
#endif
#if MOUSE_X2
        record_input(old_input.button_x2, new_input.button_x2, button_is_down(VK_XBUTTON2));
#endif
	}



	static void record_keyboard_input(WPARAM wparam, KeyboardInput const& old_input, KeyboardInput& new_input, bool is_down)
	{
		switch (wparam)
		{
#if KEYBOARD_A
		case 'A':
			record_input(old_input.a_key, new_input.a_key, is_down);
			break;
#endif
#if KEYBOARD_B
		case 'B':
			record_input(old_input.b_key, new_input.b_key, is_down);
			break;
#endif
#if KEYBOARD_C
		case 'C':
			record_input(old_input.c_key, new_input.c_key, is_down);
			break;
#endif
#if KEYBOARD_D
		case 'D':
			record_input(old_input.d_key, new_input.d_key, is_down);
			break;
#endif
#if KEYBOARD_E
		case 'E':
			record_input(old_input.e_key, new_input.e_key, is_down);
			break;
#endif
#if KEYBOARD_F
		case 'F':
			record_input(old_input.f_key, new_input.f_key, is_down);
			break;
#endif
#if KEYBOARD_G
		case 'G':
			record_input(old_input.g_key, new_input.g_key, is_down);
			break;
#endif
#if KEYBOARD_H
		case 'H':
			record_input(old_input.h_key, new_input.h_key, is_down);
			break;
#endif
#if KEYBOARD_I
		case 'I':
			record_input(old_input.i_key, new_input.i_key, is_down);
			break;
#endif
#if KEYBOARD_J
		case 'J':
			record_input(old_input.j_key, new_input.j_key, is_down);
			break;
#endif
#if KEYBOARD_K
		case 'K':
			record_input(old_input.k_key, new_input.k_key, is_down);
			break;
#endif
#if KEYBOARD_L
		case 'L':
			record_input(old_input.l_key, new_input.l_key, is_down);
			break;
#endif
#if KEYBOARD_M
		case 'M':
			record_input(old_input.m_key, new_input.m_key, is_down);
			break;
#endif
#if KEYBOARD_N
		case 'N':
			record_input(old_input.n_key, new_input.n_key, is_down);
			break;
#endif
#if KEYBOARD_O
		case 'O':
			record_input(old_input.o_key, new_input.o_key, is_down);
			break;
#endif
#if KEYBOARD_P
		case 'P':
			record_input(old_input.p_key, new_input.p_key, is_down);
			break;
#endif
#if KEYBOARD_Q
		case 'Q':
			record_input(old_input.q_key, new_input.q_key, is_down);
			break;
#endif
#if KEYBOARD_R
		case 'R':
			record_input(old_input.r_key, new_input.r_key, is_down);
			break;
#endif
#if KEYBOARD_S
		case 'S':
			record_input(old_input.s_key, new_input.s_key, is_down);
			break;
#endif
#if KEYBOARD_T
		case 'T':
			record_input(old_input.t_key, new_input.t_key, is_down);
			break;
#endif
#if KEYBOARD_U
		case 'U':
			record_input(old_input.u_key, new_input.u_key, is_down);
			break;
#endif
#if KEYBOARD_V
		case 'V':
			record_input(old_input.v_key, new_input.v_key, is_down);
			break;
#endif
#if KEYBOARD_W
		case 'W':
			record_input(old_input.w_key, new_input.w_key, is_down);
			break;
#endif
#if KEYBOARD_X
		case 'X':
			record_input(old_input.x_key, new_input.x_key, is_down);
			break;
#endif
#if KEYBOARD_Y
		case 'Y':
			record_input(old_input.y_key, new_input.y_key, is_down);
			break;
#endif
#if KEYBOARD_Z
		case 'Z':
			record_input(old_input.z_key, new_input.z_key, is_down);
			break;
#endif
#if KEYBOARD_0
		case '0':
			record_input(old_input.zero_key, new_input.zero_key, is_down);
			break;
#endif
#if KEYBOARD_1
		case '1':
			record_input(old_input.one_key, new_input.one_key, is_down);
			break;
#endif
#if KEYBOARD_2
		case '2':
			record_input(old_input.two_key, new_input.two_key, is_down);
			break;
#endif
#if KEYBOARD_3
		case '3':
			record_input(old_input.three_key, new_input.three_key, is_down);
			break;
#endif
#if KEYBOARD_4
		case '4':
			record_input(old_input.four_key, new_input.four_key, is_down);
			break;
#endif
#if KEYBOARD_5
		case '5':
			record_input(old_input.five_key, new_input.five_key, is_down);
			break;
#endif
#if KEYBOARD_6
		case '6':
			record_input(old_input.six_key, new_input.six_key, is_down);
			break;
#endif
#if KEYBOARD_7
		case '7':
			record_input(old_input.seven_key, new_input.seven_key, is_down);
			break;
#endif
#if KEYBOARD_8
		case '8':
			record_input(old_input.eight_key, new_input.eight_key, is_down);
			break;
#endif
#if KEYBOARD_9
		case '9':
			record_input(old_input.nine_key, new_input.nine_key, is_down);
			break;
#endif
#if KEYBOARD_UP
		case VK_UP:
			record_input(old_input.up_key, new_input.up_key, is_down);
			break;
#endif
#if KEYBOARD_DOWN
		case VK_DOWN:
			record_input(old_input.down_key, new_input.down_key, is_down);
			break;
#endif
#if KEYBOARD_LEFT
		case VK_LEFT:
			record_input(old_input.left_key, new_input.left_key, is_down);
			break;
#endif
#if KEYBOARD_RIGHT
		case VK_RIGHT:
			record_input(old_input.right_key, new_input.right_key, is_down);
			break;
#endif
#if KEYBOARD_RETURN
		case VK_RETURN:
			record_input(old_input.return_key, new_input.return_key, is_down);
			break;
#endif
#if KEYBOARD_ESCAPE
		case VK_ESCAPE:
			record_input(old_input.escape_key, new_input.escape_key, is_down);
			break;
#endif
#if KEYBOARD_SPACE
		case VK_SPACE:
			record_input(old_input.space_key, new_input.space_key, is_down);
			break;
#endif
#if KEYBOARD_SHIFT
		case VK_SHIFT:
			record_input(old_input.shift_key, new_input.shift_key, is_down);
			break;
#endif
#if KEYBOARD_PLUS
		case VK_ADD:
			record_input(old_input.plus_key, new_input.plus_key, is_down);
			break;
#endif

#if KEYBOARD_MINUS
		case VK_SUBTRACT:
			record_input(old_input.minus_key, new_input.minus_key, is_down);
			break;
#endif
#if KEYBOARD_MULTIPLY
		case VK_MULTIPLY:
			record_input(old_input.mult_key, new_input.mult_key, is_down);
			break;
#endif
#if KEYBOARD_DIVIDE
		case VK_DIVIDE:
			record_input(old_input.div_key, new_input.div_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_0
		case VK_NUMPAD0:
			record_input(old_input.np_zero_key, new_input.np_zero_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_1
		case VK_NUMPAD1:
			record_input(old_input.np_one_key, new_input.np_one_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_2
		case VK_NUMPAD2:
			record_input(old_input.np_two_key, new_input.np_two_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_3
		case VK_NUMPAD3:
			record_input(old_input.np_three_key, new_input.np_three_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_4
		case VK_NUMPAD4:
			record_input(old_input.np_four_key, new_input.np_four_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_5
		case VK_NUMPAD5:
			record_input(old_input.np_five_key, new_input.np_five_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_6
		case VK_NUMPAD6:
			record_input(old_input.np_six_key, new_input.np_six_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_7
		case VK_NUMPAD7:
			record_input(old_input.np_seven_key, new_input.np_seven_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_8
		case VK_NUMPAD8:
			record_input(old_input.np_eight_key, new_input.np_eight_key, is_down);
			break;
#endif
#if KEYBOARD_NUMPAD_9
		case VK_NUMPAD9:
			record_input(old_input.np_nine_key, new_input.np_nine_key, is_down);
			break;
#endif
		}
	}


	void process_keyboard_input(KeyboardInput const& old_input, KeyboardInput& new_input)
	{
		copy_keyboard_state(old_input, new_input);

		auto const key_was_down = [](MSG const& msg) { return (msg.lParam & (1u << 30)) != 0; };
		auto const key_is_down = [](MSG const& msg) { return (msg.lParam & (1u << 31)) == 0; };
		

		MSG message;
		bool was_down = false;
		bool is_down = false;

		while (PeekMessage(&message, 0, 0, 0, PM_REMOVE))
		{
			switch (message.message)
			{
			case WM_SYSKEYDOWN:
			case WM_SYSKEYUP:
			case WM_KEYDOWN:
			case WM_KEYUP:
			{
				was_down = key_was_down(message);
				is_down = key_is_down(message);
				if (was_down == is_down)
				{
					break;
				}

				if (is_down && handle_alt_key_down(message))
				{
					return;
				}

				record_keyboard_input(message.wParam, old_input, new_input, is_down);

			} break;

			default:
			{
				TranslateMessage(&message);
				DispatchMessage(&message);
			}
			}
		}
	}
	
}


