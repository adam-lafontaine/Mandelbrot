#pragma once

#include "../input/input_state.hpp"
#include "../util/numeric.hpp"
#include "sdl_include.hpp"

namespace num = numeric;


/* gamepad device */

namespace sdl
{
    class GamepadDevice
    {
    public:
        u32 n_controllers = 0;

        SDL_GameController* controllers[input::MAX_CONTROLLERS];
        SDL_Haptic* rumbles[input::MAX_CONTROLLERS];

        u32 n_joysticks = 0;

        SDL_Joystick* joysticks[input::MAX_JOYSTICKS];
    };


    static void open_gamepad_device(GamepadDevice& device, input::InputArray& input)
    {        
        int cmax = (int)input::MAX_CONTROLLERS;
        int jmax = (int)input::MAX_JOYSTICKS;

        #ifdef NO_CONTROLLER
        cmax = 0;
        #endif

        #ifdef NO_JOYSTICK
        jmax = 0;
        #endif
        
        int c = 0;
        int j = 0;

        int num_sdl_joysticks = SDL_NumJoysticks();

        for(int i = 0; i < num_sdl_joysticks; ++i)
        {
            if (SDL_IsGameController(i))
            {
                sdl::print_message("found a controller");
                if (c >= cmax)
                {
                    continue;
                }

                device.controllers[c] = SDL_GameControllerOpen(i);
                auto joystick = SDL_GameControllerGetJoystick(device.controllers[c]);
                if(!joystick)
                {
                    sdl::print_message("no joystick");
                }

                device.rumbles[c] = SDL_HapticOpenFromJoystick(joystick);
                if(!device.rumbles[c])
                {
                    sdl::print_message("no rumble from joystick");
                }
                else if(SDL_HapticRumbleInit(device.rumbles[c]) != 0)
                {
                    sdl::print_error("SDL_HapticRumbleInit failed");
                    SDL_HapticClose(device.rumbles[c]);
                    device.rumbles[c] = 0;
                }
                else
                {
                    sdl::print_message("found a rumble");
                }

                ++c;
            }
            else
            {
                sdl::print_message("found a joystick");
                if (j >= jmax)
                {
                    continue;
                }

                auto joystick = SDL_JoystickOpen(j);
                device.joysticks[j] = joystick;

                //auto name = SDL_JoystickName(joystick);
                //auto axes = SDL_JoystickNumAxes(joystick);
                //auto buttons = SDL_JoystickNumButtons(joystick);
                //printf("%s | %d | %d\n", name, axes, buttons);

                ++j;
            }            
        }

        device.n_controllers = c;
        device.n_joysticks = j;

        input.n_controllers = c;
        input.n_joysticks = j;
    }


    static void close_gamepad_device(GamepadDevice& device)
    {
        for(u32 c = 0; c < device.n_controllers; ++c)
        {
            if(device.rumbles[c])
            {
                SDL_HapticClose(device.rumbles[c]);
            }
            SDL_GameControllerClose(device.controllers[c]);
        }

        for (u32 j = 0; j < device.n_joysticks; j++)
        {
            SDL_JoystickClose(device.joysticks[j]);
        }
    }
}


/* static data */

namespace sdl
{
    static GamepadDevice gamepad;
}


/* helpers */

namespace sdl
{
    static i32 map_joystick_id(SDL_JoystickID joystick_id)
    {
        for (u32 i = 0; i < gamepad.n_joysticks; i++)
        {
            if (SDL_JoystickInstanceID(gamepad.joysticks[i]) == joystick_id)
            {
                return i;
            }
        }

        return -1;
    }
    
    
    static f32 normalize_axis_value(Sint16 axis)
    {
        constexpr f32 min = -1.0f;
        constexpr f32 max = 1.0f;

        f32 norm = (f32)axis / 32767;

        return num::abs(norm) < 0.3f ? 0.0f : num::clamp(norm, min, max);
    }


    static void set_unit_vector_state(input::VectorState<i8>& vs, Sint16 x, Sint16 y)
    {
        auto& vec = vs.vec;
        auto& unit = vs.unit_direction;

        vec.x = num::sign_i8(x);
        vec.y = num::sign_i8(y);

        unit.x = (f32)vec.x;
        unit.y = (f32)vec.y;

        constexpr f32 hypot = 1.4142135f;
        constexpr f32 i_hypot = 1.0f / hypot;
        
        auto mag = (x || y) ? 1.0f : 0.0f;
        auto i_mag = (x && y) ? i_hypot : (x || y) ? 1.0f : 0.0f;

        vs.magnitude = mag;
        unit.x *= i_mag;
        unit.y *= i_mag;
    }


    static void set_vector_state(input::VectorState<f32>& vs, Sint16 x, Sint16 y)
    {
        auto& vec = vs.vec;
        auto& unit = vs.unit_direction;

        vec.x = normalize_axis_value(x);
        vec.y = normalize_axis_value(y);

        vs.magnitude = num::magnitude(vec);

        auto mag = vs.magnitude > 0.0f ? vs.magnitude : 1.0f;

        unit.x = vec.x / mag;
        unit.y = vec.y / mag;
    }


    static void handle_window_event(SDL_WindowEvent const& w_event, input::Input& input)
    {
    #ifndef NO_WINDOW

        auto window = SDL_GetWindowFromID(w_event.windowID);

        switch(w_event.event)
        {
            case SDL_WINDOWEVENT_SIZE_CHANGED:
            {
                input.window_size_changed = 1;
            }break;
            case SDL_WINDOWEVENT_EXPOSED:
            {
                
            } break;
        }

    #endif
    }


    static void toggle_fullscreen(SDL_Window* window)
    {
    #ifndef NO_WINDOW

        static bool is_fullscreen = false;

        if (is_fullscreen)
        {
            SDL_SetWindowFullscreen(window, 0);
        }
        else
        {
            SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
        }

        is_fullscreen = !is_fullscreen;

    #endif
    }


    static void handle_sdl_event(SDL_Event const& event, input::Input& input)
    {
        switch(event.type)
        {
        case SDL_WINDOWEVENT:
            handle_window_event(event.window, input);
            break;

        case SDL_QUIT:
            print_message("SDL_QUIT");
            end_program();
            break;

        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            auto key_code = event.key.keysym.sym;
            auto alt = event.key.keysym.mod & KMOD_ALT;

            if (alt)
            {
                switch (key_code)
                {
                case SDLK_F4:
                    print_message("ALT F4");
                    end_program();
                    break;

                #ifndef NDEBUG
                case SDLK_RETURN:
                case SDLK_KP_ENTER:
                    print_message("ALT ENTER");
                    auto window = SDL_GetWindowFromID(event.window.windowID);
                    toggle_fullscreen(window);
                    input.window_size_changed = 1;
                    break;
                #endif
                }
            }

            #ifndef NDEBUG

            else
            {
                switch (key_code)
                {
                case SDLK_ESCAPE:
                    print_message("ESC");
                    end_program();
                    break;

                default:
                    break;
                }
            }

            #endif           

        } break;

        default:
            break;
            
        }
    }
}


/* keyboard */

namespace input
{
#ifndef NO_KEYBOARD

    static void record_keyboard_input(SDL_Keycode key_code, KeyboardInput const& old_keyboard, KeyboardInput& new_keyboard, bool is_down)
    {
        switch (key_code)
        {
        #if KEYBOARD_A
        case SDLK_a:
            record_button_input(old_keyboard.kbd_A, new_keyboard.kbd_A, is_down);
            break;
        #endif
        #if KEYBOARD_B
        case SDLK_b:
            record_button_input(old_keyboard.kbd_B, new_keyboard.kbd_B, is_down);
            break;
        #endif
        #if KEYBOARD_C
        case SDLK_c:
            record_button_input(old_keyboard.kbd_C, new_keyboard.kbd_C, is_down);
            break;
        #endif
        #if KEYBOARD_D
        case SDLK_d:
            record_button_input(old_keyboard.kbd_D, new_keyboard.kbd_D, is_down);
            break;
        #endif
        #if KEYBOARD_E
        case SDLK_e:
            record_button_input(old_keyboard.kbd_E, new_keyboard.kbd_E, is_down);
            break;
        #endif
        #if KEYBOARD_F
        case SDLK_f:
            record_button_input(old_keyboard.kbd_F, new_keyboard.kbd_F, is_down);
            break;
        #endif
        #if KEYBOARD_G
        case SDLK_g:
            record_button_input(old_keyboard.kbd_G, new_keyboard.kbd_G, is_down);
            break;
        #endif
        #if KEYBOARD_H
        case SDLK_h:
            record_button_input(old_keyboard.kbd_H, new_keyboard.kbd_H, is_down);
            break;
        #endif
        #if KEYBOARD_I
        case SDLK_i:
            record_button_input(old_keyboard.kbd_I, new_keyboard.kbd_I, is_down);
            break;
        #endif
        #if KEYBOARD_J
        case SDLK_j:
            record_button_input(old_keyboard.kbd_J, new_keyboard.kbd_J, is_down);
            break;
        #endif
        #if KEYBOARD_K
        case SDLK_k:
            record_button_input(old_keyboard.kbd_K, new_keyboard.kbd_K, is_down);
            break;
        #endif
        #if KEYBOARD_L
        case SDLK_l:
            record_button_input(old_keyboard.kbd_L, new_keyboard.kbd_L, is_down);
            break;
        #endif
        #if KEYBOARD_M
        case SDLK_m:
            record_button_input(old_keyboard.kbd_M, new_keyboard.kbd_M, is_down);
            break;
        #endif
        #if KEYBOARD_N
        case SDLK_n:
            record_button_input(old_keyboard.kbd_N, new_keyboard.kbd_N, is_down);
            break;
        #endif
        #if KEYBOARD_O
        case SDLK_o:
            record_button_input(old_keyboard.kbd_O, new_keyboard.kbd_O, is_down);
            break;
        #endif
        #if KEYBOARD_P
        case SDLK_p:
            record_button_input(old_keyboard.kbd_P, new_keyboard.kbd_P, is_down);
            break;
        #endif
        #if KEYBOARD_Q
        case SDLK_q:
            record_button_input(old_keyboard.kbd_Q, new_keyboard.kbd_Q, is_down);
            break;
        #endif
        #if KEYBOARD_R
        case SDLK_r:
            record_button_input(old_keyboard.kbd_R, new_keyboard.kbd_R, is_down);
            break;
        #endif
        #if KEYBOARD_S
        case SDLK_s:
            record_button_input(old_keyboard.kbd_S, new_keyboard.kbd_S, is_down);
            break;
        #endif
        #if KEYBOARD_T
        case SDLK_t:
            record_button_input(old_keyboard.kbd_T, new_keyboard.kbd_T, is_down);
            break;
        #endif
        #if KEYBOARD_U
        case SDLK_u:
            record_button_input(old_keyboard.kbd_U, new_keyboard.kbd_U, is_down);
            break;
        #endif
        #if KEYBOARD_V
        case SDLK_v:
            record_button_input(old_keyboard.kbd_V, new_keyboard.kbd_V, is_down);
            break;
        #endif
        #if KEYBOARD_W
        case SDLK_w:
            record_button_input(old_keyboard.kbd_W, new_keyboard.kbd_W, is_down);
            break;
        #endif
        #if KEYBOARD_X
        case SDLK_x:
            record_button_input(old_keyboard.kbd_X, new_keyboard.kbd_X, is_down);
            break;
        #endif
        #if KEYBOARD_Y
        case SDLK_y:
            record_button_input(old_keyboard.kbd_Y, new_keyboard.kbd_Y, is_down);
            break;
        #endif
        #if KEYBOARD_Z
        case SDLK_z:
            record_button_input(old_keyboard.kbd_Z, new_keyboard.kbd_Z, is_down);
            break;
        #endif
        #if KEYBOARD_0
        case SDLK_0:
            record_button_input(old_keyboard.kbd_0, new_keyboard.kbd_0, is_down);
            break;
        #endif
        #if KEYBOARD_1
        case SDLK_1:
            record_button_input(old_keyboard.kbd_1, new_keyboard.kbd_1, is_down);
            break;
        #endif
        #if KEYBOARD_2
        case SDLK_2:
            record_button_input(old_keyboard.kbd_2, new_keyboard.kbd_2, is_down);
            break;
        #endif
        #if KEYBOARD_3
        case SDLK_3:
            record_button_input(old_keyboard.kbd_3, new_keyboard.kbd_3, is_down);
            break;
        #endif
        #if KEYBOARD_4
        case SDLK_4:
            record_button_input(old_keyboard.kbd_4, new_keyboard.kbd_4, is_down);
            break;
        #endif
        #if KEYBOARD_5
        case SDLK_5:
            record_button_input(old_keyboard.kbd_5, new_keyboard.kbd_5, is_down);
            break;
        #endif
        #if KEYBOARD_6
        case SDLK_6:
            record_button_input(old_keyboard.kbd_6, new_keyboard.kbd_6, is_down);
            break;
        #endif
        #if KEYBOARD_7
        case SDLK_7:
            record_button_input(old_keyboard.kbd_7, new_keyboard.kbd_7, is_down);
            break;
        #endif
        #if KEYBOARD_8
        case SDLK_8:
            record_button_input(old_keyboard.kbd_8, new_keyboard.kbd_8, is_down);
            break;
        #endif
        #if KEYBOARD_9
        case SDLK_9:
            record_button_input(old_keyboard.kbd_9, new_keyboard.kbd_9, is_down);
            break;
        #endif
        #if KEYBOARD_UP
        case SDLK_UP:
            record_button_input(old_keyboard.kbd_up, new_keyboard.kbd_up, is_down);
            break;
        #endif
        #if KEYBOARD_DOWN
        case SDLK_DOWN:
            record_button_input(old_keyboard.kbd_down, new_keyboard.kbd_down, is_down);
            break;
        #endif
        #if KEYBOARD_LEFT
        case SDLK_LEFT:
            record_button_input(old_keyboard.kbd_left, new_keyboard.kbd_left, is_down);
            break;
        #endif
        #if KEYBOARD_RIGHT
        case SDLK_RIGHT:
            record_button_input(old_keyboard.kbd_right, new_keyboard.kbd_right, is_down);
            break;
        #endif
        #if KEYBOARD_RETURN
        case SDLK_RETURN:
        case SDLK_KP_ENTER:
            record_button_input(old_keyboard.kbd_return, new_keyboard.kbd_return, is_down);
            break;
        #endif
        #if KEYBOARD_ESCAPE
        case SDLK_ESCAPE:
            record_button_input(old_keyboard.kbd_escape, new_keyboard.kbd_escape, is_down);
            break;
        #endif
        #if KEYBOARD_SPACE
        case SDLK_SPACE:
            record_button_input(old_keyboard.kbd_space, new_keyboard.kbd_space, is_down);
            break;
        #endif
        #if KEYBOARD_LSHIFT
        case SDLK_LSHIFT:
            record_button_input(old_keyboard.kbd_left_shift, new_keyboard.kbd_left_shift, is_down);
            break;
        #endif
        #if KEYBOARD_RSHIFT
        case SDLK_RSHIFT:
            record_button_input(old_keyboard.kbd_right_shift, new_keyboard.kbd_right_shift, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_0
        case SDLK_KP_0:
            record_button_input(old_keyboard.npd_0, new_keyboard.npd_0, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_1
        case SDLK_KP_1:
            record_button_input(old_keyboard.npd_1, new_keyboard.npd_1, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_2
        case SDLK_KP_2:
            record_button_input(old_keyboard.npd_2, new_keyboard.npd_2, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_3
        case SDLK_KP_3:
            record_button_input(old_keyboard.npd_3, new_keyboard.npd_3, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_4
        case SDLK_KP_4:
            record_button_input(old_keyboard.npd_4, new_keyboard.npd_4, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_5
        case SDLK_KP_5:
            record_button_input(old_keyboard.npd_5, new_keyboard.npd_5, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_6
        case SDLK_KP_6:
            record_button_input(old_keyboard.npd_6, new_keyboard.npd_6, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_7
        case SDLK_KP_7:
            record_button_input(old_keyboard.npd_7, new_keyboard.npd_7, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_8
        case SDLK_KP_8:
            record_button_input(old_keyboard.npd_8, new_keyboard.npd_8, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_9
        case SDLK_KP_9:
            record_button_input(old_keyboard.npd_9, new_keyboard.npd_9, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_PLUS
        case SDLK_KP_PLUS:
            record_button_input(old_keyboard.npd_plus, new_keyboard.npd_plus, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_MINUS
        case SDLK_KP_MINUS:
            record_button_input(old_keyboard.npd_minus, new_keyboard.npd_minus, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_MULTIPLY
        case SDLK_KP_MULTIPLY:
            record_button_input(old_keyboard.npd_mult, new_keyboard.npd_mult, is_down);
            break;
        #endif
        #if KEYBOARD_NUMPAD_DIVIDE
        case SDLK_KP_DIVIDE:
            record_button_input(old_keyboard.npd_div, new_keyboard.npd_div, is_down);
            break;
        #endif
        #if KEYBOARD_CTRL
        case SDLK_LCTRL:
        case SDLK_RCTRL:
            record_button_input(old_keyboard.kbd_ctrl, new_keyboard.kbd_ctrl, is_down);
            break;
        #endif
        
        default:
            break;
        }

        set_is_active(new_keyboard);
    }

#endif
    void record_keyboard_input(SDL_Event const& event, KeyboardInput const& old_keyboard, KeyboardInput& new_keyboard)
    {
    #ifndef NO_KEYBOARD

        switch (event.type)
        {
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            if (event.key.repeat)
            {
                return;
            }

            bool is_down = event.type == SDL_KEYDOWN; //event.key.state == SDL_PRESSED;

            auto key_code = event.key.keysym.sym;
            record_keyboard_input(key_code, old_keyboard, new_keyboard, is_down);
        } break;
        }

        set_is_active(new_keyboard);

    #endif
    }
}


/* mouse */

namespace input
{
#ifndef NO_MOUSE

    static void record_mouse_button_input(Uint8 button_code, MouseInput const& old_mouse, MouseInput& new_mouse, bool is_down)
    {
        switch(button_code)
        {		
        #if MOUSE_LEFT
            case SDL_BUTTON_LEFT:
            {
                record_button_input(old_mouse.btn_left, new_mouse.btn_left, is_down);
            } break;
        #endif
        #if MOUSE_RIGHT
            case SDL_BUTTON_RIGHT:
            {
                record_button_input(old_mouse.btn_right, new_mouse.btn_right, is_down);
            } break;
        #endif
        #if MOUSE_MIDDLE
            case SDL_BUTTON_MIDDLE:
            {
                record_button_input(old_mouse.btn_middle, new_mouse.btn_middle, is_down);
            } break;
        #endif
        #if MOUSE_X1
            case SDL_BUTTON_X1:
            {
                record_button_input(old_mouse.btn_x1, new_mouse.btn_x1, is_down);
            } break;
        #endif
        #if MOUSE_X2
            case SDL_BUTTON_X2:
            {
                record_button_input(old_mouse.btn_x1, new_mouse.btn_x1, is_down);
            } break;
        #endif
        }
    }


    static void record_mouse_position_input(MouseInput& mouse, SDL_MouseMotionEvent const& motion)
    {
    #if MOUSE_POSITION
        mouse.window_pos.x = motion.x;
        mouse.window_pos.y = motion.y;
    #endif
    }


    static void record_mouse_wheel_input(MouseInput& mouse, SDL_MouseWheelEvent const& wheel)
    {
    #if MOUSE_WHEEL
        mouse.wheel.x = wheel.x;
        mouse.wheel.y = wheel.y;
    #endif
    }

#endif

    static void record_mouse_input(SDL_Event const& event, MouseInput const& old_mouse, MouseInput& new_mouse)
    {
    #ifndef NO_MOUSE

        auto& mouse = new_mouse;

        switch (event.type)
        {
        case SDL_MOUSEBUTTONDOWN:
        case SDL_MOUSEBUTTONUP:
        {
            bool is_down = event.type == SDL_MOUSEBUTTONDOWN;
            auto button_code = event.button.button;

            record_mouse_button_input(button_code, old_mouse, mouse, is_down);
        } break;

        #if MOUSE_POSITION

        case SDL_MOUSEMOTION:
        {
            record_mouse_position_input(mouse, event.motion);
            mouse.is_active = true;
        } break;

        #endif

        #if MOUSE_WHEEL

        case SDL_MOUSEWHEEL:
        {
            record_mouse_wheel_input(mouse, event.wheel);
            mouse.is_active = true;
        } break;
        #endif
        }

        set_is_active(mouse);

    #endif
    }
}


/* joystick */

namespace input
{
#ifndef NO_JOYSTICK

    void record_joystic_button_input(Uint8 btn_id, JoystickInput const& old_jsk, JoystickInput& new_jsk, bool is_down)
    {
        switch (btn_id)
        {
        #if JOYSTICK_BTN_0
        case 0:
            record_button_input(old_jsk.btn_0, new_jsk.btn_0, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_1
        case 1:
            record_button_input(old_jsk.btn_1, new_jsk.btn_1, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_2
        case 2:
            record_button_input(old_jsk.btn_2, new_jsk.btn_2, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_3
        case 3:
            record_button_input(old_jsk.btn_3, new_jsk.btn_3, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_4
        case 4:
            record_button_input(old_jsk.btn_4, new_jsk.btn_4, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_5
        case 5:
            record_button_input(old_jsk.btn_5, new_jsk.btn_5, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_6
        case 6:
            record_button_input(old_jsk.btn_6, new_jsk.btn_6, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_7
        case 7:
            record_button_input(old_jsk.btn_7, new_jsk.btn_7, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_8
        case 8:
            record_button_input(old_jsk.btn_8, new_jsk.btn_8, is_down);
            break;
        #endif
        #if JOYSTICK_BTN_9
        case 9:
            record_button_input(old_jsk.btn_9, new_jsk.btn_9, is_down);
            break;
        #endif
        }
    }


    static void record_joystick_axis_input(Uint8 axis_id, JoystickInput const& old_jsk, JoystickInput& new_jsk, Sint16 value)
    {
        Sint16 x = 0;
        Sint16 y = 0;

        value /= 3200;

        switch (axis_id)
        {
        case 0:
            x = value;
            break;

        case 1:
            y = value;
            break;

        default:
            break;
        }

        sdl::set_unit_vector_state(new_jsk.vec_axis, x, y);

    #if JOYSTICK_BTN_AXIS

        record_button_input(old_jsk.btn_axis_left,  new_jsk.btn_axis_left,  x < 0);
        record_button_input(old_jsk.btn_axis_right, new_jsk.btn_axis_right, x > 0);
        record_button_input(old_jsk.btn_axis_up,    new_jsk.btn_axis_up,    y < 0);
        record_button_input(old_jsk.btn_axis_down,  new_jsk.btn_axis_down,  y > 0);

    #endif
    }

#endif

    static void record_joystick_input(SDL_Event const& event, Input const& pre, Input& cur, u32 n_joysticks)
    {
    #ifndef NO_JOYSTICK

        int id = -1;

        Uint8 btn = 255;
        bool is_down = false;
        
        Uint8 axis = 255;
        Sint16 value = 0;

        switch (event.type)
        {
        case SDL_JOYBUTTONDOWN:
        case SDL_JOYBUTTONUP:
        {
            is_down = event.type == SDL_JOYBUTTONDOWN;
            btn = event.jbutton.button;

            id = sdl::map_joystick_id(event.jbutton.which);
            if (id >= 0)
            {
                record_joystic_button_input(btn, pre.joysticks[id], cur.joysticks[id], is_down);
            }

        } break;

        case SDL_JOYAXISMOTION:
        {
            axis = event.jaxis.axis;
            value = event.jaxis.value;

            id = sdl::map_joystick_id(event.jaxis.which);
            if (id >= 0)
            {
                record_joystick_axis_input(axis, pre.joysticks[id], cur.joysticks[id], value);
            }

        } break;

        default:
            break;
        }

        for (u32 i = 0; i < n_joysticks; i++)
        {
            set_is_active(cur.joysticks[i]);
        }

    #endif
    }


}


/* controller */

namespace input
{ 
#ifndef NO_CONTROLLER

    static f32 get_controller_axis(SDL_GameController* sdl_controller, SDL_GameControllerAxis axis_key)
    {
        auto axis = SDL_GameControllerGetAxis(sdl_controller, axis_key);
        return sdl::normalize_axis_value(axis);
    }


    static void record_controller_button(SDL_GameController* sdl_controller, SDL_GameControllerButton btn_key, ButtonState const& old_btn, ButtonState& new_btn)
    {
        auto is_down = SDL_GameControllerGetButton(sdl_controller, btn_key);
        record_button_input(old_btn, new_btn, is_down);
    }


    static void record_controller_stick_button(SDL_GameController* sdl_controller, SDL_GameControllerButton btn_key, ButtonState const& old_btn, ButtonState& new_btn, VectorState<f32> const& stick)
    {
        // ignore button press if stick is used for direction
        auto is_down = SDL_GameControllerGetButton(sdl_controller, btn_key) && stick.magnitude < 0.3;
        record_button_input(old_btn, new_btn, is_down);
    }


    static void record_controller_button_input(SDL_GameController* sdl_controller, ControllerInput const& old_controller, ControllerInput& new_controller)
    {

    #if CONTROLLER_BTN_DPAD_UP
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_DPAD_UP, old_controller.btn_dpad_up, new_controller.btn_dpad_up);
    #endif
    #if CONTROLLER_BTN_DPAD_DOWN
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_DPAD_DOWN, old_controller.btn_dpad_down, new_controller.btn_dpad_down);
    #endif
    #if CONTROLLER_BTN_DPAD_LEFT
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_DPAD_LEFT, old_controller.btn_dpad_left, new_controller.btn_dpad_left);
    #endif
    #if CONTROLLER_BTN_DPAD_RIGHT
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_DPAD_RIGHT, old_controller.btn_dpad_right, new_controller.btn_dpad_right);
    #endif
    #if CONTROLLER_BTN_START
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_START, old_controller.btn_start, new_controller.btn_start);
    #endif
    #if CONTROLLER_BTN_BACK
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_BACK, old_controller.btn_back, new_controller.btn_back);
    #endif
    #if CONTROLLER_BTN_A
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_A, old_controller.btn_a, new_controller.btn_a);
    #endif
    #if CONTROLLER_BTN_B
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_B, old_controller.btn_b, new_controller.btn_b);
    #endif
    #if CONTROLLER_BTN_X
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_X, old_controller.btn_x, new_controller.btn_x);
    #endif
    #if CONTROLLER_BTN_Y
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_Y, old_controller.btn_y, new_controller.btn_y);
    #endif
    #if CONTROLLER_BTN_SHOULDER_LEFT
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_LEFTSHOULDER, old_controller.btn_shoulder_left, new_controller.btn_shoulder_left);
    #endif
    #if CONTROLLER_BTN_SHOULDER_RIGHT
        record_controller_button(sdl_controller, SDL_CONTROLLER_BUTTON_RIGHTSHOULDER, old_controller.btn_shoulder_right, new_controller.btn_shoulder_right);
    #endif
    #if CONTROLLER_BTN_STICK_LEFT
        record_controller_stick_button(sdl_controller, SDL_CONTROLLER_BUTTON_LEFTSTICK, old_controller.btn_stick_left, new_controller.btn_stick_left, new_controller.stick_left);
    #endif
    #if CONTROLLER_BTN_STICK_RIGHT
        record_controller_stick_button(sdl_controller, SDL_CONTROLLER_BUTTON_RIGHTSTICK, old_controller.btn_stick_right, new_controller.btn_stick_right, new_controller.stick_right);        
    #endif

    }


    static void record_controller_dpad_vector(ControllerInput& controller)
    {
    #if CONTROLLER_BTN_DPAD_ALL

        Sint16 x = (Sint16)controller.btn_dpad_right.is_down - (Sint16)controller.btn_dpad_left.is_down;
        Sint16 y = (Sint16)controller.btn_dpad_down.is_down - (Sint16)controller.btn_dpad_up.is_down;

        sdl::set_unit_vector_state(controller.vec_dpad, x, y);

    #endif
    }


    static void record_controller_trigger_input(SDL_GameController* sdl_controller, ControllerInput& controller)
    {

    #if CONTROLLER_TRIGGER_LEFT
        controller.trigger_left = get_controller_axis(sdl_controller, SDL_CONTROLLER_AXIS_TRIGGERLEFT);
    #endif
    #if CONTROLLER_TRIGGER_RIGHT
        controller.trigger_right = get_controller_axis(sdl_controller, SDL_CONTROLLER_AXIS_TRIGGERRIGHT);
    #endif

    }


    static void record_controller_axis_input(SDL_GameController* sdl_controller, ControllerInput& controller)
    {
        Sint16 x = 0;
        Sint16 y = 0;

    #if CONTROLLER_AXIS_STICK_LEFT

        x = SDL_GameControllerGetAxis(sdl_controller, SDL_CONTROLLER_AXIS_LEFTX);
        y = SDL_GameControllerGetAxis(sdl_controller, SDL_CONTROLLER_AXIS_LEFTY);

        sdl::set_vector_state(controller.stick_left, x, y);

    #endif
    #if CONTROLLER_AXIS_STICK_RIGHT
        
        x = SDL_GameControllerGetAxis(sdl_controller, SDL_CONTROLLER_AXIS_RIGHTX);
        y = SDL_GameControllerGetAxis(sdl_controller, SDL_CONTROLLER_AXIS_RIGHTY);

        sdl::set_vector_state(controller.stick_right, x, y);
        
    #endif
    }


    static void record_gamepad_controller_input(SDL_GameController* sdl_controller, ControllerInput const& old_controller, ControllerInput& new_controller)
    {
        if(!sdl_controller || !SDL_GameControllerGetAttached(sdl_controller))
        {
            return;
        }

        record_controller_button_input(sdl_controller, old_controller, new_controller);
        record_controller_dpad_vector(new_controller);        
        record_controller_trigger_input(sdl_controller, new_controller);
        record_controller_axis_input(sdl_controller, new_controller);        
    }

#endif


    static void record_controller_input(Input const& pre, Input& cur, u32 n_controllers)
    {
    #ifndef NO_CONTROLLER

        for (u32 i = 0; i < n_controllers; i++)
        {
            record_gamepad_controller_input(sdl::gamepad.controllers[i], pre.controllers[i], cur.controllers[i]);
            set_is_active(cur.controllers[i]);
        }

    #endif
    }
}


/* api */

namespace input
{
    static constexpr Uint32 subsystem_flags()
    {
        Uint32 flags = SDL_INIT_GAMECONTROLLER;
        flags |= SDL_INIT_JOYSTICK;

    #ifndef NO_HAPTIC
        flags |= SDL_INIT_HAPTIC;
    #endif

        return flags;
    }


    bool init(InputArray& input)
    {
        auto error = SDL_InitSubSystem(subsystem_flags());
        if (error)
        {
            sdl::print_error("Init Input failed");
            return false;
        }

        reset_input_state(input.pre());
        reset_input_state(input.cur());

        sdl::open_gamepad_device(sdl::gamepad, input);

        return true;
    }


    void close()
    {
        sdl::close_gamepad_device(sdl::gamepad);
        SDL_QuitSubSystem(subsystem_flags());
    }


    void record_input(InputArray& input)
    {
        auto& pre = input.pre();
        auto& cur = input.cur();

        copy_input_state(pre, cur);
        cur.frame = pre.frame + 1;
        cur.dt_frame = 1.0f / 60.0f; // TODO
        cur.window_size_changed = 0;

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            sdl::handle_sdl_event(event, cur);
            record_keyboard_input(event, pre.keyboard, cur.keyboard);
            record_mouse_input(event, pre.mouse, cur.mouse);
            record_joystick_input(event, pre, cur, input.n_joysticks);
        }

        record_controller_input(pre, cur, input.n_controllers);
    }


    void record_input(InputArray& input, event_cb handle_event)
    {
        auto& pre = input.pre();
        auto& cur = input.cur();

        copy_input_state(pre, cur);
        cur.frame = pre.frame + 1;
        cur.dt_frame = 1.0f / 60.0f; // TODO
        cur.window_size_changed = 0;

        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            handle_event(&event);
            sdl::handle_sdl_event(event, cur);
            record_keyboard_input(event, pre.keyboard, cur.keyboard);
            record_mouse_input(event, pre.mouse, cur.mouse);
            record_joystick_input(event, pre, cur, input.n_joysticks);
        }

        record_controller_input(pre, cur, input.n_controllers);
    }
}