#include "../../../../libs/output/window.hpp"
#include "../../../../libs/input/input.hpp"

#include "../../app/app.hpp"

#include <emscripten.h>
#include <cstdio>
#include <cstdlib>

namespace mb = memory_buffer;
namespace img = image;
namespace game = game_mbt;


constexpr auto WINDOW_TITLE = game::APP_TITLE;
constexpr f64 TARGET_FRAMERATE_HZ = 60.0;


class EmControllerState
{
public:
    b8 dpad_up = 0;
    b8 dpad_down = 0;
    b8 dpad_left = 0;
    b8 dpad_right = 0;
    b8 btn_a = 0;
    b8 btn_b = 0;
    b8 btn_x = 0;
    b8 btn_y = 0;    
};


union EmState
{
    u32 state = 0;

    struct
    {
        u32 has_console:1;
        u32 has_gamepad:1;
        u32 has_btn_up:1;
        u32 has_btn_down:1;
        u32 has_btn_left:1;
        u32 has_btn_right:1;
        u32 has_btn_a:1;
        u32 has_btn_b:1;
        u32 has_btn_x:1;
        u32 has_btn_y:1;
    };
};


static void read_controller_state(EmControllerState state, input::ControllerInput const& prev, input::ControllerInput& curr)
{
    auto const record_button_state = [](input::ButtonState const& old_state, input::ButtonState& new_state, b32 is_down)
    {
        new_state.pressed = !old_state.is_down && is_down;
		new_state.is_down = is_down;
		new_state.raised = old_state.is_down && !is_down;
    };

    record_button_state(prev.btn_dpad_up, curr.btn_dpad_up, state.dpad_up);
    record_button_state(prev.btn_dpad_down, curr.btn_dpad_down, state.dpad_down);
    record_button_state(prev.btn_dpad_left, curr.btn_dpad_left, state.dpad_left);
    record_button_state(prev.btn_dpad_right, curr.btn_dpad_right, state.dpad_right);
    record_button_state(prev.btn_a, curr.btn_a, state.btn_a);
    record_button_state(prev.btn_b, curr.btn_b, state.btn_b);
    record_button_state(prev.btn_x, curr.btn_x, state.btn_x);
    record_button_state(prev.btn_y, curr.btn_y, state.btn_y);
}


/* static main variables */

enum class RunState : int
{
    Begin,
    Run,
    Error,
    End
};


namespace smn
{
    constexpr int MAIN_ERROR = 1;
    constexpr int MAIN_OK = 0;

    RunState run_state = RunState::Begin;

    window::Window window;
    input::InputArray input;

    EmControllerState em_controller{};

    game::AppState app_state;
}


void end_program()
{
    smn::run_state = RunState::End;
}


static inline bool is_running()
{
    return smn::run_state != RunState::End;
}


int update_em_controller(char btn, int is_down)
{
    b8 btn_down = (is_down > 0) ? 1 : 0;

    int res = btn;

    switch (btn)
    {

    case 'u':
    case 'U':
        smn::em_controller.dpad_up = btn_down;
        break;
    
    case 'd':
    case 'D':
        smn::em_controller.dpad_down = btn_down;
        break;

    case 'l':
    case 'L':
        smn::em_controller.dpad_left = btn_down;
        break;

    case 'r':
    case 'R':
        smn::em_controller.dpad_right = btn_down;
        break;

    case 'a':
    case 'A':
        smn::em_controller.btn_a = btn_down;
        break;

    case 'b':
    case 'B':
        smn::em_controller.btn_b = btn_down;
        break;

    case 'x':
    case 'X':
        smn::em_controller.btn_x = btn_down;
        break;

    case 'y':
    case 'Y':
        smn::em_controller.btn_y = btn_down;
        break;

    default: return -1;
    }

    return res;
}


class InitParams
{
public:
    u32 max_width = 0;
    u32 max_height = 0;

    b32 is_mobile = 0;
};


bool create_window(Vec2Du32 game_dims, InitParams const& params)
{
    auto const min = [](auto a, auto b) { return a < b ? a : b; };

    auto game_w = game_dims.x;
    auto game_h = game_dims.y;

    auto max_w = params.max_width ? params.max_width : game_w;
    auto max_h = params.max_height ? params.max_height : game_h;

    auto scale_w = (f32)max_w / game_w;
    auto scale_h = (f32)max_h / game_h;

    auto scale = min(scale_w, scale_h);

    Vec2Du32 window_dims = {
        (u32)(scale * game_w),
        (u32)(scale * game_h)
    };
    
    return window::create(smn::window, game::APP_TITLE, window_dims, game_dims);
}


img::ImageView make_window_view()
{
    static_assert(window::PIXEL_SIZE == sizeof(img::Pixel));

    img::ImageView view{};
    view.matrix_data_ = (img::Pixel*)smn::window.pixel_buffer;
    view.width = smn::window.width_px;
    view.height = smn::window.height_px;

    return view;
}


static bool main_init(InitParams const& params)
{
    if (!window::init())
    {
        return false;
    }

    if (!input::init(smn::input))
    {
        return false;
    }

    auto w = params.max_width;
    auto h = params.max_height;

    Vec2Du32 dims = { w, h };

    auto result = game::init(smn::app_state, dims);
    if (!result.success)
    {
        printf("Error: game::init()\n");
        return false;
    }

    if (!create_window(result.app_dimensions, params))
    {
        printf("Error: create_window()\n");
        return false;
    }

    if (!game::set_screen_memory(smn::app_state, make_window_view()))
    {
        printf("Error: game::set_screen_memory()\n");
        return false;
    }

    return true;
}


void main_close()
{
    smn::run_state = RunState::End;

    game::close(smn::app_state);
    input::close();
    window::close();
}


static void main_loop()
{
    input::record_input(smn::input);
    read_controller_state(smn::em_controller, smn::input.prev().controller, smn::input.curr().controller);

    game::update(smn::app_state, smn::input.curr());

    window::render(smn::window);

    smn::input.swap();

    if (!is_running())
    {
        emscripten_cancel_main_loop();
    }
}


static void print_controls()
{
    constexpr auto str = 
    " ___________________________________________________\n"
    "|                          | Gamepad  | Keyboard    |\n" 
    "| Gameplay Controls        | (mobile) | (desktop)   |\n" 
    "|__________________________|__________|_____________|\n"
    "| Movement                 | D-Pad    | W A S D     |\n"
    "|                          |          | Arrow keys  |\n"
    "| Zoom                     | A B      | O P         |\n"
    "| Resolution               | X Y      | K L         |\n"
    "| Change colors            |          | Space key   |\n"
    "|___________________________________________________|\n"
    ;

    printf(str);
}


int main(int argc, char* argv[])
{
    InitParams params{};

    auto const get_arg = [&](int idx) 
    { 
        auto arg = std::atoi(argv[idx]);
        return arg > 0 ? (u32)arg : (u32)0;
    };

    switch (argc)
    {
    case 0:
    case 1:        
        break;

    case 2:
        params.is_mobile = get_arg(1);
        break;

    case 3:
        params.max_width = get_arg(1);
        params.max_height = get_arg(2);
        break;

    default:
        params.max_width = get_arg(1);
        params.max_height = get_arg(2);
        params.is_mobile = get_arg(3);
        break;
    }

    printf("\n%s v %s\n\n", game::APP_TITLE, game::VERSION);

    if (!main_init(params))
    {
        return EXIT_FAILURE;
    }

    print_controls();

    smn::run_state = RunState::Run;

    emscripten_set_main_loop(main_loop, 30, 1);
    
    main_close();

    return EXIT_SUCCESS;
}


extern "C"
{
    EMSCRIPTEN_KEEPALIVE
    void kill()
    {
        end_program();
    }


    EMSCRIPTEN_KEEPALIVE
    int gamepad_button(char btn, int is_down)
    {
        return update_em_controller(btn, is_down);
    }
    

    EMSCRIPTEN_KEEPALIVE
    u32 get_state()
    {
        EmState state{};

        state.has_console = 1;
        state.has_gamepad = 1;
        state.has_btn_up = 1;
        state.has_btn_down = 1;
        state.has_btn_left = 1;
        state.has_btn_right = 1;
        state.has_btn_a = 1;
        state.has_btn_b = 1;
        state.has_btn_x = 1;
        state.has_btn_y = 1;

        return state.state;
    }
}


#include "../../app/app.cpp"