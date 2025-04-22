#include "../../../../libs/output/window.hpp"
#include "../../../../libs/input/input.hpp"
#include "../../../../libs/util/stopwatch.hpp"

#include "../../app/app.hpp"

#include <thread>

namespace game = game_mbt;
namespace img = image;


#ifndef APP_FULLSCREEN

constexpr u32 WINDOW_SCALE = 1;

#endif


constexpr f64 NANO = 1'000'000'000;

constexpr f64 TARGET_FPS = 60.0;
constexpr f64 TARGET_NS_PER_FRAME = NANO / TARGET_FPS;


static void cap_framerate(Stopwatch& sw, f64 target_ns)
{
    constexpr f64 fudge = 0.9;

    auto sleep_ns = target_ns - sw.get_time_nano();
    if (sleep_ns > 0.0)
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds((i64)(sleep_ns * fudge)));
    }

    sw.start();
}


enum class RunState : int
{
    Begin,
    Run,
    Error,
    End
};


namespace mn
{
    constexpr int MAIN_ERROR = 1;
    constexpr int MAIN_OK = 0;

    RunState run_state = RunState::Begin;

    window::Window window;
    input::InputArray input;

    game::AppState app_state;
}


bool create_window(Vec2Du32 game_dims)
{
#include "../../res/icon/icon_mbt_64.cpp"
    window::Icon64 icon{};

    static_assert(sizeof(icon_64.pixel_data) >= icon.min_data_size);

    assert(icon_64.width == icon.width);
    assert(icon_64.height == icon.height);

    icon.pixel_data = (u8*)icon_64.pixel_data;

#ifndef APP_FULLSCREEN

    auto w = game_dims.x * WINDOW_SCALE;
    auto h = game_dims.y * WINDOW_SCALE;

    Vec2Du32 window_dims = {
        game_dims.x * WINDOW_SCALE,
        game_dims.y * WINDOW_SCALE
    };

    if (!window::create(mn::window, game::APP_TITLE, window_dims, game_dims, icon))
    {
        return false;
    }

    window::hide_mouse_cursor();

#else

    if (!window::create_fullscreen(mn::window, game::APP_TITLE, game_dims, icon))
    {
        return false;
    }

#endif

    return true;
}


img::ImageView make_window_view()
{
    static_assert(window::PIXEL_SIZE == sizeof(img::Pixel));

    img::ImageView view{};
    view.matrix_data_ = (img::Pixel*)mn::window.pixel_buffer;
    view.width = mn::window.width_px;
    view.height = mn::window.height_px;

    return view;
}


void end_program()
{
    mn::run_state = RunState::End;
}


static bool is_running()
{
    return mn::run_state == RunState::Run;
}


static bool reset_game(input::Input const& input)
{
    return
        (input.controller.btn_back.is_down && input.controller.btn_start.is_down) ||
        (input.joystick.btn_4.is_down && input.joystick.btn_5.is_down) || // L & R
        (input.keyboard.kbd_ctrl.is_down && input.keyboard.kbd_space.is_down);
}


static bool main_init()
{    
    if (!window::init())
    {
        return false;
    }

    if (!input::init(mn::input))
    {
        return false;
    }

    auto result = game::init(mn::app_state);
    if (!result.success)
    {
        // result.error_code
        return false;
    }

    if (!create_window(result.app_dimensions))
    {
        return false;
    }

    if (!game::set_screen_memory(mn::app_state, make_window_view()))
    {
        return false;
    }

    return true;
}


static void main_close()
{
    mn::run_state = RunState::End;

    window::show_mouse_cursor();

    game::close(mn::app_state);
    input::close();
    window::close();
}


static void main_loop()
{
    Stopwatch sw;
    sw.start();

    b32 window_size_changed = 0;

    while(is_running())
    {
        input::record_input(mn::input);

        if (reset_game(mn::input.cur()))
        {
            game::reset(mn::app_state);
        }

        game::update(mn::app_state, mn::input.cur());

        window::render(mn::window, mn::input.cur().window_size_changed);

        mn::input.swap();
        cap_framerate(sw, TARGET_NS_PER_FRAME);
    }
}


int main()
{
    if (!main_init())
    {
        main_close();
        return mn::MAIN_ERROR;
    }

    mn::run_state = RunState::Run;

    main_loop();

    main_close();

    return mn::MAIN_OK;
}

#include "../../app/app.cpp"