#include "../imgui_sdl2_ogl3/imgui_include.hpp"
#include "../../ui/ui.hpp"
#include "../../io_test/app/app.hpp"
#include "../../game_state/game_state.hpp"

#include <thread>


namespace img = image;
namespace iot = game_io_test;
namespace gs = game_state;


enum class RunState : int
{
    Begin,
    Run,
    End
};


namespace
{
    ui_imgui::UIState ui_state{};
    RunState run_state = RunState::Begin;

    input::InputArray inputs;

    EngineState state{};

    constexpr u32 N_TEXTURES = 2;
    ogl_imgui::TextureList<N_TEXTURES> textures;

    img::Image io_test_screen;
    iot::AppState io_test_state;    
    constexpr ogl_imgui::TextureId io_test_texture_id = ogl_imgui::to_texture_id(0);

    img::Image game_screen;
    constexpr ogl_imgui::TextureId game_texture_id = ogl_imgui::to_texture_id(1);
}


static void end_program()
{
    run_state = RunState::End;
}


static bool is_running()
{
    return run_state != RunState::End;
}


static void cap_game_framerate()
{
    constexpr f64 fudge = 0.9;

    state.game_nano = state.game_sw.get_time_nano();

    auto sleep_ns = TARGET_NS_PER_FRAME - state.game_nano;
    if (sleep_ns > 0.0)
    {
        std::this_thread::sleep_for(std::chrono::nanoseconds((i64)(sleep_ns * fudge)));
    }
    state.game_sw.start();
}


static void set_window_icon(SDL_Window* window)
{
#include "../../../res/icon_64.c"

    ui_imgui::set_window_icon(window, icon_64);
}



static void handle_window_event(SDL_Event const& event, SDL_Window* window)
{
    switch (event.type)
    {
    case SDL_QUIT:
        end_program();
        break;

    case SDL_WINDOWEVENT:
    {
        switch (event.window.event)
        {
        case SDL_WINDOWEVENT_SIZE_CHANGED:
        //case SDL_WINDOWEVENT_RESIZED:
            int w, h;
            SDL_GetWindowSize(window, &w, &h);
            glViewport(0, 0, w, h);
            break;

        case SDL_WINDOWEVENT_CLOSE:
            end_program();
            break;
        
        default:
            break;
        }
    } break;

    case SDL_KEYDOWN:
    case SDL_KEYUP:
    {
        auto key_code = event.key.keysym.sym;
        switch (key_code)
        {
            
    #ifndef NDEBUG
        case SDLK_ESCAPE:
            //sdl::print_message("ESC");
            end_program();
            break;
    #endif

        default:
            break;
        }

    } break;
    
    default:
        break;
    }
}


static void handle_imgui_sdl_event(void* event_ptr)
{
    auto event = (SDL_Event*)event_ptr;
    handle_window_event(*event, ui_state.window);
    ImGui_ImplSDL2_ProcessEvent(event);
}


static void process_user_input()
{
    input::record_input(inputs, handle_imgui_sdl_event);
}


static void render_textures()
{
    ogl_imgui::render_texture(textures.get_ogl_texture(io_test_texture_id));
    ogl_imgui::render_texture(textures.get_ogl_texture(game_texture_id));
}


static void render_imgui_frame()
{
    ui_imgui::new_frame();

#ifdef SHOW_IMGUI_DEMO
    ui_imgui::show_imgui_demo(ui_state);
#endif

    auto t = textures.get_imgui_texture(io_test_texture_id);
    auto w = io_test_screen.width;
    auto h = io_test_screen.height;
    ui::io_test_window(t, w, h, state);

    t = textures.get_imgui_texture(game_texture_id);
    w = game_screen.width;
    h = game_screen.height;
    ui::game_window("GAME", t, w, h, state);

    ui::diagnostics_window();
    ui::input_frames_window(state);
    ui::game_control_window(state);

    ui_imgui::render(ui_state);
}


static bool io_test_init()
{
    auto result = iot::init(io_test_state);
    if (!result.success)
    {
        return false;
    }

    auto w = result.screen_dimensions.x;
    auto h = result.screen_dimensions.y;
    if (!w ||! h)
    {
        return false;
    }

    if (!img::create_image(io_test_screen, w, h))
    {
        return false;
    }

    auto& texture = textures.get_ogl_texture(io_test_texture_id);
    ogl_imgui::init_texture(io_test_screen.data_, (int)w, (int)h, texture);

    return iot::set_screen_memory(io_test_state, img::make_view(io_test_screen));
}


static bool game_state_init()
{
    Vec2Du32 dims = { 0 };
    if (!gs::init(dims))
    {
        return false;
    }

    auto w = dims.x;
    auto h = dims.y;
    if (!w ||! h)
    {
        return false;
    }

    if (!img::create_image(game_screen, w, h))
    {
        return false;
    }

    auto& texture = textures.get_ogl_texture(game_texture_id);
    ogl_imgui::init_texture(game_screen.data_, (int)w, (int)h, texture);

    return gs::set_screen_memory(img::make_view(game_screen));
}


static bool main_init()
{
#ifdef NDEBUG
    ui_state.window_title = "Mandelbrot Engine";
#else
    ui_state.window_title = "Mandelbrot Engine (Debug)";
#endif
    
    ui_state.window_width = 1300;
    ui_state.window_height = 800;
    
    if (!ui_imgui::init(ui_state))
    {
        return false;
    }

    set_window_icon(ui_state.window);

    if (!input::init(inputs))
    {
        return false;
    }

    textures = ogl_imgui::create_textures<N_TEXTURES>();

    if (!io_test_init())
    {
        return false;
    }

    if (!game_state_init())
    {
        return false;
    }

    return true;
}


static void main_close()
{
    iot::close(io_test_state);
    img::destroy_image(io_test_screen);

    ui_imgui::close(ui_state);
}


static void game_loop()
{
    state.thread_sw.start();
    state.game_sw.start();
    while(is_running())
    {
        if (state.cmd_reset_game)
        {
            gs::reset();
            state.cmd_reset_game = 0;
        }

        if (state.cmd_toggle_pause)
        {
            state.cmd_toggle_pause = 0;
            state.hard_pause = !state.hard_pause;
        }

        if (!state.hard_pause)
        {
            auto input_copy = inputs.curr();
            gs::update(input_copy);
        }        

        cap_game_framerate();

        state.thread_nano = state.thread_sw.get_time_nano();
        state.thread_sw.start();

        auto ratio = (f32)(state.game_nano / state.thread_nano);
        state.frame_times.add_time(ratio);
    }
}


static void main_loop()
{
    std::thread th(game_loop);

    while(is_running())
    {
        process_user_input();
        auto& input = inputs.curr();

        iot::update(io_test_state, input);        

        render_textures();

        render_imgui_frame();

        inputs.swap();
    }

    th.join();
}


int main()
{
    if (!main_init())
    {
        return 1;
    }

    run_state = RunState::Run;

    main_loop();

    main_close();

    return 0;
}

#include "main_o.cpp"