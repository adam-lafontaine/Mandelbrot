// D-Bus not build with -rdynamic...
// sudo killall ibus-daemon



#include "../app/app.hpp"
#include "../utils/stopwatch.hpp"
#include "sdl_include.hpp"

#include <cstdio>
#include <thread>


constexpr int WINDOW_WIDTH = app::BUFFER_WIDTH;
constexpr int WINDOW_HEIGHT = app::BUFFER_HEIGHT;

// control the framerate of the application
constexpr r32 TARGET_FRAMERATE_HZ = 60.0f;
constexpr r32 TARGET_MS_PER_FRAME = 1000.0f / TARGET_FRAMERATE_HZ;

GlobalVariable bool g_running = false;


u32 platform_to_color_32(u8 red, u8 green, u8 blue)
{
    return red << 16 | green << 8 | blue;
}


void platform_signal_stop()
{
    g_running = false;
}


static void allocate_app_memory(app::AppMemory& memory)
{
    memory.permanent_storage_size = Megabytes(256);

    size_t total_size = memory.permanent_storage_size;

    memory.permanent_storage = malloc(total_size);
}


static void destroy_app_memory(app::AppMemory& memory)
{
    if(memory.permanent_storage)
    {
        free(memory.permanent_storage);
    }    
}


static void set_app_screen_buffer(ScreenMemory const& memory, app::ScreenBuffer& app_buffer)
{
    app_buffer.memory = memory.image_data;
    app_buffer.width = memory.image_width;
    app_buffer.height = memory.image_height;
    app_buffer.bytes_per_pixel = SCREEN_BYTES_PER_PIXEL;
}


static void handle_sdl_event(SDL_Event const& event)
{
    switch(event.type)
    {
        case SDL_WINDOWEVENT:
        {
            handle_sdl_window_event(event.window);
        } break;
        case SDL_QUIT:
        {
            printf("SDL_QUIT\n");
            g_running = false;
        } break;
        case SDL_KEYDOWN:
        case SDL_KEYUP:
        {
            auto key_code = event.key.keysym.sym;
            auto alt = event.key.keysym.mod & KMOD_ALT;
            if(key_code == SDLK_F4 && alt)
            {
                printf("ALT F4\n");
                g_running = false;
            }
            else if(key_code == SDLK_ESCAPE)
            {
                printf("ESC\n");
                g_running = false;
            }

        } break;
        
    }
}


void print_controls()
{
    printf("\nKEYBOARD:\n");
    printf("          W, A, S, D : Pan up, left, down, right\n");
    printf(" 8, 4, 2, 6 (numpad) : Pan up, left, down, right\n");
    printf("        '+' (numpad) : Zoom in\n");
    printf("        '-' (numpad) : Zoom out\n");
    printf("        '*' (numpad) : Increase zoom speed\n");
    printf("        '/' (numpad) : Decrease zoom speed\n");
    printf("            Arrow up : Increase resolution\n");
    printf("          Arrow Down : Decrease resolution\n");
    printf("Arrow left and right : Change colors\n");
    printf("                 Esc : Close program\n\n");

    printf("\nCONTROLLER:\n");
    printf("    Right thumbstick : Pan up, left, down, right\n");
    printf("     Left thumbstick : Zoom in and out\n");
    printf("       Right trigger : Increase zoom rate\n");
    printf("        Left trigger : Decrease zoom rate\n");
    printf("            D-pad up : Increase resolution\n");
    printf("          D-pad down : Decrease resolution\n");
    printf("D-pad left and right : Change colors\n");
    printf("           B buttton : Close program\n\n");
}

/*

KEYBOARD:
          W, A, S, D : Pan up, left, down, right
 8, 4, 2, 6 (numpad) : Pan up, left, down, right
        '+' (numpad) : Zoom in
        '-' (numpad) : Zoom out
        '*' (numpad) : Increase zoom speed
        '/' (numpad) : Decrease zoom speed
            Arrow up : Increase resolution
          Arrow Down : Decrease resolution
Arrow left and right : Change colors
                 Esc : Close program

CONTROLLER:
    Right thumbstick : Pan up, left, down, right
     Left thumbstick : Zoom in and out
       Right trigger : Increase zoom rate
        Left trigger : Decrease zoom rate
            D-pad up : Increase resolution
          D-pad down : Decrease resolution
D-pad left and right : Change colors
           B buttton : Close program


*/


int main(int argc, char *argv[])
{
    char WINDOW_TITLE[50] = { 0 };
    snprintf(WINDOW_TITLE, 50, "%s v%s", app::APP_TITLE, app::VERSION);

    printf("\n%s\n", WINDOW_TITLE);
    if(!init_sdl())
    {        
        return EXIT_FAILURE;
    }

    ScreenMemory screen{};
    if(!create_screen_memory(screen, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT))
    {
        return EXIT_FAILURE;
    }

    print_controls();

    app::AppMemory app_memory = {};
    app::ScreenBuffer app_buffer = {};
    Input input[2] = {};
    SDLControllerInput controller_input = {};

    auto const cleanup = [&]()
    {
        close_game_controllers(controller_input, input[0]);
        destroy_screen_memory(screen);
        close_sdl();
        destroy_app_memory(app_memory);
    };

    open_game_controllers(controller_input, input[0]);
    input[1].num_controllers = input[0].num_controllers;
    printf("controllers = %d\n", input[0].num_controllers);

    set_app_screen_buffer(screen, app_buffer);
    
    allocate_app_memory(app_memory);
    if (!app_memory.permanent_storage)
    {
        display_error("Allocating application memory failed");
        cleanup();

        return EXIT_FAILURE;
    }

    if(!app::initialize_memory(app_memory, app_buffer))
    {
        display_error("Initializing application memory failed");
        cleanup();
        return EXIT_FAILURE;
    }

    g_running = true;      
    
    bool in_current = 0;
    bool in_old = 1;
    Stopwatch sw;
    r64 frame_ms_elapsed = TARGET_MS_PER_FRAME;
    char dbg_title[50] = { 0 };
    r64 ms_elapsed = 0.0;
    r64 title_refresh_ms = 500.0;

    app::DebugInfo dbg{};

    auto const wait_for_framerate = [&]()
    {
        frame_ms_elapsed = sw.get_time_milli();

        if(ms_elapsed >= title_refresh_ms)
        {
            ms_elapsed = 0.0;
            #ifndef NDEBUG
            snprintf(dbg_title, 50, "%s (%d)", WINDOW_TITLE, (int)frame_ms_elapsed);
            SDL_SetWindowTitle(screen.window, dbg_title);
            #endif
        }

        auto sleep_ms = (u32)(TARGET_MS_PER_FRAME - frame_ms_elapsed);
        if (frame_ms_elapsed < TARGET_MS_PER_FRAME && sleep_ms > 0)
        { 
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
            while (frame_ms_elapsed < TARGET_MS_PER_FRAME)
            {
                frame_ms_elapsed = sw.get_time_milli();
            }        
        }

        ms_elapsed += frame_ms_elapsed;        

        sw.start();
    };
    
    sw.start();
    while(g_running)
    {
        SDLEventInfo evt{};
        evt.first_in_queue = true;
        evt.has_event = false;

        while (SDL_PollEvent(&evt.event))
        {
            evt.has_event = true;
            handle_sdl_event(evt.event);
            process_keyboard_input(evt, input[in_old].keyboard, input[in_current].keyboard);
            process_mouse_input(evt, input[in_old].mouse, input[in_current].mouse);
            evt.first_in_queue = false;
        }

        if (!evt.has_event)
        {
            process_keyboard_input(evt, input[in_old].keyboard, input[in_current].keyboard);
            process_mouse_input(evt, input[in_old].mouse, input[in_current].mouse);
        }

        process_controller_input(controller_input, input[in_old], input[in_current]);

        // does not miss frames but slows animation
        input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

        // animation speed maintained but frames missed
        //input[in_current].dt_frame = frame_ms_elapsed / 1000.0f; // TODO:        

        app::update_and_render(app_memory, input[in_current], dbg);

        wait_for_framerate();
        render_screen(screen);

        // swap inputs
        in_current = in_old;
        in_old = !in_old;        
    }
    
    app::end_program(app_memory);
    cleanup();

    return EXIT_SUCCESS;
}