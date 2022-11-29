#include "../app/app.hpp"
#include "../sdl/sdl_include.hpp"

#include <cstdio>
#include <cassert>

#include <emscripten.h>

constexpr auto WINDOW_TITLE = app::APP_TITLE;
constexpr int WINDOW_WIDTH = app::BUFFER_WIDTH;
constexpr int WINDOW_HEIGHT = app::BUFFER_HEIGHT;

// assume 30 FPS
constexpr r32 TARGET_FRAMERATE_HZ = 30.0f;
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


static void allocate_app_memory(app::AppMemory &memory)
{
    memory.permanent_storage_size = Megabytes(256);

    size_t total_size = memory.permanent_storage_size;

    memory.permanent_storage = malloc(total_size);
}


static void destroy_app_memory(app::AppMemory &memory)
{
    if (memory.permanent_storage)
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


static void handle_sdl_event(SDL_Event const &event)
{/*
    switch (event.type)
    {
    case SDL_QUIT:
    {
        printf("SDL_QUIT\n");
        g_running = false;
    }
    break;
    case SDL_KEYDOWN:
    case SDL_KEYUP:
    {
        auto key_code = event.key.keysym.sym;
        auto alt = event.key.keysym.mod & KMOD_ALT;
        if (key_code == SDLK_F4 && alt)
        {
            printf("ALT F4\n");
            g_running = false;
        }
        else if (key_code == SDLK_ESCAPE)
        {
            printf("ESC\n");
            g_running = false;
        }
    }
    break;
    }*/
}


app::AppMemory app_memory = {};
app::ScreenBuffer app_buffer = {};
ScreenMemory screen = {};
Input input[2] = {};
//SDLControllerInput controller_input = {};

bool in_current = 0;
bool in_old = 1;

app::DebugInfo dbg{};


void main_loop()
{
    SDLEventInfo evt{};
    evt.first_in_queue = true;
    evt.has_event = false;

    while (SDL_PollEvent(&evt.event))
    {
        evt.has_event = true;
        //handle_sdl_event(evt.event);
        process_keyboard_input(evt, input[in_old].keyboard, input[in_current].keyboard);
        //process_mouse_input(evt, input[in_old].mouse, input[in_current].mouse);
        evt.first_in_queue = false;
    }

    if (!evt.has_event)
    {
        process_keyboard_input(evt, input[in_old].keyboard, input[in_current].keyboard);
        //process_mouse_input(evt, input[in_old].mouse, input[in_current].mouse);
    }

    input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

    //process_controller_input(controller_input, input[in_old], input[in_current]);

    app::update_and_render(app_memory, input[in_current], dbg);
    
    render_screen(screen);

    // swap inputs
    in_current = in_old;
    in_old = !in_old;

    if (!g_running)
    {
        emscripten_cancel_main_loop();
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
    printf("                 Esc : End program\n\n");
}


int main(int argc, char *argv[])
{
    printf("\n%s v %s\n", app::APP_TITLE, app::VERSION);
    if (!init_sdl())
    {
        return EXIT_FAILURE;
    }

    if(!create_screen_memory(screen, WINDOW_TITLE, WINDOW_WIDTH, WINDOW_HEIGHT))
    {
        return EXIT_FAILURE;
    }

    print_controls();

    auto const cleanup = [&]()
    {
        close_sdl();
        destroy_screen_memory(screen);
        destroy_app_memory(app_memory);
    };
    
    set_app_screen_buffer(screen, app_buffer);

    allocate_app_memory(app_memory);
    if (!app_memory.permanent_storage)
    {
        display_error("Allocating application memory failed");
        cleanup();

        return EXIT_FAILURE;
    }

    g_running = app::initialize_memory(app_memory, app_buffer);

    emscripten_set_main_loop(main_loop, 0, 1);
    
    app::end_program(app_memory);
    cleanup();

    return EXIT_SUCCESS;
}