#include "../app/app.hpp"
#include "../sdl/sdl_input.hpp"

#include <cstdio>
#include <cassert>

#include <emscripten.h>

constexpr u32 BYTES_PER_PIXEL = 4; // sizeof(BufferPixel);

constexpr auto WINDOW_TITLE = app::APP_TITLE;
constexpr int WINDOW_WIDTH = app::BUFFER_WIDTH;
constexpr int WINDOW_HEIGHT = app::BUFFER_HEIGHT;

// control the framerate of the application
constexpr r32 TARGET_FRAMERATE_HZ = 30.0f;
constexpr r32 TARGET_MS_PER_FRAME = 1000.0f / TARGET_FRAMERATE_HZ;

GlobalVariable bool g_running = false;


class BitmapBuffer
{
public:
    u32 bytes_per_pixel = BYTES_PER_PIXEL;
    
    int width = 0;
    int height = 0;

    SDL_Window* window = nullptr;
    SDL_Surface *surface = nullptr;
    SDL_Renderer* renderer = nullptr;

    bool surface_locked = false;
};


static void lock_surface(BitmapBuffer& buffer)
{
    assert(buffer.surface);

    if (!buffer.surface_locked && SDL_MUSTLOCK(buffer.surface)) 
    {
        SDL_LockSurface(buffer.surface);
        buffer.surface_locked = true;
    }
}


static void unlock_surface(BitmapBuffer& buffer)
{
    assert(buffer.surface);

    if (buffer.surface_locked && SDL_MUSTLOCK(buffer.surface)) 
    {
        SDL_UnlockSurface(buffer.surface);
        buffer.surface_locked = false;
    }
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

static void set_app_screen_buffer(BitmapBuffer const &back_buffer, app::ScreenBuffer &app_buffer)
{
    // app will write directly to surface memory
    app_buffer.memory = (void*)((back_buffer.surface)->pixels);

    app_buffer.width = back_buffer.width;
    app_buffer.height = back_buffer.height;
    app_buffer.bytes_per_pixel = back_buffer.bytes_per_pixel;
}


static bool init_bitmap_buffer(BitmapBuffer &buffer, int width, int height)
{
    buffer.width = width;
    buffer.height = height;

    auto error = SDL_CreateWindowAndRenderer(width, height, 0, &(buffer.window), &(buffer.renderer));
    if(error)
    {
        printf("SDL_CreateWindowAndRenderer/n");
        return false;
    }

    buffer.surface = SDL_CreateRGBSurface(
        0,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
        BYTES_PER_PIXEL * 8,
        0, 0, 0, 0);

    if (!buffer.surface)
    {
        printf("SDL_CreateRGBSurface failed\n");
        return false;
    }

    lock_surface(buffer);

    return true;
}


static void destroy_bitmap_buffer(BitmapBuffer &buffer)
{
    unlock_surface(buffer);
}


u32 platform_to_color_32(u8 red, u8 green, u8 blue)
{
    return red << 16 | green << 8 | blue;
}


void platform_signal_stop()
{
    g_running = false;
}


static void end_program(app::AppMemory &memory)
{
    g_running = false;
    app::end_program(memory);
}


static void display_error(const char *msg)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "ERROR", msg, 0);
    printf("ERROR: %s\n", msg);
}


static void display_bitmap_in_screen(BitmapBuffer& buffer)
{
    auto texture = SDL_CreateTextureFromSurface(buffer.renderer, buffer.surface);

    unlock_surface(buffer);

    SDL_RenderCopy(buffer.renderer, texture, 0, 0);    
    SDL_RenderPresent(buffer.renderer);
    SDL_DestroyTexture(texture);

    lock_surface(buffer);
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


static bool init_sdl()
{
    auto sdl_options = SDL_INIT_VIDEO;

    if (SDL_Init(sdl_options) != 0)
    {
        printf("SDL_Init failed\n%s\n", SDL_GetError());
        return false;
    }

    return true;
}


static void close_sdl()
{
    SDL_Quit();
}


app::AppMemory app_memory = {};
app::ScreenBuffer app_buffer = {};
BitmapBuffer back_buffer = {};
Input input[2] = {};
SDLInput sdl_input = {};

bool in_current = 0;
bool in_old = 1;

app::DebugInfo dbg{};


void main_loop()
{
    SDL_Event event;
    bool has_event = SDL_PollEvent(&event);
    if (has_event)
    {
        handle_sdl_event(event);
    }

    // does not miss frames but slows animation
    input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

    // animation speed maintained but frames missed
    // input[in_current].dt_frame = frame_ms_elapsed / 1000.0f; // TODO:

    process_keyboard_input(has_event, event, input[in_old], input[in_current]);

    process_controller_input(sdl_input, input[in_old], input[in_current]);

    process_mouse_input(has_event, event, input[in_old], input[in_current]);

    app::update_and_render(app_memory, input[in_current], dbg);

    display_bitmap_in_screen(back_buffer);

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
    printf("\nCONTROLS:\n");
    printf("Pan up, left, down, right with W, A, S, D or 8, 4, 2, 6 (numpad)\n");
    printf("Zoom in with '+' (numpad)\n");
    printf("Zoom out with '-' (numpad)\n");
    printf("Increase zoom rate with '*'\n");
    printf("Decrease zoom rate with '/'\n");
    printf("Increase resolution with up arrow\n");
    printf("Decrease resolution with down arrow\n");
}


int main(int argc, char *argv[])
{
    printf("\n%s v %s\n", app::APP_TITLE, app::VERSION);
    if (!init_sdl())
    {
        display_error("Init SDL failed");
        return EXIT_FAILURE;
    }

    print_controls();

    auto const cleanup = [&]()
    {
        close_sdl();
        destroy_bitmap_buffer(back_buffer);
        destroy_app_memory(app_memory);
    };

    if (!init_bitmap_buffer(back_buffer, WINDOW_WIDTH, WINDOW_HEIGHT))
    {
        display_error("Creating back buffer failed");
        cleanup();

        return EXIT_FAILURE;
    }

    set_app_screen_buffer(back_buffer, app_buffer);

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