// D-Bus not build with -rdynamic...
// sudo killall ibus-daemon

#include "../app/app.hpp"
#include "../utils/stopwatch.hpp"
#include "sdl_input.hpp"

#include <cstdio>
#include <thread>


constexpr u32 BYTES_PER_PIXEL = 4;


class BitmapBuffer
{
public:
    u32 bytes_per_pixel = BYTES_PER_PIXEL;

    void* memory;    
    int width;
    int height;

    SDL_Renderer* renderer;
    SDL_Texture* texture;
};


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


static void set_app_screen_buffer(BitmapBuffer const& back_buffer, app::ScreenBuffer& app_buffer)
{
    app_buffer.memory = back_buffer.memory;
    app_buffer.width = back_buffer.width;
    app_buffer.height = back_buffer.height;
    app_buffer.bytes_per_pixel = back_buffer.bytes_per_pixel;
}


static void open_game_controllers(SDLInput& sdl, Input& input)
{
    int num_joysticks = SDL_NumJoysticks();
    int c = 0;
    for(int j = 0; j < num_joysticks; ++j)
    {
        if (!SDL_IsGameController(j))
        {
            continue;
        }

        printf("found a controller\n");

        sdl.controllers[c] = SDL_GameControllerOpen(j);
        auto joystick = SDL_GameControllerGetJoystick(sdl.controllers[c]);
        if(!joystick)
        {
            printf("no joystick\n");
        }

        sdl.rumbles[c] = SDL_HapticOpenFromJoystick(joystick);
        if(!sdl.rumbles[c])
        {
            printf("no rumble from joystick\n");
        }
        else if(SDL_HapticRumbleInit(sdl.rumbles[c]))
        {
            printf("%s\n", SDL_GetError());
            SDL_HapticClose(sdl.rumbles[c]);
            sdl.rumbles[c] = 0;
        }
        else
        {
            printf("found a rumble\n");
        }

        ++c;

        if (c >= MAX_CONTROLLERS)
        {
            break;
        }
    }

    input.num_controllers = c;
}


static void close_game_controllers(SDLInput& sdl, Input const& input)
{
    for(int c = 0; c < input.num_controllers; ++c)
    {
        if(sdl.rumbles[c])
        {
            SDL_HapticClose(sdl.rumbles[c]);
        }
        SDL_GameControllerClose(sdl.controllers[c]);
    }
}


static void resize_offscreen_buffer(BitmapBuffer& buffer, int width, int height)
{ 
    if(width == buffer.width && height == buffer.height)
    {
        return;
    }

    buffer.width = width;
    buffer.height = height;

    if(buffer.texture)
    {
        SDL_DestroyTexture(buffer.texture);
    }

    buffer.texture = SDL_CreateTexture(
        buffer.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height);

    if(!buffer.texture)
    {
        printf("SDL_CreateTexture failed\n%s\n", SDL_GetError());
    }

    if(buffer.memory)
    {
        free(buffer.memory);
    }
    
    buffer.memory = malloc(width * height * buffer.bytes_per_pixel);    
}


static bool init_bitmap_buffer(BitmapBuffer& buffer, SDL_Window* window, int width, int height)
{
    buffer.renderer = SDL_CreateRenderer(window, -1, 0);
    
    if(!buffer.renderer)
    {
        printf("SDL_CreateRenderer failed\n%s\n", SDL_GetError());
        return false;
    }

    resize_offscreen_buffer(buffer, width, height);
    
    if(!buffer.memory)
    {
        printf("Back buffer memory failed\n");
        return false;
    }

    return true;
}


static void destroy_bitmap_buffer(BitmapBuffer& buffer)
{
    if(buffer.texture)
    {
        SDL_DestroyTexture(buffer.texture);
    }

    if(buffer.memory)
    {
        free(buffer.memory);
    }
}


constexpr auto WINDOW_TITLE = app::APP_TITLE;
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


static void end_program(app::AppMemory& memory)
{
    g_running = false;
    app::end_program(memory);
}


static void display_error(const char* msg)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "ERROR", msg, 0);
}


static void display_bitmap_in_window(BitmapBuffer const& buffer)
{
    auto error = SDL_UpdateTexture(buffer.texture, 0, buffer.memory, buffer.width * buffer.bytes_per_pixel);
    if(error)
    {
        printf("%s\n", SDL_GetError());
    }

    SDL_RenderCopy(buffer.renderer, buffer.texture, 0, 0);
    
    SDL_RenderPresent(buffer.renderer);
}


static void handle_sdl_window_event(SDL_WindowEvent const& w_event)
{
    auto window = SDL_GetWindowFromID(w_event.windowID);
    //auto renderer = SDL_GetRenderer(window);

    switch(w_event.event)
    {
        case SDL_WINDOWEVENT_SIZE_CHANGED:
        {
            /*int width, height;
            SDL_GetWindowSize(window, &width, &height);
            resize_offscreen_buffer(g_back_buffer, width, height);
            set_app_pixel_buffer(g_back_buffer, g_app_buffer);*/

        }break;
        case SDL_WINDOWEVENT_EXPOSED:
        {
            
        } break;
    }
}


static void handle_sdl_event(SDL_Event const& event)
{
    switch(event.type)
    {
        case SDL_WINDOWEVENT:
        {
            handle_sdl_window_event(event.window);
        }break;
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


static bool init_sdl()
{
    auto sdl_options = 
        SDL_INIT_VIDEO | 
        SDL_INIT_GAMECONTROLLER | 
        SDL_INIT_HAPTIC;    
    
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


int main(int argc, char *argv[])
{
    printf("\n");
    if(!init_sdl())
    {
        display_error("Init SDL failed");
        return EXIT_FAILURE;
    }

    auto window = SDL_CreateWindow(
                    WINDOW_TITLE,
                    SDL_WINDOWPOS_UNDEFINED,
                    SDL_WINDOWPOS_UNDEFINED,
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT,
                    SDL_WINDOW_RESIZABLE);
    if(!window)
    {
        display_error("SDL_CreateWindow failed");
        return EXIT_FAILURE;
    }

    SDL_SetWindowTitle(window, WINDOW_TITLE);

    app::AppMemory app_memory = {};
    app::ScreenBuffer app_buffer = {};
    BitmapBuffer back_buffer = {};
    Input input[2] = {};
    SDLInput sdl_input = {};

    auto const cleanup = [&]()
    {
        close_game_controllers(sdl_input, input[0]);
        close_sdl();
        destroy_bitmap_buffer(back_buffer);
        destroy_app_memory(app_memory);
    };

    open_game_controllers(sdl_input, input[0]);
    input[1].num_controllers = input[0].num_controllers;
    printf("controllers = %d\n", input[0].num_controllers);

    if(!init_bitmap_buffer(back_buffer, window, WINDOW_WIDTH, WINDOW_HEIGHT))
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
    
    bool in_current = 0;
    bool in_old = 1;
    Stopwatch sw;
    r64 frame_ms_elapsed = TARGET_MS_PER_FRAME;
    char title_buffer[50];
    r64 ms_elapsed = 0.0;
    r64 title_refresh_ms = 500.0;

    app::DebugInfo dbg{};

    auto const wait_for_framerate = [&]()
    {
        frame_ms_elapsed = sw.get_time_milli();

        if(ms_elapsed >= title_refresh_ms)
        {
            ms_elapsed = 0.0;
            //snprintf(title_buffer, 50, "%s (%u | %.1f | %d)", WINDOW_TITLE, dbg.max_iter, dbg.zoom, (int)frame_ms_elapsed);
            snprintf(title_buffer, 50, "%s (%d)", WINDOW_TITLE, (int)frame_ms_elapsed);
            SDL_SetWindowTitle(window, title_buffer);
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
        SDL_Event event;
        bool has_event = SDL_PollEvent(&event);
        if(has_event)
        {            
            handle_sdl_event(event);
        }

        // does not miss frames but slows animation
        input[in_current].dt_frame = TARGET_MS_PER_FRAME / 1000.0f;

        // animation speed maintained but frames missed
        //input[in_current].dt_frame = frame_ms_elapsed / 1000.0f; // TODO:

        process_keyboard_input(has_event, event, input[in_old], input[in_current]);

        process_controller_input(sdl_input, input[in_old], input[in_current]);

        process_mouse_input(has_event, event, input[in_old], input[in_current]);

        app::update_and_render(app_memory, input[in_current], dbg);

        wait_for_framerate();
        display_bitmap_in_window(back_buffer);

        // swap inputs
        in_current = in_old;
        in_old = !in_old;
    }
    
    app::end_program(app_memory);
    cleanup();

    return EXIT_SUCCESS;
}