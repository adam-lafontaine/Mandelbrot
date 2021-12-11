// D-Bus not build with -rdynamic...
// sudo killall ibus-daemon

#include "../app/app.hpp"
#include "../utils/stopwatch.hpp"
#include "sdl_input.hpp"

#include <cstdio>


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
    memory.transient_storage_size = 0; // Gigabytes(1);

    size_t total_size = memory.permanent_storage_size + memory.transient_storage_size;

    memory.permanent_storage = malloc(total_size);
    memory.transient_storage = (u8*)memory.permanent_storage + memory.permanent_storage_size;
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