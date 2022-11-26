#pragma once

#include "../input/input.hpp"

#if defined(_WIN32)
#define SDL_MAIN_HANDLED
#endif

#include <SDL2/SDL.h>

#define PRINT_MESSAGES

#ifdef PRINT_MESSAGES
#include <cstdio>
#endif

class SDLInput
{
public:
    SDL_GameController* controllers[MAX_CONTROLLERS];
    SDL_Haptic* rumbles[MAX_CONTROLLERS];
};


// sdl_input.cpp
void process_controller_input(SDLInput const& sdl, Input const& old_input, Input& new_input);

void process_keyboard_input(bool has_event, SDL_Event const& event, Input const& old_input, Input& new_input);

void process_mouse_input(bool has_event, SDL_Event const& event, Input const& old_input, Input& new_input);


constexpr u32 SCREEN_BYTES_PER_PIXEL = 4;


class ScreenMemory
{
public:

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;

    void* image_data;
    int image_width;
    int image_height;
};


static void print_message(const char* msg)
{
#ifdef PRINT_MESSAGES
    printf("%s\n", msg);
#endif
}


static void print_sdl_error(const char* msg)
{
#ifdef PRINT_MESSAGES
    printf("%s\n%s\n", msg, SDL_GetError());
#endif
}


static void display_error(const char* msg)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "ERROR", msg, 0);

    print_sdl_error(msg);
}


static bool init_sdl(bool web_only = false)
{
    auto sdl_options = 
        SDL_INIT_VIDEO | 
        SDL_INIT_GAMECONTROLLER | 
        SDL_INIT_HAPTIC;
    
    if (SDL_Init(sdl_options) != 0)
    {
        print_sdl_error("SDL_Init failed");
        return false;
    }

    return true;
}


static bool init_sdl_web()
{
    auto sdl_options = 
        SDL_INIT_VIDEO;
    
    if (SDL_Init(sdl_options) != 0)
    {
        print_sdl_error("SDL_Init failed");
        return false;
    }

    return true;
}


static void close_sdl()
{
    SDL_Quit();
}


static void destroy_screen_memory(ScreenMemory& screen)
{
    if(screen.window)
    {
        SDL_DestroyWindow(screen.window);
    }

    if(screen.renderer)
    {
        SDL_DestroyRenderer(screen.renderer);
    }

    if(screen.texture)
    {
        SDL_DestroyTexture(screen.texture);
    }

    if(screen.image_data)
    {
        free(screen.image_data);
    }
}


static bool create_screen_memory(ScreenMemory& screen, const char* title, int width, int height)
{
    destroy_screen_memory(screen);

    screen.window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_RESIZABLE);

    if(!screen.window)
    {
        display_error("SDL_CreateWindow failed");
        return false;
    }

    screen.renderer = SDL_CreateRenderer(screen.window, -1, 0);

    if(!screen.renderer)
    {
        display_error("SDL_CreateRenderer failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.texture =  SDL_CreateTexture(
        screen.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height);
    
    if(!screen.texture)
    {
        display_error("SDL_CreateTexture failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.image_data = malloc(SCREEN_BYTES_PER_PIXEL * width * height);

    if(!screen.image_data)
    {
        display_error("Allocating image memory failed");
        destroy_screen_memory(screen);
        return false;
    }

    screen.image_width = width;
    screen.image_height = height;

    return true;
}


static void render_screen(ScreenMemory const& screen)
{
    auto const pitch = screen.image_width * SCREEN_BYTES_PER_PIXEL;
    auto error = SDL_UpdateTexture(screen.texture, 0, screen.image_data, pitch);
    if(error)
    {
        print_sdl_error("SDL_UpdateTexture failed");
    }

    SDL_RenderCopy(screen.renderer, screen.texture, 0, 0);
    
    SDL_RenderPresent(screen.renderer);
}


static void handle_sdl_window_event(SDL_WindowEvent const& w_event)
{
    auto window = SDL_GetWindowFromID(w_event.windowID);
    //auto renderer = SDL_GetRenderer(window);

    switch(w_event.event)
    {
        case SDL_WINDOWEVENT_SIZE_CHANGED:
        {

        }break;
        case SDL_WINDOWEVENT_EXPOSED:
        {
            
        } break;
    }
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

        print_message("found a controller");

        sdl.controllers[c] = SDL_GameControllerOpen(j);
        auto joystick = SDL_GameControllerGetJoystick(sdl.controllers[c]);
        if(!joystick)
        {
            print_message("no joystick");
        }

        sdl.rumbles[c] = SDL_HapticOpenFromJoystick(joystick);
        if(!sdl.rumbles[c])
        {
            print_message("no rumble from joystick");
        }
        else if(SDL_HapticRumbleInit(sdl.rumbles[c]) != 0)
        {
            print_sdl_error("SDL_HapticRumbleInit failed");
            SDL_HapticClose(sdl.rumbles[c]);
            sdl.rumbles[c] = 0;
        }
        else
        {
            print_message("found a rumble");
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
    for(u32 c = 0; c < input.num_controllers; ++c)
    {
        if(sdl.rumbles[c])
        {
            SDL_HapticClose(sdl.rumbles[c]);
        }
        SDL_GameControllerClose(sdl.controllers[c]);
    }
}