#pragma once

#include "../input/input.hpp"

#if defined(_WIN32)
#define SDL_MAIN_HANDLED
#endif

#include <SDL2/SDL.h>

#ifndef NDEBUG
#include <cstdio>
#endif

class SDLInput
{
public:
    SDL_GameController* controllers[MAX_CONTROLLERS];
    SDL_Haptic* rumbles[MAX_CONTROLLERS];
};


void process_controller_input(SDLInput const& sdl, Input const& old_input, Input& new_input);

void process_keyboard_input(bool has_event, SDL_Event const& event, Input const& old_input, Input& new_input);

void process_mouse_input(bool has_event, SDL_Event const& event, Input const& old_input, Input& new_input);


constexpr u32 SCREEN_BYTES_PER_PIXEL = 4;


class WindowMemory
{
public:

    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* texture = nullptr;

    void* image_data;
    int image_width;
    int image_height;
};


void display_error(const char* msg)
{
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "ERROR", msg, 0);
#ifndef NDEBUG
    printf("%s\n%s\n", msg, SDL_GetError());
#endif
}


static bool init_sdl()
{
    auto sdl_options = 
        SDL_INIT_VIDEO | 
        SDL_INIT_GAMECONTROLLER | 
        SDL_INIT_HAPTIC;    
    
    if (SDL_Init(sdl_options) != 0)
    {
        //printf("SDL_Init failed\n%s\n", SDL_GetError());
        return false;
    }

    return true;
}


static void close_sdl()
{
    SDL_Quit();
}


void destroy_window_memory(WindowMemory& memory)
{
    if(memory.window)
    {
        SDL_DestroyWindow(memory.window);
    }

    if(memory.renderer)
    {
        SDL_DestroyRenderer(memory.renderer);
    }

    if(memory.texture)
    {
        SDL_DestroyTexture(memory.texture);
    }

    if(memory.image_data)
    {
        free(memory.image_data);
    }
}


bool create_window_memory(WindowMemory& memory, const char* title, int width, int height)
{
    destroy_window_memory(memory);

    memory.window = SDL_CreateWindow(
        title,
        SDL_WINDOWPOS_UNDEFINED,
        SDL_WINDOWPOS_UNDEFINED,
        width,
        height,
        SDL_WINDOW_RESIZABLE);

    if(!memory.window)
    {
        display_error("SDL_CreateWindow failed");
        return false;
    }

    memory.renderer = SDL_CreateRenderer(memory.window, -1, 0);

    if(!memory.renderer)
    {
        display_error("");
        return false;
    }

    memory.texture =  SDL_CreateTexture(
        memory.renderer,
        SDL_PIXELFORMAT_ARGB8888,
        SDL_TEXTUREACCESS_STREAMING,
        width,
        height);
    
    if(!memory.texture)
    {
        display_error("SDL_CreateTexture failed");
        return false;
    }

    memory.image_data = malloc(SCREEN_BYTES_PER_PIXEL * width * height);

    if(!memory.image_data)
    {
        display_error("Allocating image memory failed");
        return false;
    }

    return true;
}


static void render_window(WindowMemory const& memory)
{
    auto const pitch = memory.image_width * SCREEN_BYTES_PER_PIXEL;
    auto error = SDL_UpdateTexture(memory.texture, 0, memory.image_data, pitch);
    if(error)
    {
        printf("%s\n", SDL_GetError());
    }

    SDL_RenderCopy(memory.renderer, memory.texture, 0, 0);
    
    SDL_RenderPresent(memory.renderer);
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