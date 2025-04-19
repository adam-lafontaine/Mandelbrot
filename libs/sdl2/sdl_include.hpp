#pragma once

#if defined(_WIN32)
#define SDL_MAIN_HANDLED
#endif

#include <SDL2/SDL.h>

#ifndef NDEBUG
#define PRINT_MESSAGES
#endif

#ifdef __EMSCRIPTEN__
#define NO_HAPTIC
#define NO_WINDOW
#ifdef WASM_PRINTF
#define PRINT_MESSAGES
#endif
#endif

#ifdef PRINT_MESSAGES
#include <cstdio>
#endif


namespace sdl
{
    inline void print_message(const char* msg)
    {
    #ifdef PRINT_MESSAGES
        printf("%s\n", msg);
    #endif
    }


    inline void print_error(const char* msg)
    {
    #ifdef PRINT_MESSAGES
        printf("%s\n%s\n", msg, SDL_GetError());
    #endif
    }


    inline void display_message(const char* msg)
    {
    #ifndef NO_WINDOW
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, "Message", msg, 0);
    #endif

        print_error(msg);
    }


    inline void display_error(const char* title)
    {
    #ifndef NO_WINDOW
        SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_INFORMATION, title, SDL_GetError(), 0);
    #endif

        print_error(title);
    }


    inline void set_window_icon(SDL_Window* window, auto const& icon_64)
    {
        // these masks are needed to tell SDL_CreateRGBSurface(From)
        // to assume the data it gets is byte-wise RGB(A) data
        Uint32 rmask, gmask, bmask, amask;
    #if SDL_BYTEORDER == SDL_BIG_ENDIAN
        int shift = (icon_64.bytes_per_pixel == 3) ? 8 : 0;
        rmask = 0xff000000 >> shift;
        gmask = 0x00ff0000 >> shift;
        bmask = 0x0000ff00 >> shift;
        amask = 0x000000ff >> shift;
    #else // little endian, like x86
        rmask = 0x000000ff;
        gmask = 0x0000ff00;
        bmask = 0x00ff0000;
        amask = (icon_64.bytes_per_pixel == 3) ? 0 : 0xff000000;
    #endif

        SDL_Surface* icon = SDL_CreateRGBSurfaceFrom(
            (void*)icon_64.pixel_data,
            icon_64.width,
            icon_64.height,
            icon_64.bytes_per_pixel * 8,
            icon_64.bytes_per_pixel * icon_64.width,
            rmask, gmask, bmask, amask);

        SDL_SetWindowIcon(window, icon);

        SDL_FreeSurface(icon);
    }
}