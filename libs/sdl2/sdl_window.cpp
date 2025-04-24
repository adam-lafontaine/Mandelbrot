#pragma once

#include "../output/window.hpp"
#include "../alloc_type/alloc_type.hpp"
#include "sdl_include.hpp"


/* screen memory */

namespace sdl
{
    class ScreenMemory
    {
    public:
        SDL_Window* window = 0;
        SDL_Renderer* renderer = 0;
        SDL_Texture* texture = 0;

        SDL_Rect render_rect;

        u32 width_px = 0;
        u32 height_px = 0;
    };


    static void destroy_screen_memory(ScreenMemory& screen)
    {
        if (screen.texture)
        {
            SDL_DestroyTexture(screen.texture);
        }

        if (screen.renderer)
        {
            SDL_DestroyRenderer(screen.renderer);
        }

        if(screen.window)
        {
            SDL_DestroyWindow(screen.window);
        }

        SDL_zero(screen);
    }


    static bool create_window(ScreenMemory& screen, cstr title, u32 width, u32 height)
    {
        screen.window = SDL_CreateWindow(
            title,
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            (int)width,
            (int)height,
            SDL_WINDOW_RESIZABLE);

        if(!screen.window)
        {
            display_error("SDL_CreateWindow failed");
            return false;
        }

        return true;
    }


    static bool create_window_fullscreen(ScreenMemory& screen, cstr title)
    {
        screen.window = SDL_CreateWindow(
            title,
            SDL_WINDOWPOS_UNDEFINED,
            SDL_WINDOWPOS_UNDEFINED,
            0,
            0,
            SDL_WINDOW_FULLSCREEN_DESKTOP);

        if(!screen.window)
        {
            display_error("SDL_CreateWindow failed");
            return false;
        }

        return true;
    }


    static bool create_renderer(ScreenMemory& screen)
    {
        screen.renderer = SDL_CreateRenderer(screen.window, -1, 0);

        if(!screen.renderer)
        {
            display_error("SDL_CreateRenderer failed");
            return false;
        }

        return true;
    }


    static bool create_texture(ScreenMemory& screen, u32 width, u32 height)
    {
        static_assert(window::PIXEL_SIZE == 4); // SDL_PIXELFORMAT_ABGR8888

        screen.texture =  SDL_CreateTexture(
            screen.renderer,
            SDL_PIXELFORMAT_ABGR8888,
            SDL_TEXTUREACCESS_STREAMING,
            width,
            height);
        
        if(!screen.texture)
        {
            display_error("SDL_CreateTexture failed");
            return false;
        }

        screen.width_px = width;
        screen.height_px = height;

        return true;
    }


    static bool create_screen_memory(ScreenMemory& screen, cstr title, Vec2Du32 window_size, Vec2Du32 pixel_size)
    {
        destroy_screen_memory(screen);

        if (!create_window(screen, title, window_size.x, window_size.y))
        {
            destroy_screen_memory(screen);
            return false;
        }
        
        if (!create_renderer(screen))
        {
            destroy_screen_memory(screen);
            return false;
        }

        if (!create_texture(screen, pixel_size.x, pixel_size.y))
        {
            destroy_screen_memory(screen);
            return false;
        }       

        return true;
    }


    static bool create_screen_memory_fullscreen(ScreenMemory& screen, cstr title, Vec2Du32 pixel_size)
    {
        destroy_screen_memory(screen);

        if (!create_window_fullscreen(screen, title))
        {
            destroy_screen_memory(screen);
            return false;
        }
        
        if (!create_renderer(screen))
        {
            destroy_screen_memory(screen);
            return false;
        }

        int width = 0;
        int height = 0;

        auto err = SDL_GetRendererOutputSize(screen.renderer, &width, &height);
        if (err)
        {
            destroy_screen_memory(screen);
            return false;
        }

        if (!create_texture(screen, pixel_size.x, pixel_size.y))
        {
            destroy_screen_memory(screen);
            return false;
        }

        return true;
    }
}


/* static data */

namespace sdl
{
    constexpr u32 N_SCREEN_MEMORY = 2;

    static ScreenMemory screen_data[N_SCREEN_MEMORY] = { 0 };
    static u32 screen_data_size = 0;
}


/* sdl helpers */

namespace sdl
{    
    static ScreenMemory* allocate_screen_memory()
    {
        if (screen_data_size >= N_SCREEN_MEMORY)
        {
            return 0;
        }

        auto data = screen_data + screen_data_size;
        ++screen_data_size;

        return data;
    }


    static void set_out_rect(ScreenMemory& screen)
    {
        SDL_SetRenderDrawColor(screen.renderer, 0, 0, 0, 255); // Black background
        SDL_RenderClear(screen.renderer);

        int width;
        int height;
        SDL_GetRendererOutputSize(screen.renderer, &width, &height);

        auto scale_w = (f32)width / screen.width_px;
        auto scale_h = (f32)height / screen.height_px;

        auto scale = num::min(scale_w, scale_h);

        auto& r = screen.render_rect;

        auto w = (int)(scale * screen.width_px);
        auto h = (int)(scale * screen.height_px);

        r.x = (width - w) / 2;
        r.y = (height - h) / 2;
        r.w = w;
        r.h = h;
    }

}


/* window helpers */

namespace window
{    
    static bool create_window_memory(Window& window, cstr title, Vec2Du32 window_size, Vec2Du32 pixel_size)
    {
        auto data = sdl::allocate_screen_memory();
        if (!data)
        {
            return false;
        }

        auto& screen = *data;

        if (!sdl::create_screen_memory(screen, title, window_size, pixel_size))
        {
            return false;
        }

        sdl::set_out_rect(screen);

        window.handle = (u64)data;

        return true;
    }


    static bool create_window_memory_fullscreen(Window& window, cstr title, Vec2Du32 pixel_size)
    {
        auto data = sdl::allocate_screen_memory();
        if (!data)
        {
            return false;
        }

        auto& screen = *data;

        if (!sdl::create_screen_memory_fullscreen(screen, title, pixel_size))
        {
            return false;
        }

        sdl::set_out_rect(screen);

        window.handle = (u64)data;

        return true;
    }


    static sdl::ScreenMemory& get_screen(Window const& window)
    {
        return *(sdl::ScreenMemory*)window.handle;
    }
    

    static void set_window_icon_64(sdl::ScreenMemory& screen, window::Icon64 const& icon_64)
    {
        static_assert(window::PIXEL_SIZE == window::Icon64::bytes_per_pixel);

        sdl::set_window_icon(screen.window, icon_64);
    }
}


/* api */

namespace window
{
    constexpr auto subsystem_flags = SDL_INIT_VIDEO;


    bool init()
    {
        auto error = SDL_InitSubSystem(subsystem_flags);
        if (error)
        {
            sdl::print_error("Init Video failed");
            return false;
        }

        return true;
    }


    void close()
    {
        SDL_QuitSubSystem(subsystem_flags);
    }


    bool create(Window& window, cstr title, Vec2Du32 window_size, Vec2Du32 pixel_size)
    {
        SDL_zero(window);
        
        if (!create_window_memory(window, title, window_size, pixel_size))
        {
            return false;
        }

        auto& screen = get_screen(window);

        auto buffer = mem::alloc<u32>(pixel_size.x * pixel_size.y, "window.pixel_buffer");
        if (!buffer)
        {
            sdl::destroy_screen_memory(screen);
            SDL_zero(window);
            return false;
        }

        window.pixel_buffer = buffer;
        window.width_px = screen.width_px;
        window.height_px = screen.height_px;

        return true;
    }


    bool create(Window& window, cstr title, Vec2Du32 window_size, Vec2Du32 pixel_size, Icon64 const& icon)
    {
        if (!create(window, title, window_size, pixel_size))
        {
            return false;
        }

        auto& screen = get_screen(window);
        set_window_icon_64(screen, icon);

        return true;
    }


    bool create_fullscreen(Window& window, cstr title, Vec2Du32 pixel_size)
    {
        SDL_zero(window);
        
        if (!create_window_memory_fullscreen(window, title, pixel_size))
        {
            return false;
        }

        auto& screen = get_screen(window);

        auto width = screen.width_px;
        auto height = screen.height_px;

        auto buffer = mem::alloc<u32>(pixel_size.x * pixel_size.y, "window.pixel_buffer");
        if (!buffer)
        {
            sdl::destroy_screen_memory(screen);
            SDL_zero(window);
            return false;
        }

        window.pixel_buffer = buffer;
        window.width_px = screen.width_px;
        window.height_px = screen.height_px;

        return true;
    }


    bool create_fullscreen(Window& window, cstr title, Vec2Du32 pixel_size, Icon64 const& icon)
    {
        if (!create_fullscreen(window, title, pixel_size))
        {
            return false;
        }

        auto& screen = get_screen(window);
        set_window_icon_64(screen, icon);

        return true;
    }


    void destroy(Window& window)
    {
        auto& screen = get_screen(window);

        sdl::destroy_screen_memory(screen);
        mem::free(window.pixel_buffer);

        SDL_zero(window);
    }


    bool resize_pixel_buffer(Window& window, u32 width, u32 height)
    {
        auto& screen = get_screen(window);

        if (width == screen.width_px && height == screen.height_px)
        {
            return true;
        }
        
        if (screen.texture)
        {
            SDL_DestroyTexture(screen.texture);
            screen.texture = 0;
        }

        if (!sdl::create_texture(screen, width, height))
        {
            return false;
        }
        
        auto n_pixels = width * height;
        if (screen.width_px * screen.height_px == n_pixels)
        {            
            return true;
        }

        if (window.pixel_buffer)
        {
            mem::free(window.pixel_buffer);
            window.pixel_buffer = 0;
        }        

        auto buffer = mem::alloc<u32>(n_pixels, "window.pixel_buffer");
        if (!buffer)
        {
            return false;
        }

        window.pixel_buffer = buffer;
        window.width_px = screen.width_px;
        window.height_px = screen.height_px;

        return true;
    }


    void render(Window const& window, b32 size_changed)
    {
        auto& screen = get_screen(window);
        int err = 0;

        if (size_changed)
        {
            sdl::set_out_rect(screen);
        }        

        auto pitch = screen.width_px * sizeof(window.pixel_buffer[0]);
        

        #ifdef PRINT_MESSAGES

        err = SDL_UpdateTexture(screen.texture, 0, (void*)window.pixel_buffer, pitch);
        if(err)
        {
            sdl::print_error("SDL_UpdateTexture failed");
        }

        err = SDL_RenderCopy(screen.renderer, screen.texture, 0, &screen.render_rect);
        if(err)
        {
            sdl::print_error("SDL_RenderCopy failed");
        }

        #else

        SDL_UpdateTexture(screen.texture, 0, (void*)window.pixel_buffer, pitch);
        SDL_RenderCopy(screen.renderer, screen.texture, 0, &screen.render_rect);

        #endif
        
        SDL_RenderPresent(screen.renderer);
    }


    void hide_mouse_cursor()
    {
        SDL_ShowCursor(SDL_DISABLE);
    }


    void show_mouse_cursor()
    {
        SDL_ShowCursor(SDL_ENABLE);
    }
}