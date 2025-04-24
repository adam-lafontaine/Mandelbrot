#pragma once

#include "../util/types.hpp"


namespace window
{
    static constexpr u32 PIXEL_SIZE = 4;


    class Icon64
    {
    public:
        static constexpr u32 width = 64;
        static constexpr u32 height = 64;
        static constexpr u32 bytes_per_pixel = PIXEL_SIZE;

        static constexpr u32 min_data_size = width * height * bytes_per_pixel;

        u8* pixel_data = 0;
    };


    class Window
    {
    public:

        u32* pixel_buffer = 0;
        u32 width_px = 0;
        u32 height_px = 0;

        u64 handle = 0;
    };
}


namespace window
{
    bool init();

    void close();

    bool create(Window& window, cstr title, Vec2Du32 window_size, Vec2Du32 pixel_size, Icon64 const& icon);

    bool create(Window& window, cstr title, Vec2Du32 window_size, Vec2Du32 pixel_size);

    bool create_fullscreen(Window& window, cstr title, Vec2Du32 pixel_size, Icon64 const& icon);

    bool create_fullscreen(Window& window, cstr title, Vec2Du32 pixel_size);

    void destroy(Window& window);

    bool resize_pixel_buffer(Window& window, u32 width, u32 height);

    void render(Window const& window, b32 size_changed = 0);

    void hide_mouse_cursor();

    void show_mouse_cursor();
}