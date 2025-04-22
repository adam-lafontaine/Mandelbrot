#pragma once

#include "framework.h"
#include "../../MandelbrotWin32/resource.h"

#include "../input/input.hpp"


namespace win32
{
    typedef struct memory_state_t
    {
        size_t total_size;
        void* memory_block;

    } MemoryState;


    typedef struct window_dims_t
    {
        int width;
        int height;

    } WindowDims;


    typedef struct bitmap_buffer_t
    {
        unsigned bytes_per_pixel = 4;
        BITMAPINFO info;
        void* memory;
        int width;
        int height;

    } BitmapBuffer;


    // win32_input.cpp
    void process_keyboard_input(KeyboardInput const& old_input, KeyboardInput& new_input);

    // win32_input.cpp
    void process_mouse_input(HWND window, MouseInput const& old_input, MouseInput& new_input);

    // win_32_main.cpp
    bool handle_alt_key_down(MSG const& msg);


}
