#pragma once

#include "device.hpp"


class DeviceMemory
{
public:
    DeviceBuffer buffer;

    DeviceMatrix iterations;
    DeviceImage pixels;
    DeviceColorPalette palette;

    DeviceArray<u32> min_iters;
    DeviceArray<u32> max_iters;
};


class AppState
{
public:
	bool render_new;
    bool draw_new;

	Point2Dr64 mbt_pos;
    r64 mbt_screen_width;
    r64 mbt_screen_height;

	Vec2Di32 pixel_shift;

	r64 zoom_level;
	r64 zoom_speed;

	u32 rgb_option;
	u32 iter_limit;
    
    DeviceMemory device;

    image_t screen_buffer;
};