#pragma once

#include "device.hpp"


class Point2Dr64
{
public:
	r64 x;
	r64 y;
};


class Point2Du32
{
public:
	u32 x;
	u32 y;
};


class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
};


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