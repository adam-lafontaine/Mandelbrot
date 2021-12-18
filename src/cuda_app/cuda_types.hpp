#pragma once

#include "device.hpp"


class Point2Dr64
{
public:
	r64 x;
	r64 y;
};


class DeviceMemory
{
public:
    DeviceBuffer buffer;

    DeviceMatrix iterations;
    DeviceImage pixels;
    DeviceColorPalette palette;
};


class AppState
{
public:
	bool render_new;

	image_t screen_buffer;

	Point2Dr64 screen_pos;
	Vec2Di32 pixel_shift;

	r64 zoom_level;
	r64 zoom_speed;

	u32 rgb_option;

	u32 max_iter;

    DeviceMemory device;
};