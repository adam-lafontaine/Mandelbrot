#pragma once
#include "../utils/image.hpp"


class Point2Dr64
{
public:
	r64 x;
	r64 y;
};


class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
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

	mat_u32_t iterations;
	u32 iter_min;
	u32 iter_max;

	image_t screen_buffer;
};