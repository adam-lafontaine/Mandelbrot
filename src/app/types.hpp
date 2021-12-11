#pragma once
#include "../utils/image.hpp"


class Point2Dr64
{
public:
	r64 x;
	r64 y;
};


class Point2Di32
{
public:
	i32 x;
	i32 y;
};

using Vec2Di32 = Point2Di32;


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

	image_t buffer_image;

	Point2Dr64 screen_pos;
	Vec2Di32 pixel_shift;

	r64 zoom_level;
	r64 zoom_speed;

	u32 rgb_option;

	u32 max_iter;
	mat_u32_t iterations;
};