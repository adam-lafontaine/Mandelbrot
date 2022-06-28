#pragma once
#include "../utils/image.hpp"


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

	Mat2Du32 iterations;
	u32 iter_min;
	u32 iter_max;

	Image screen_buffer;
};