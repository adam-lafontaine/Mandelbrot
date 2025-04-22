#pragma once

#include "../utils/types.hpp"
#include "../input/input.hpp"


class AppInput
{
public:
    bool render_new;
	bool draw_new;
    bool zoom;

	Point2Dr64 mbt_pos;
    r64 mbt_screen_width;
    r64 mbt_screen_height;

	Vec2Di32 pixel_shift;

	r64 zoom_level;
	r64 zoom_speed;

	u32 rgb_option;

	u32 iter_limit;	
};


class ChannelOptions
{
public:
	u32 channel1;
    u32 channel2;
    u32 channel3;
};


void set_rgb_channels(ChannelOptions& options, u32 rgb_option);

void init_app_input(AppInput& state);

void process_input(Input const& input, AppInput& state);


