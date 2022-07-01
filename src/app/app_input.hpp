#pragma once

#include "../utils/types.hpp"

class ChannelOptions
{
public:
	u32 channel1;
    u32 channel2;
    u32 channel3;
};


class AppInput
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
};


static inline void set_rgb_channels(ChannelOptions& options, u32 rgb_option)
{
	auto& c1 = options.channel1;
    auto& c2 = options.channel2;
    auto& c3 = options.channel3;

	switch (rgb_option)
	{
	case 1:
		c1 = 0;
		c2 = 1;
		c3 = 2;
		break;
	case 2:
		c1 = 0;
		c2 = 2;
		c3 = 1;
		break;
	case 3:
		c1 = 1;
		c2 = 0;
		c3 = 2;
		break;
	case 4:
		c1 = 1;
		c2 = 2;
		c3 = 0;
		break;
	case 5:
		c1 = 2;
		c2 = 0;
		c3 = 1;
		break;
	case 6:
		c1 = 2;
		c2 = 1;
		c3 = 0;
		break;
	}
}