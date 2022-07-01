#pragma once

#include "device.hpp"

#include <cassert>

constexpr auto RGB_CHANNELS = 3u;
constexpr auto RGBA_CHANNELS = 4u;


typedef union pixel_t
{
	struct
	{
		u8 red;
		u8 green;
		u8 blue;
		u8 alpha;
	};

	u8 channels[RGBA_CHANNELS];

	u32 value;

} Pixel;


template <typename T>
class Matrix
{
public:
	u32 width;
	u32 height;

	T* data;
};


using Mat2Di32 = Matrix<i32>;
using Image = Matrix<Pixel>;


class ColorPalette
{
public:

	u8* channel1;
	u8* channel2;
	u8* channel3;

	u32 n_colors;

	u32 padding;
};



class DeviceMemory
{
public:
    Mat2Di32 color_ids[2];

	ColorPalette color_palette;
    
};


class UnifiedMemory
{
public:
    Image screen_buffer;

	u32 channel1;
    u32 channel2;
    u32 channel3;

	Range2Du32 copy_src;
    Range2Du32 copy_dst;

	bool ids_old = false;
	bool ids_current = true;

	r64 min_mx;
	r64 min_my;
	r64 mx_step;
	r64 my_step;

    u32 iter_limit;
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
    
    MemoryBuffer<DeviceMemory> device;
	MemoryBuffer<i32> device_i32;
	MemoryBuffer<u8> device_u8;

    MemoryBuffer<UnifiedMemory> unified;
	MemoryBuffer<Pixel> unified_pixel;
};