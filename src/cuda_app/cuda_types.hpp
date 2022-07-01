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
	u32 n_colors;

	u8* channel1;
	u8* channel2;
	u8* channel3;
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

	bool ids_old = 0;
	bool ids_current = 1;	
    
    DeviceMemory device;
    UnifiedMemory unified;

	MemoryBuffer<i32> device_i32;
	MemoryBuffer<u8> device_u8;
	MemoryBuffer<Pixel> unified_pixel;
};