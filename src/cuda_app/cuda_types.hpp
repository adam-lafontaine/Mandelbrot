#pragma once

#include "device.hpp"
#include "../app/app_input.hpp"

#include <cassert>

constexpr u32 SCREEN_HEIGHT_PX = 800;
constexpr u32 SCREEN_WIDTH_PX = SCREEN_HEIGHT_PX * 9 / 8;


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


class DeviceState
{
public:
    Mat2Di32 color_ids[2];

	ColorPalette color_palette;
    
	Image screen_pixels;
};


class UnifiedState
{
public:
    //Image screen_buffer;

	ChannelOptions channel_options;

	Range2Du32 copy_src;
    Range2Du32 copy_dst;

	bool prev_id = 0;
	bool current_id = 1;

	r64 min_mx;
	r64 min_my;
	r64 mx_step;
	r64 my_step;

    u32 iter_limit;
};


class AppState
{
public:
	AppInput app_input;

	Image device_pixels;
    Image screen_pixels;
    
    MemoryBuffer<DeviceState> device;
	MemoryBuffer<Pixel> device_pixel_buffer;
	MemoryBuffer<i32> device_i32_buffer;
	MemoryBuffer<u8> device_u8_buffer;

    MemoryBuffer<UnifiedState> unified;
	
};