#pragma once

#include "device.hpp"


constexpr auto RGB_CHANNELS = 3u;
constexpr auto RGBA_CHANNELS = 4u;


typedef union Pixel
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

} pixel_t;


class Image
{
public:
	using pixel_t = Pixel;

	u32 width;
	u32 height;

	pixel_t* data;
};


template <typename T>
class Matrix
{
public:
	u32 width;
	u32 height;

	T* data;
};


using image_t = Image;
using pixel_t = image_t::pixel_t;

using mat_u32_t = Matrix<u32>;


class Point2Dr64
{
public:
	r64 x;
	r64 y;
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
	mat_u32_t iterations;

    DeviceBuffer device_buffer;
    DeviceArray<u32> device_iterations;
    DeviceArray<pixel_t> device_pixels;
};