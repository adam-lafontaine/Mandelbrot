#pragma once
#include "app_input.hpp"

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

	T* row_begin(u64 y) const
	{
		assert(y < height);

		auto offset = y * width;

		auto ptr = data + (u64)(offset);
		assert(ptr);

		return ptr;
	}

	T* begin() { return data; }
	T* end() { return data + (u64)(width) * (u64)(height); }
	T* begin() const { return data; }
	T* end() const { return data + (u64)(width) * (u64)(height); }
};


using Mat2Du32 = Matrix<u32>;
using Mat2Di32 = Matrix<i32>;
using Image = Matrix<Pixel>;


class ChannelOptions
{
public:
	u32 channel1;
    u32 channel2;
    u32 channel3;
};


class AppState
{
public:
	AppInput app_input;

	bool ids_prev = 0;
	bool ids_current = 1;

	Mat2Di32 color_ids[2];

	Image screen_buffer;

	ChannelOptions channel_options;

	Range2Du32 copy_src;
    Range2Du32 copy_dst;
	Range2Du32 write_h;
	Range2Du32 write_v; // temp

	r64 min_mx;
	r64 min_my;
	r64 mx_step;
	r64 my_step;
};