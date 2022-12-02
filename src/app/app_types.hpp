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
};


template <typename T>
T* row_begin(Matrix<T> const& mat, u32 y)
{
	assert(y < mat.height);

	auto offset = y * mat.width;

	auto ptr = mat.data + (u64)(offset);
	assert(ptr);

	return ptr;
}


using Mat2Du32 = Matrix<u32>;
using Mat2Di32 = Matrix<i32>;
using Image = Matrix<Pixel>;



class AppState
{
public:
	AppInput app_input;

	bool prev_id = 0;
	bool current_id = 1;

	Mat2Di32 color_ids[2];

	Image screen_buffer;

	ChannelOptions channel_options;

	Range2Du32 copy_src;
    Range2Du32 copy_dst;

	r64 min_mx;
	r64 min_my;
	r64 mx_step;
	r64 my_step;	
};