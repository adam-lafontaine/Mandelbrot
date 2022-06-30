#pragma once
#include "../utils/types.hpp"

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
	u32 color_count_option;

	u32 iter_limit;

	b32 ids_old = 0;
	b32 ids_current = 1;

	Mat2Di32 color_indeces[2];

	Image screen_buffer;
};