#pragma once

#include "types.hpp"
#include <cassert>


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

	pixel_t* row_begin(u64 y) const
	{
		auto x = y;
		assert(y < height);

		auto offset = y * width;

		auto ptr = data + static_cast<u64>(offset);
		assert(ptr);

		return ptr;
	}

	pixel_t* begin() { return data; }
	pixel_t* end() { return data + static_cast<u64>(width) * static_cast<u64>(height); }
	pixel_t* begin() const { return data; }
	pixel_t* end() const { return data + static_cast<u64>(width) * static_cast<u64>(height); }
};


class FloatPixel
{
public:
	r32 red;
	r32 green;
	r32 blue;
};


class FloatImage
{
public:
	using pixel_t = FloatPixel;

	u32 width;
	u32 height;

	pixel_t* data;

	pixel_t* row_begin(u64 y) const
	{
		assert(y < height);

		auto offset = y * width;

		auto ptr = data + static_cast<u64>(offset);
		assert(ptr);

		return ptr;
	}

	pixel_t* begin() { return data; }
	pixel_t* end() { return data + static_cast<u64>(width) * static_cast<u64>(height); }
	pixel_t* begin() const { return data; }
	pixel_t* end() const { return data + static_cast<u64>(width) * static_cast<u64>(height); }
};


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

		auto ptr = data + static_cast<u64>(offset);
		assert(ptr);

		return ptr;
	}

	T* begin() { return data; }
	T* end() { return data + static_cast<u64>(width) * static_cast<u64>(height); }
	T* begin() const { return data; }
	T* end() const { return data + static_cast<u64>(width) * static_cast<u64>(height); }
};





using image_t = Image;
using pixel_t = image_t::pixel_t;

//using fimage_t = FloatImage;
//using fpixel_t = fimage_t::pixel_t;

using mat_u32_t = Matrix<u32>;
using mat_r64_t = Matrix<r64>;
