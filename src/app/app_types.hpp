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


namespace mat
{	
	template <typename T>
	T* row_begin(Matrix<T> const& mat, u32 y)
	{
		assert(y < mat.height);

		auto offset = y * mat.width;

		auto ptr = mat.data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <typename T>
	class View
	{
	public:

		T* matrix_data = nullptr;
		u32 matrix_width = 0;

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;
		u32 y_end = 0;

		u32 width = 0;
		u32 height = 0;
	};


	template <typename T>
	View<T> sub_view(Matrix<T> const& matrix, Range2Du32 const& range)
	{
		assert(matrix.width);
		assert(matrix.height);
		assert(matrix.data);

		View<T> sub_view{};

		sub_view.matrix_data = matrix.data;
		sub_view.matrix_width = matrix.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}
}