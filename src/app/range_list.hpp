#pragma once

#include "../utils/types.hpp"

#include <cmath>

constexpr Range2Du32 ZERO_RANGE = { 0, 0, 0, 0 };


class RangeList
{
public:
	Range2Du32 copy_src = ZERO_RANGE;
	Range2Du32 copy_dst = ZERO_RANGE;
	Range2Du32 mbt_h = ZERO_RANGE;
	Range2Du32 mbt_v = ZERO_RANGE;
};





static inline RangeList get_ranges(Range2Du32 const& full_range, Vec2Di32 const& direction)
{
	RangeList list{};

	auto no_horizontal = direction.x == 0;
	auto no_vertical = direction.y == 0;	

	if (no_horizontal && no_vertical)
	{
		return list;
	}

	auto up = direction.y < 0;
	auto right = direction.x > 0;
	auto left = direction.x < 0;
	auto down = direction.y > 0;

	auto d_up = up && no_horizontal;
	auto d_up_right = up && right;
	auto d_right = no_vertical && right;
	auto d_down_right = down && right;
	auto d_down = down && no_horizontal;
	auto d_down_left = down && left;
	auto d_left = no_vertical && left;
	auto d_up_left = up && left;	

	auto const n_cols = (u32)(std::abs(direction.x));
	auto const n_rows = (u32)(std::abs(direction.y));

	list.copy_src = full_range;
	list.copy_dst = full_range;
	list.mbt_h = full_range;
	list.mbt_v = full_range;

	if (d_up)
	{
		list.copy_src.y_begin = n_rows;
		list.copy_dst.y_end -= n_rows;

		list.mbt_h.y_begin = list.copy_dst.y_end;
		list.mbt_v = ZERO_RANGE;
	}
	else if (d_up_right)
	{
		list.copy_src.y_begin = n_rows;
		list.copy_dst.y_end -= n_rows;
		list.copy_dst.x_begin = n_cols;
		list.copy_src.x_end -= n_cols;

		list.mbt_h.y_begin = list.copy_dst.y_end;
		list.mbt_v.x_end = list.copy_dst.x_begin;
	}
	else if (d_right)
	{
		list.copy_dst.x_begin = n_cols;
		list.copy_src.x_end -= n_cols;

		list.mbt_h = ZERO_RANGE;
		list.mbt_v.x_end = list.copy_dst.x_begin;		
	}
	else if (d_down_right)
	{
		list.copy_dst.x_begin = n_cols;
		list.copy_src.x_end -= n_cols;
		list.copy_dst.y_begin = n_rows;
		list.copy_src.y_end -= n_rows;

		list.mbt_v.x_end = list.copy_dst.x_begin;
		list.mbt_h.y_end = list.copy_dst.y_begin;

	}
	else if (d_down)
	{
		list.copy_dst.y_begin = n_rows;
		list.copy_src.y_end -= n_rows;

		list.mbt_h.y_end = list.copy_dst.y_begin;
		list.mbt_v = ZERO_RANGE;
	}
	else if (d_down_left)
	{
		list.copy_dst.y_begin = n_rows;
		list.copy_src.y_end -= n_rows;
		list.copy_src.x_begin = n_cols;
		list.copy_dst.x_end -= n_cols;

		list.mbt_h.y_end = list.copy_dst.y_begin;
		list.mbt_v.x_begin = list.copy_dst.x_end;
	}
	else if (d_left)
	{
		list.copy_src.x_begin = n_cols;
		list.copy_dst.x_end -= n_cols;

		list.mbt_h = ZERO_RANGE;
		list.mbt_v.x_begin = list.copy_dst.x_end;
	}
	else if (d_up_left)
	{
		list.copy_src.y_begin = n_rows;
		list.copy_dst.y_end -= n_rows;
		list.copy_src.x_begin = n_cols;
		list.copy_dst.x_end -= n_cols;

		list.mbt_h.y_begin = list.copy_dst.y_end;
		list.mbt_v.x_begin = list.copy_dst.x_end;
	}

	return list;
}


static inline Range2Du32 make_range(u32 width, u32 height)
{
	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = width;
	r.y_begin = 0;
	r.y_end = height;

	return r;
}


