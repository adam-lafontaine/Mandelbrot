#pragma once

#include "../utils/types.hpp"

#include <cmath>


class RangeList
{
public:
	Range2Du32 copy_src;
	Range2Du32 copy_dst;
	Range2Du32 mbt_h;
	Range2Du32 mbt_v;
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

	auto copy_right = direction.x > 0;
	auto copy_left = direction.x < 0;
	auto copy_down = direction.y > 0;
	auto copy_up = direction.y < 0;

	auto const n_cols = (u32)(std::abs(direction.x));
	auto const n_rows = (u32)(std::abs(direction.y));

	list.copy_src = full_range;
	list.copy_dst = full_range;
	list.mbt_h = full_range;
	list.mbt_v = full_range;

	if (copy_left)
	{
		list.copy_src.x_begin = n_cols;
		list.copy_dst.x_end -= n_cols;
		list.mbt_v.x_begin = list.copy_dst.x_end;
	}
	else if (copy_right)
	{
		list.copy_dst.x_begin = n_cols;
		list.copy_src.x_end -= n_cols;
		list.mbt_v.x_end = list.copy_dst.x_begin;
	}

	if (copy_up)
	{
		list.copy_src.y_begin = n_rows;
		list.copy_dst.y_end -= n_rows;
		list.mbt_h.y_begin = list.copy_dst.y_end;

		if (copy_left || copy_right)
		{
			list.mbt_v.y_end = list.copy_dst.y_end;
		}
	}
	else if (copy_down)
	{
		list.copy_dst.y_begin = n_rows;
		list.copy_src.y_end -= n_rows;
		list.mbt_h.y_end = list.copy_dst.y_begin;

		if (copy_left || copy_right)
		{
			list.mbt_v.y_begin = list.copy_dst.y_begin;
		}
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


