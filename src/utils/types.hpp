#pragma once

#include "defines.hpp"


template <typename T>
class Vec2D
{
public:
	T x;
	T y;
};


class Range2Du32
{
public:
	u32 x_begin = 0;
	u32 x_end = 0;
	u32 y_begin = 0;
	u32 y_end = 0;
};

using Vec2Di32 = Vec2D<i32>;
using Vec2Du32 = Vec2D<u32>;
using Vec2Dr32 = Vec2D<r32>;
using Vec2Dr64 = Vec2D<r32>;

using Point2Di32 = Vec2Di32;
using Point2Du32 = Vec2Du32;
using Point2Dr32 = Vec2Dr32;
using Point2Dr64 = Vec2Dr64;
