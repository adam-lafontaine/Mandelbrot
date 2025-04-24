#pragma once

#include "defines.hpp"


class Vec2Di32
{
public:
    i32 x;
    i32 y;
};


class Vec2Dr32
{
public:
    r32 x;
    r32 y;
};


class Vec2Du32
{
public:
	u32 x;
	u32 y;
};


class Vec2Dr64
{
public:
	r64 x;
	r64 y;
};


class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
};


using Point2Di32 = Vec2Di32;
using Point2Dr32 = Vec2Di32;
using Point2Dr64 = Vec2Dr64;
using Point2Du32 = Vec2Du32;

