#pragma once

#include "cuda_types.hpp"


constexpr r64 MBT_MIN_X = -2.0;
constexpr r64 MBT_MAX_X = 0.7;
constexpr r64 MBT_MIN_Y = -1.2;
constexpr r64 MBT_MAX_Y = 1.2;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;

inline r64 mbt_screen_width(r64 zoom)
{
	return MBT_WIDTH / zoom;
}


inline r64 mbt_screen_height(r64 zoom)
{
	return MBT_HEIGHT / zoom;
}


void render(AppState& state);