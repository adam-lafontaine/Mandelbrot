#pragma once

#include "types.hpp"


constexpr r64 MBT_MIN_X = -2.0;
constexpr r64 MBT_MAX_X = 0.7;
constexpr r64 MBT_MIN_Y = -1.2;
constexpr r64 MBT_MAX_Y = 1.2;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;

inline r64 screen_width(AppState const& state)
{
	return MBT_WIDTH / state.zoom_level;
}


inline r64 screen_height(AppState const& state)
{
	return MBT_HEIGHT / state.zoom_level;
}

void render(image_t const& dst, AppState& state);