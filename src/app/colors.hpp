#pragma once

#include "../utils/types.hpp"

#include <array>

constexpr std::array< std::array<u8, 16>, 3> palettes16 =
{ {
	{ 66, 25,  9,  4,   0,  12,  24,  57, 134, 211, 241, 248, 255, 204, 153, 106 },
	{ 30,  7,  1,  4,   7,  44,  82, 125, 181, 236, 233, 201, 170, 128,  87,  57 },
	{ 15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191,  95,   0,   0,   0,   3 }
} };


static constexpr std::array<u8, 32> make_palette32(u32 c)
{
	auto& palette16 = palettes16[c];

	auto const lerp = [](u8 a, u8 b, r64 t)
	{
		return static_cast<u8>(a + t * (b - a));
	};

	std::array<u8, 32> palette = {};

	for (u32 i = 0; i < 15; ++i)
	{
		u32 p = 2 * i;
		u32 p1 = p + 1;
		u32 i1 = i + 1;
		palette[p] = palette16[i];
		palette[p1] = lerp(palette16[i], palette16[i1], 0.5);
	}

	palette[30] = palette16[15];
	palette[31] = lerp(palette16[15], palette16[0], 0.5);

	return palette;
}


constexpr std::array< std::array<u8, 32>, 3> palettes32 =
{ {
	make_palette32(0),
	make_palette32(1),
	make_palette32(2)
} };


static constexpr std::array<u8, 64> make_palette64(u32 c)
{
	auto& palette16 = palettes16[c];

	auto const lerp = [](u8 a, u8 b, r64 t)
	{
		return static_cast<u8>(a + t * (b - a));
	};

	std::array<u8, 64> palette = {};

	for (u32 i = 0; i < 15; ++i)
	{
		u32 p = 2 * i;
		u32 p1 = p + 1;
		u32 p2 = p1 + 1;
		u32 p3 = p2 + 1;
		u32 i1 = i + 1;
		palette[p] = palette16[i];
		palette[p1] = lerp(palette16[i], palette16[i1], 0.25);
		palette[p2] = lerp(palette16[i], palette16[i1], 0.5);
		palette[p3] = lerp(palette16[i], palette16[i1], 0.75);
	}

	palette[60] = palette16[15];
	palette[61] = lerp(palette16[15], palette16[0], 0.25);
	palette[62] = lerp(palette16[15], palette16[0], 0.5);
	palette[63] = lerp(palette16[15], palette16[0], 0.75);

	return palette;
}


constexpr std::array< std::array<u8, 64>, 3> palettes64 =
{ {
	make_palette64(0),
	make_palette64(1),
	make_palette64(2)
} };


