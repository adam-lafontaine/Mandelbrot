#pragma once

#include "../utils/types.hpp"

#include <array>

constexpr std::array< std::array<u8, 16>, 3> palettes16 =
{ {
	{ 66, 25,  9,  4,   0,  12,  24,  57, 134, 211, 241, 248, 255, 204, 153, 106 },
	{ 30,  7,  1,  4,   7,  44,  82, 125, 181, 236, 233, 201, 170, 128,  87,  57 },
	{ 15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191,  95,   0,   0,   0,   3 }
} };


constexpr u8 lerp(u8 a, u8 b, r64 t)
{
	return (u8)(a + t * (b - a) + 0.5);
}


template<size_t N>
constexpr std::array<u8, N> make_palette(u32 c)
{
	static_assert(N % 16 == 0);

	auto& palette16 = palettes16[c];

	std::array<u8, N> palette = {};

	auto C = N / 16;

	for (u32 i = 0; i < 15; ++i)
	{
		u32 i1 = i + 1;
		auto a = palette16[i];
		auto b = palette16[i1];
		
		for (u32 c = 0; c < C; ++c)
		{
			auto p = i * C + c;
			r64 t = c / (r64)C;
			palette[p] = lerp(a, b, t);
		}
	}

	auto a = palette16[15];
	auto b = palette16[0];

	for (u32 c = 0; c < C; ++c)
	{
		auto p = 15 * C + c;
		r64 t = c / (r64)C;
		palette[p] = lerp(a, b, t);
	}

	return palette;
}


template<size_t N>
constexpr std::array< std::array<u8, N>, 3> make_palettes()
{
	return 
	{ {
		make_palette<N>(0),
		make_palette<N>(1),
		make_palette<N>(2)
	} };
}


constexpr u32 N_COLOR_LEVELS = 8;


constexpr u32 calc_n_palette_colors()
{
	u32 n = 16;

	for (u32 i = 0; i < N_COLOR_LEVELS; ++i)
	{
		n *= 2;
	}

	return n;
}


constexpr std::array<u32, N_COLOR_LEVELS> make_color_levels()
{
	u32 min_level = 50;
	u32 max_level = 1000;

	constexpr std::array<u32, N_COLOR_LEVELS> levels = { 50, 100, 200, 300, 400, 500, 600, 800 };

	return levels;
}


constexpr std::array<u32, N_COLOR_LEVELS> color_levels = make_color_levels();

constexpr u32 N_COLORS = calc_n_palette_colors();

constexpr auto palettes = make_palettes<N_COLORS>();


static constexpr u32 num_rgb_combinations()
{
	return 3 * 2 * 1; // 3!
}