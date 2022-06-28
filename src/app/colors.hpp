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
	return (u8)(a + t * (b - a));
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


constexpr auto palettes32 = make_palettes<32>();


constexpr auto palettes48 = make_palettes<48>();

constexpr auto palettes64 = make_palettes<64>();


constexpr auto palettes80 = make_palettes<80>();


constexpr auto palettes96 = make_palettes<96>();


constexpr auto palettes112 = make_palettes<112>();


constexpr auto palettes128 = make_palettes<128>();


constexpr auto palettes144 = make_palettes<144>();


constexpr auto palettes160 = make_palettes<160>();


constexpr auto palettes176 = make_palettes<176>();


constexpr auto palettes192 = make_palettes<192>();


constexpr auto palettes208 = make_palettes<208>();


constexpr auto palettes224 = make_palettes<224>();


constexpr auto palettes240 = make_palettes<240>();


constexpr auto palettes256 = make_palettes<256>();