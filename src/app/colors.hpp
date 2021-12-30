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


constexpr std::array< std::array<u8, 32>, 3> palettes32 =
{ {
	make_palette<32>(0),
	make_palette<32>(1),
	make_palette<32>(2)
} };


constexpr std::array< std::array<u8, 48>, 3> palettes48 =
{ {
	make_palette<48>(0),
	make_palette<48>(1),
	make_palette<48>(2)
} };


constexpr std::array< std::array<u8, 64>, 3> palettes64 =
{ {
	make_palette<64>(0),
	make_palette<64>(1),
	make_palette<64>(2)
} };


constexpr std::array< std::array<u8, 80>, 3> palettes80 =
{ {
	make_palette<80>(0),
	make_palette<80>(1),
	make_palette<80>(2)
} };


constexpr std::array< std::array<u8, 96>, 3> palettes96 =
{ {
	make_palette<96>(0),
	make_palette<96>(1),
	make_palette<96>(2)
} };


constexpr std::array< std::array<u8, 112>, 3> palettes112 =
{ {
	make_palette<112>(0),
	make_palette<112>(1),
	make_palette<112>(2)
} };


constexpr std::array< std::array<u8, 128>, 3> palettes128 =
{ {
	make_palette<128>(0),
	make_palette<128>(1),
	make_palette<128>(2)
} };


constexpr std::array< std::array<u8, 144>, 3> palettes144 =
{ {
	make_palette<144>(0),
	make_palette<144>(1),
	make_palette<144>(2)
} };


constexpr std::array< std::array<u8, 160>, 3> palettes160 =
{ {
	make_palette<160>(0),
	make_palette<160>(1),
	make_palette<160>(2)
} };


constexpr std::array< std::array<u8, 176>, 3> palettes176 =
{ {
	make_palette<176>(0),
	make_palette<176>(1),
	make_palette<176>(2)
} };


constexpr std::array< std::array<u8, 192>, 3> palettes192 =
{ {
	make_palette<192>(0),
	make_palette<192>(1),
	make_palette<192>(2)
} };


constexpr std::array< std::array<u8, 208>, 3> palettes208 =
{ {
	make_palette<208>(0),
	make_palette<208>(1),
	make_palette<208>(2)
} };


constexpr std::array< std::array<u8, 224>, 3> palettes224 =
{ {
	make_palette<224>(0),
	make_palette<224>(1),
	make_palette<224>(2)
} };


constexpr std::array< std::array<u8, 240>, 3> palettes240 =
{ {
	make_palette<240>(0),
	make_palette<240>(1),
	make_palette<240>(2)
} };


constexpr std::array< std::array<u8, 256>, 3> palettes256 =
{ {
	make_palette<256>(0),
	make_palette<256>(1),
	make_palette<256>(2)
} };