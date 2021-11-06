#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

#include <cstdint>
#include <cstddef>
#include <array>

using u8 = uint8_t;
using u32 = uint32_t;
using u64 = uint64_t;
using r32 = float;
using r64 = double;
using i32 = int32_t;

namespace libimage
{
	// region of interest in an image
	typedef struct
	{
		u32 x_begin;
		u32 x_end;   // one past last x
		u32 y_begin;
		u32 y_end;   // one past last y

	} pixel_range_t;


	typedef struct
	{
		u32 x;
		u32 y;

	} xy_loc_t;


	constexpr size_t CHANNEL_SIZE = 256; // 8 bit channel
	constexpr size_t N_HIST_BUCKETS = 256;

	using hist_t = std::array<u32, N_HIST_BUCKETS>;


	typedef struct channel_stats_t
	{
		r32 mean;
		r32 std_dev;
		hist_t hist;

	} stats_t;


	typedef union rgb_channel_stats_t
	{
		struct
		{
			stats_t red;
			stats_t green;
			stats_t blue;
		};

		stats_t stats[3];

	} rgb_stats_t;
}


//#define NDEBUG