#pragma once

#include "app.hpp"

#include "../../../libs/alloc_type/alloc_type.hpp"
#include "../../../libs/util/numeric.hpp"
#include "../../../libs/util/stopwatch.hpp"
#include "../../../libs/util/rng.hpp"

/* defines */

namespace game_mbt
{
    namespace img = image;
    namespace num = numeric;
    namespace mb = memory_buffer;

    using fmbt = double;

    using p32 = img::Pixel;
    using ImageView = img::ImageView;
    using Input = input::Input;
}

#include "colors.hpp"
#include "mbt_process.hpp"
#include "map_input.cpp"


/* constants */

namespace game_mbt
{

    #ifdef SDL2_WASM

	constexpr u32 BUFFER_WIDTH = 640;
	constexpr u32 BUFFER_HEIGHT = BUFFER_WIDTH * 8 / 9;

	#else

	// allocate memory
	constexpr u32 BUFFER_HEIGHT = 800;
	constexpr u32 BUFFER_WIDTH = BUFFER_HEIGHT * 9 / 8;

	#endif

	constexpr u32 PIXELS_PER_SECOND = (u32)(0.2 * BUFFER_HEIGHT);

    constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = colors::N_COLORS;
    constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = MAX_ITERTAIONS_LOWER_LIMIT * 16;
    constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
    constexpr f32 ZOOM_RATE_LOWER_LIMIT = 1.0f;


    constexpr fmbt MBT_MIN_X = -2.0;
    constexpr fmbt MBT_MAX_X = 0.7;
    constexpr fmbt MBT_MIN_Y = -1.2;
    constexpr fmbt MBT_MAX_Y = 1.2;
    constexpr fmbt MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
    constexpr fmbt MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;
}


/* mandelbrot */

namespace game_mbt
{
    static inline Vec2D<fmbt> mbt_screen_dims(f32 zoom)
    {
        auto scale = 1.0f / zoom;

        return {
            MBT_WIDTH * scale,
            MBT_HEIGHT * scale
        };
    }
}