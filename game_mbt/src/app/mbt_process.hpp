#pragma once



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


/* color ids */

namespace game_mbt
{
    using ColorFormat = colors::ColorFormat;
    using ColorId = colors::ColorId<colors::N_COLORS>;


    class ColorIdMatrix
    {
    private:
        u8 p = 1;
        u8 c = 0;

    public:

        MatrixView2D<ColorId> data_[2];

        MatrixView2D<ColorId> prev() const { return data_[p]; }
        MatrixView2D<ColorId> curr() const { return data_[c]; }

        void swap() { p = c; c = !p; }

        MemoryBuffer<ColorId> buffer;
    };


    void destroy_color_ids(ColorIdMatrix& mat);

    bool create_color_ids(ColorIdMatrix& mat, u32 width, u32 height);

    void copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst);


    inline auto to_span(MatrixView2D<ColorId> const& mat)
    {
        return img::to_span(mat);
    }


    inline auto sub_view(MatrixView2D<ColorId> const& mat, Rect2Du32 const& range)
    {
        return img::sub_view(mat, range);
    }
}


/* render */

namespace game_mbt
{
    void render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format);
}


/* mandelbrot */

namespace game_mbt
{
    void mbt_proc(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);


    static inline Vec2D<fmbt> mbt_screen_dims(f32 zoom)
    {
        auto scale = 1.0f / zoom;

        return {
            MBT_WIDTH * scale,
            MBT_HEIGHT * scale
        };
    }
}