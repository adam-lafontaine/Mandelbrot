#pragma once


namespace game_mbt
{

/* n colors */

namespace colors
{
    constexpr u32 calc_n_palette_colors(u32 n_levels)
    {
        u32 n = 16;

        for (u32 i = 0; i < n_levels; ++i)
        {
            n *= 2;
        }

        return n;
    }


    constexpr u32 N_COLOR_LEVELS = 2;

    constexpr u32 N_COLORS = calc_n_palette_colors(N_COLOR_LEVELS);
}


/* color format */

namespace colors
{
    class ColorFormat
    {
    public:
        u8 R = 0;
        u8 G = 1;
        u8 B = 2;
    };


    ColorFormat make_color_format();


    template <u32 PaletteSize>
    class ColorId
    {
    public:
        static constexpr u32 max = PaletteSize;

    private:

        constexpr ColorId(u16 val) { assert(val <= max); value = val; }

        template <typename T>
        static constexpr ColorId make_pvt(T val) { return ColorId((u16)val); }

    public:
        u16 value = (u16)PaletteSize;

        static constexpr ColorId make(u8 v) { return make_pvt(v); }
        static constexpr ColorId make(u16 v) { return make_pvt(v); }
        static constexpr ColorId make(u32 v) { return make_pvt(v); }

        static constexpr ColorId make_default() { return make_pvt((u16)PaletteSize); }

        ColorId() = delete;
    };
}

    

} // game_mbt



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

    void proc_copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst);


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
    void proc_mbt(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);


    static inline Vec2D<fmbt> mbt_screen_dims(f32 zoom)
    {
        auto scale = 1.0f / zoom;

        return {
            MBT_WIDTH * scale,
            MBT_HEIGHT * scale
        };
    }
}