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


/* color ids */

namespace game_mbt
{
    using ColorFormat = colors::ColorFormat;
    using ColorId = colors::ColorId<colors::N_COLORS>;


    class ColorMatrix
    {
    private:
        u8 p = 1;
        u8 c = 0;

    public:

        MatrixView2D<ColorId> data_[2];

        MatrixView2D<ColorId> id_prev() const { return data_[p]; }
        MatrixView2D<ColorId> id_curr() const { return data_[c]; }

        void swap() { p = c; c = !p; }

        MemoryBuffer<ColorId> buffer;
    };


    void destroy_color_ids(ColorMatrix& mat);

    bool create_color_ids(ColorMatrix& mat, u32 width, u32 height);    


    inline auto to_span(MatrixView2D<ColorId> const& mat)
    {
        return img::to_span(mat);
    }


    inline auto sub_view(MatrixView2D<ColorId> const& mat, Rect2Du32 const& range)
    {
        return img::sub_view(mat, range);
    }
}