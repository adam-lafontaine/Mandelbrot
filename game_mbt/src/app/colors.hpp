#pragma once 


namespace game_mbt
{

/* n colors */

namespace colors
{
    constexpr u32 n_level_colors(u32 level)
    {
        u32 n = 32;

        for (u32 i = 0; i < level - 1; ++i)
        {
            n *= 2;
        }

        return n;
    }


    constexpr u32 n_colors_level(u32 n)
    {
        u32 level = 1;

        while (n > 32)
        {
            n /= 2;
            level++;
        }

        return level;
    }


    static constexpr void static_test_level_colors()
    {
        static_assert(n_colors_level(n_level_colors(1)) == 1);
        static_assert(n_colors_level(n_level_colors(2)) == 2);
        static_assert(n_colors_level(n_level_colors(3)) == 3);
        static_assert(n_colors_level(n_level_colors(4)) == 4);
        static_assert(n_colors_level(n_level_colors(5)) == 5);
        static_assert(n_colors_level(n_level_colors(6)) == 6);
        static_assert(n_colors_level(n_level_colors(7)) == 7);
        static_assert(n_colors_level(n_level_colors(8)) == 8);
    }


    constexpr u32 N_COLOR_LEVELS = 7;

    constexpr u32 N_COLORS = n_level_colors(N_COLOR_LEVELS);
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


    class MatrixViewRGB
    {
    public:
        u32 width = 0;
        u32 height = 0;

        u8* r = 0;
        u8* g = 0;
        u8* b = 0;

        img::GrayView view_red()
        {
            img::GrayView view{};

            view.matrix_data_ = r;
            view.width = width;
            view.height = height;

            return view;
        }


        img::GrayView view_green()
        {
            img::GrayView view{};

            view.matrix_data_ = g;
            view.width = width;
            view.height = height;

            return view;
        }


        img::GrayView view_blue()
        {
            img::GrayView view{};

            view.matrix_data_ = b;
            view.width = width;
            view.height = height;

            return view;
        }
    };


    class ColorMatrix
    {
    private:
        u8 p = 1;
        u8 c = 0;

    public:

        MatrixView2D<ColorId> id_data_[2];
        MatrixViewRGB rgb_data_[2];

        MatrixView2D<ColorId> id_prev() const { return id_data_[p]; }
        MatrixView2D<ColorId> id_curr() const { return id_data_[c]; }

        MatrixViewRGB rgb_prev() const { return rgb_data_[p]; }
        MatrixViewRGB rgb_curr() const { return rgb_data_[c]; }

        void swap() { p = c; c = !p; }

        MemoryBuffer<ColorId> id_buffer;
        MemoryBuffer<u8> rgb_buffer;
    };


    void destroy_color_ids(ColorMatrix& mat);

    bool create_color_ids(ColorMatrix& mat, u32 width, u32 height);
}