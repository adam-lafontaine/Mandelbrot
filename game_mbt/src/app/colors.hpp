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


    constexpr u32 N_COLOR_LEVELS = 6;

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


    template <u32 N>
    class ColorId
    {
    public:
        static constexpr u32 max = N;

    private:

        constexpr ColorId(u16 val) { assert(val <= max); value = val; }

        template <typename T>
        static constexpr ColorId make_pvt(T val) { return ColorId((u16)val); }

    public:
        u16 value = (u16)N;

        static constexpr ColorId make(u8 v) { return make_pvt(v); }
        static constexpr ColorId make(u16 v) { return make_pvt(v); }
        static constexpr ColorId make(u32 v) { return make_pvt(v); }

        static constexpr ColorId make_default() { return make_pvt((u16)N); }

        ColorId() = delete;
    };
}

    

} // game_mbt


/* color ids */

namespace game_mbt
{
    using ColorFormat = colors::ColorFormat;
    using ColorId = colors::ColorId<colors::N_COLORS>;


    static inline ColorId to_color_id(u32 iter, u32 limit, u32 n_colors)
    {   
        constexpr auto N = colors::N_COLORS;
        constexpr auto DEF = ColorId::make_default();

        if (iter >= limit)
        {
            return DEF;
        }
        
        u32 d = n_colors;
        u32 n = iter % d;

        return ColorId::make(N * n / d);
    }
}