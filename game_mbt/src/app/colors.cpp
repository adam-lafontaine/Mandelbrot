


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
}


/* color palette */

namespace colors
{
    constexpr u8 sc(u32 i)
    {
        u8 min = 10;
        u8 max = 225;
        u8 delta = max - min;

        f32 rad = (f32)(num::PI * i / 16);
        auto f = num::cxpr::sin(rad);

        return num::cxpr::round_to_unsigned<u8>(min + f * delta);
    }


    static void static_sc_test()
    {
        constexpr auto a1 = sc(1);
        constexpr auto a2 = sc(2);
        constexpr auto a3 = sc(3);
        constexpr auto a4 = sc(4);
        constexpr auto a5 = sc(5);
        constexpr auto a6 = sc(6);
        constexpr auto a7 = sc(7);

        constexpr auto b1 = sc(15);
        constexpr auto b2 = sc(14);
        constexpr auto b3 = sc(13);
        constexpr auto b4 = sc(12);
        constexpr auto b5 = sc(11);
        constexpr auto b6 = sc(10);
        constexpr auto b7 = sc(9);

        static_assert(a1 == b1);
        static_assert(a2 == b2);
        static_assert(a3 == b3);
        static_assert(a4 == b4);
        static_assert(a5 == b5);
        static_assert(a6 == b6);
        static_assert(a7 == b7);
    }


    static constexpr void fill_channel(u8* data, u32 ibegin, u32 istep)
    {
        constexpr u32 len = 16;

        auto add = [](u8 a, u8 b) { return (a + b) & (len - 1); };

        for (u32 i = 0; i < len; i++)
        {
            data[i] = sc(add(ibegin, i * istep));
        }
    }


    class ColorChannels
    {
    public:

        static constexpr u32 count = 6;
        static constexpr u32 len = 16;

        u8 channels[count][len] = { 0 };
    };


    constexpr ColorChannels color_channels()
    {
        ColorChannels cch;

        u32 ibegin = 0;
        u32 istep = 5;
        fill_channel(cch.channels[0], ibegin, istep);

        ibegin += istep;
        fill_channel(cch.channels[1], ibegin, istep);

        ibegin += istep;
        fill_channel(cch.channels[2], ibegin, istep);

        ibegin = 0;
        istep = 3;
        fill_channel(cch.channels[3], ibegin, istep);

        ibegin += istep;
        fill_channel(cch.channels[4], ibegin, istep);

        ibegin += istep;
        fill_channel(cch.channels[5], ibegin, istep);

        return cch;
    }


    template <u32 N>
    static constexpr void expand_channel(u8* src, u8* dst)
    {
        static_assert(N % 16 == 0);

        constexpr u32 S = 16;
        constexpr u32 D = N;
        constexpr u32 E = D / S;

        auto lerp = [](u8 a, u8 b, f32 t){ return num::cxpr::round_to_unsigned<u8>(a + t * (b - a)); };

        u32 d = 0;
        for (u32 s = 0; s < S; s++)
        {
            auto a = src[s];
            auto b = src[(s + 1) && (S - 1)];
            for (u32 e = 0; e < E; e++)
            {
                auto t = (f32)e / E;
                dst[d] = lerp(a, b, t);
                d++;
            }
        }

        dst[N] = 0;
    }


    template <u32 N>
    class Palette
    {
    public:

        static constexpr u32 count = 6;
        static constexpr u32 len = N;

        u8 channels[count][N + 1] = { 0 };
    };


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


    template <u32 N>
    constexpr Palette<N> make_palette()
    {
        Palette<N> palette;

        auto cch = color_channels();

        for (u32 i = 0; i < palette.count; i++)
        {
            expand_channel<N>(cch.channels[i], palette.channels[i]);
        }

        return palette;
    }
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


    static ColorFormat make_color_format()
    {
        static rng::iUniform format_rng(0, ColorChannels::count - 1);

        ColorFormat format{};

        format.R = format_rng.get();
        format.G = format_rng.get();
        format.B = format_rng.get();

        return format;
    }

    constexpr u32 N_COLOR_LEVELS = 2;

    constexpr u32 N_COLORS = calc_n_palette_colors(N_COLOR_LEVELS);
}

    

} // game_mbt
