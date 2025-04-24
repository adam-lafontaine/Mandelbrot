


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

        static constexpr u32 n_channels = 6;
        static constexpr u32 len = 16;

        u8 channels[n_channels][len] = { 0 };
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
    }


    template <u32 N>
    class Palette
    {
    public:

        static constexpr u32 count = N;

        u8 channels[3][N + 1] = {
            /*{  sc(0) , sc(1) , sc(2) , sc(3) , sc(4) , sc(5) , sc(6) , sc(7) , sc(8) , sc(9) , sc(10), sc(11), sc(12), sc(13), sc(14), sc(15), 0 },
            {  sc(6) , sc(7) , sc(8) , sc(9) , sc(10), sc(11), sc(12), sc(13), sc(14), sc(15), sc(0) , sc(1) , sc(2) , sc(3) , sc(4) , sc(5) , 0 },
            {  sc(11), sc(12), sc(13), sc(14), sc(15), sc(0) , sc(1) , sc(2) , sc(3) , sc(4) , sc(5) , sc(6) , sc(7) , sc(8) , sc(9) , sc(10), 0 }*/

            /*{  sc(0) , sc(3) , sc(6) , sc(9) , sc(12), sc(15), sc(2) , sc(5) , sc(8) , sc(11), sc(14), sc(1) , sc(4) , sc(7) , sc(10), sc(13), 0 },
            {  sc(15), sc(2) , sc(5) , sc(8) , sc(11), sc(14), sc(1) , sc(4) , sc(7) , sc(10), sc(13), sc(0) , sc(3) , sc(6) , sc(9) , sc(12), 0 },
            {  sc(14), sc(1) , sc(4) , sc(7) , sc(10), sc(13), sc(0) , sc(3) , sc(6) , sc(9) , sc(12), sc(15), sc(2) , sc(5) , sc(8) , sc(11), 0 }*/

            /*{ sc(0) , sc(5) , sc(10), sc(15), sc(4) , sc(9) , sc(14), sc(3) , sc(8) , sc(13), sc(2) , sc(7) , sc(12), sc(1) , sc(6) , sc(11), 0 },
            { sc(5) , sc(10), sc(15), sc(4) , sc(9) , sc(14), sc(3) , sc(8) , sc(13), sc(2) , sc(7) , sc(11), sc(1) , sc(6) , sc(11), sc(0) , 0 },
            { sc(10), sc(15), sc(4) , sc(9) , sc(14), sc(3) , sc(8) , sc(13), sc(2) , sc(7) , sc(12), sc(1) , sc(6) , sc(11), sc(0) , sc(5) , 0 }*/

            /*{ 66, 25,  9,  4,   0,  12,  24,  57, 134, 211, 241, 248, 255, 204, 153, 106, 0 },
            { 30,  7,  1,  4,   7,  44,  82, 125, 181, 236, 233, 201, 170, 128,  87,  57, 0 },
            { 15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191,  95,  50,  30,  20,  10, 0 }*/
        };
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
    static constexpr p32 pixel_at(Palette<N> const& palette, ColorId<N> id)
    {
        auto i = id.value;
        return img::to_pixel(palette.channels[0][i], palette.channels[1][i], palette.channels[2][i]);
    }


    template <u32 N>
    constexpr Palette<N> make_palette_old()
    {
        static_assert(N % 16 == 0);

        auto lerp = [](u8 a, u8 b, f32 t){ return num::cxpr::round_to_unsigned<u8>(a + t * (b - a)); };

        constexpr Palette<16> palette16;
        Palette<N> palette;

        auto S = N / 16;
        auto C = 3;

        for (u32 i = 0; i < 15; i++)
        {
            auto a = pixel_at(palette16, ColorId<16>::make(i));
            auto b = pixel_at(palette16, ColorId<16>::make(i + 1));

            for (u32 s = 0; s < S; s++)
            {
                auto n = i * S + s;
                f32 t = (f32)s / S;

                palette.channels[0][n] = lerp(a.red, b.red, t);
                palette.channels[1][n] = lerp(a.green, b.green, t);
                palette.channels[2][n] = lerp(a.blue, b.blue, t);
            }
        }

        return palette;
    }


    template <u32 N>
    constexpr Palette<N> make_palette()
    {
        Palette<N> palette;

        auto cch = color_channels();

        expand_channel<N>(cch.channels[0], palette.channels[0]);
        expand_channel<N>(cch.channels[1], palette.channels[1]);
        expand_channel<N>(cch.channels[2], palette.channels[2]);

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


    static ColorFormat make_color_format(u8 option)
    {
        ColorFormat format{};

        auto& c1 = format.R;
        auto& c2 = format.G;
        auto& c3 = format.B;

        switch (option)
        {
        case 1:
            c1 = 0;
            c2 = 1;
            c3 = 2;
            break;

        case 2:
            c1 = 0;
            c2 = 2;
            c3 = 1;
            break;

        case 3:
            c1 = 1;
            c2 = 0;
            c3 = 2;
            break;

        case 4:
            c1 = 1;
            c2 = 2;
            c3 = 0;
            break;

        case 5:
            c1 = 2;
            c2 = 0;
            c3 = 1;
            break;

        case 6:
            c1 = 2;
            c2 = 1;
            c3 = 0;
            break;

        default:
            break;
        }

        return format;
    }

    constexpr u32 N_COLOR_LEVELS = 2;

    constexpr u32 N_COLORS = calc_n_palette_colors(N_COLOR_LEVELS);
}

    

} // game_mbt
