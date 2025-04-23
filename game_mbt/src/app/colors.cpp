


namespace game_mbt
{

/* n colors */

namespace colors
{
    constexpr u32 N_COLOR_LEVELS = 8;


    constexpr u32 calc_n_palette_colors()
    {
        u32 n = 16;

        for (u32 i = 0; i < N_COLOR_LEVELS; ++i)
        {
            n *= 2;
        }

        return n;
    }
}


/* color palette */

namespace colors
{

    template <u32 N>
    class Palette
    {
    public:

        static constexpr u32 count = N;

        u8 channels[3][N + 1] = {
            { 66, 25,  9,  4,   0,  12,  24,  57, 134, 211, 241, 248, 255, 204, 153, 106, 0 },
            { 30,  7,  1,  4,   7,  44,  82, 125, 181, 236, 233, 201, 170, 128,  87,  57, 0 },
            { 15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191,  95,   0,   0,   0,   3, 0 }
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
    constexpr Palette<N> make_palette()
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


    constexpr u32 N_COLORS = calc_n_palette_colors();
}

    

} // game_mbt
