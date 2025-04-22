


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


    constexpr u32 N_COLORS = calc_n_palette_colors();
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
    private:

        constexpr ColorId(u16 val) { assert(PaletteSize >= val); value = val; }

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


    constexpr std::array<u32, N_COLOR_LEVELS> make_color_levels()
    {
        u32 min_level = 50;
        u32 max_level = 1000;

        constexpr std::array<u32, N_COLOR_LEVELS> levels = { 50, 100, 200, 300, 400, 500, 600, 800 };

        return levels;
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


    static ColorFormat make_color_format(u32 option)
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


    
}

    using ColorTable = colors::Palette<colors::N_COLORS>;
    using ColorFormat = colors::ColorFormat;
    using ColorId = colors::ColorId<colors::N_COLORS>;


    static p32 color_at(ColorTable const& table, ColorFormat format, ColorId id)
    {
        auto r = table.channels[format.R][id.value];
        auto g = table.channels[format.G][id.value];
        auto b = table.channels[format.B][id.value];

        return img::to_pixel(r, g, b);
    }


    static ColorId to_color_id(u32 iter, u32 iter_limit)
    {
        constexpr auto C = colors::N_COLORS;
        constexpr u32 N = 8;
        
        u32 iter_levels[N] = { 50, 100, 200, 300, 400, 500, 600, 800 };

        auto id = ColorId::make_default();

        if (iter >= iter_limit)
        {
            return id;
        }

        u32 min = 0;
	    u32 max = 0;
        u32 n_colors = 8;

        for (u32 i = 0; i < N; i++)
        {
            n_colors *= 2;
            min = max;
            max = i;

            if (iter < max)
            {
                id.value = (iter - min) % n_colors * (C / n_colors);
                return id;
            }
        }

        min = max;
        id.value = (iter - min) % C;

        return id;
    }


} // game_mbt
