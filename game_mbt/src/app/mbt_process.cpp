#include "mbt_process.hpp"

#define PROC_PAR


#ifdef PROC_PAR
#include "../../../libs/for_each_in_range/for_each_in_range.hpp"
#endif


/* colors */

namespace game_mbt
{


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
    ColorFormat make_color_format()
    {
        static rng::iUniform format_rng(0, ColorChannels::count - 1);

        ColorFormat format{};

        format.R = format_rng.get();
        format.G = format_rng.get();
        format.B = format_rng.get();

        return format;
    }


    constexpr Palette<N_COLORS> make_table()
    {
        return make_palette<N_COLORS>();
    }
}


} // game_mbt


/* color ids */

namespace game_mbt
{
    void destroy_color_ids(ColorIdMatrix& mat)
    {
        mb::destroy_buffer(mat.buffer);
    }


    bool create_color_ids(ColorIdMatrix& mat, u32 width, u32 height)
    {
        auto n = width * height;

        if (!mb::create_buffer(mat.buffer, n * 2, "color_ids"))
        {
            return false;
        }

        for (u32 i = 0; i < 2; i++)
        {
            auto span = span::push_span(mat.buffer, n);
            mat.data_[i].matrix_data_ = span.data;
            mat.data_[i].width = width;
            mat.data_[i].height = height;
        }        

        return true;
    }


    
}


/* render */

namespace game_mbt
{
    static p32 color_at(ColorId id, ColorFormat format)
    {
        static constexpr auto Color_Table = colors::make_table();

        constexpr auto D = ColorId::make_default().value;

        static_assert(Color_Table.channels[0][D] == 0);
        static_assert(Color_Table.channels[1][D] == 0);
        static_assert(Color_Table.channels[2][D] == 0);
        static_assert(Color_Table.channels[3][D] == 0);
        static_assert(Color_Table.channels[4][D] == 0);
        static_assert(Color_Table.channels[5][D] == 0);

        auto r = Color_Table.channels[format.R][id.value];
        auto g = Color_Table.channels[format.G][id.value];
        auto b = Color_Table.channels[format.B][id.value];

        return img::to_pixel(r, g, b);
    }
}


/* mandelbrot */

namespace game_mbt
{
    static ColorId to_color_id(u32 iter, u32 iter_limit)
    {
        constexpr auto DEF = ColorId::make_default();

        return iter >= iter_limit ? DEF : ColorId::make(iter % colors::N_COLORS);
    }


    static u32 mandelbrot_iter(fmbt cx, fmbt cy, u32 iter_limit)
    {
        u32 iter = 0;
    
        fmbt mx = 0.0;
        fmbt my = 0.0;
        fmbt mx2 = 0.0;
        fmbt my2 = 0.0;
    
        while (iter < iter_limit && mx2 + my2 <= 4.0)
        {
            ++iter;
    
            my = (mx + mx) * my + cy;
            mx = mx2 - my2 + cx;
            my2 = my * my;
            mx2 = mx * mx;
        }
    
        return iter;
    }


    
}


/* proc */

namespace game_mbt
{
#ifdef PROC_PAR

    void proc_copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
    {
        auto src = sub_view(mat.prev(), r_src);
        auto dst = sub_view(mat.curr(), r_dst);

        assert(src.width == dst.width);
        assert(src.height == dst.height);

        auto w = src.width;
        auto h = src.height;

        auto stride = mat.curr().width;

        auto s = src.matrix_data_ + src.y_begin * stride + src.x_begin;
        auto d = dst.matrix_data_ + dst.y_begin * stride + dst.x_begin;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(span::make_view(s, w), span::make_view(d, w));

            s += stride;
            d += stride;
        }
    }


    void proc_mbt(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto view = mat.curr();
        auto dst = sub_view(view, r_dst);

        auto w = dst.width;
        auto h = dst.height;

        auto stride = view.width;

        auto d = dst.matrix_data_ + dst.y_begin * stride + dst.x_begin;

        auto cy_begin = (fmbt)dst.y_begin * delta.y + begin.y;
        auto cx_begin = (fmbt)dst.x_begin * delta.x + begin.x;

        auto cy = cy_begin;
        auto cx = cx_begin;

        for (u32 y = 0; y < h; y++)
        {
            for (u32 x = 0; x < w; x++)
            {
                auto iter = mandelbrot_iter(cx, cy, limit);                
                d[x] = to_color_id(iter, limit);

                cx += delta.x;
            }

            d += stride;
            cy += delta.y;
            cx = cx_begin;
        }
    }    


    void proc_render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format)
    {
        auto s = to_span(src.curr());
        auto d = img::to_span(dst);

        assert(s.length == d.length);

        for (u32 i = 0; i < s.length; i++)
        {
            d.data[i] = color_at(s.data[i], format);
        }
    }


#else

    void proc_copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
    {
        auto src = sub_view(mat.prev(), r_src);
        auto dst = sub_view(mat.curr(), r_dst);

        assert(src.width == dst.width);
        assert(src.height == dst.height);

        auto w = src.width;
        auto h = src.height;

        auto stride = mat.curr().width;

        auto s = src.matrix_data_ + src.y_begin * stride + src.x_begin;
        auto d = dst.matrix_data_ + dst.y_begin * stride + dst.x_begin;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(span::make_view(s, w), span::make_view(d, w));

            s += stride;
            d += stride;
        }
    }


    void proc_mbt(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto view = mat.curr();
        auto dst = sub_view(view, r_dst);

        auto w = dst.width;
        auto h = dst.height;

        auto stride = view.width;

        auto d = dst.matrix_data_ + dst.y_begin * stride + dst.x_begin;

        auto cy_begin = (fmbt)dst.y_begin * delta.y + begin.y;
        auto cx_begin = (fmbt)dst.x_begin * delta.x + begin.x;

        auto cy = cy_begin;
        auto cx = cx_begin;

        for (u32 y = 0; y < h; y++)
        {
            for (u32 x = 0; x < w; x++)
            {
                auto iter = mandelbrot_iter(cx, cy, limit);                
                d[x] = to_color_id(iter, limit);

                cx += delta.x;
            }

            d += stride;
            cy += delta.y;
            cx = cx_begin;
        }
    }    


    void proc_render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format)
    {
        auto s = to_span(src.curr());
        auto d = img::to_span(dst);

        assert(s.length == d.length);

        for (u32 i = 0; i < s.length; i++)
        {
            d.data[i] = color_at(s.data[i], format);
        }
    }


#endif
}