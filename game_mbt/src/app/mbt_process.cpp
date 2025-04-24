#include "mbt_process.hpp"


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


    void copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
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


    void render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format)
    {
        auto s = to_span(src.curr());
        auto d = img::to_span(dst);

        assert(s.length == d.length);

        for (u32 i = 0; i < s.length; i++)
        {
            d.data[i] = color_at(s.data[i], format);
        }
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


    void mbt_proc(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
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
}