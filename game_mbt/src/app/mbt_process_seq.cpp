#include "mbt_process.hpp"


/* proc */

namespace game_mbt
{


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


    void proc_mbt(ColorIdMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto dst = mat.curr();

        auto w = dst.width;
        auto h = dst.height;

        auto stride = dst.width;

        auto d = dst.matrix_data_;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

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


    void proc_mbt_range(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
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
        static constexpr auto Color_Table = colors::make_table();

        constexpr auto D = ColorId::make_default().value;

        static_assert(Color_Table.channels[0][D] == 0);
        static_assert(Color_Table.channels[1][D] == 0);
        static_assert(Color_Table.channels[2][D] == 0);
        static_assert(Color_Table.channels[3][D] == 0);
        static_assert(Color_Table.channels[4][D] == 0);
        static_assert(Color_Table.channels[5][D] == 0);

        auto s = to_span(src.curr());
        auto d = img::to_span(dst);

        assert(s.length == d.length);

        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        for (u32 i = 0; i < s.length; i++)
        {
            auto id = s.data[i].value;
            d.data[i] = img::to_pixel(r[id], g[id], b[id]);
        }
    }

}