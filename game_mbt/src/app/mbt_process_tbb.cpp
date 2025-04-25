#include "mbt_process.hpp"

#include <tbb/parallel_for.h>

/* color table */

namespace game_mbt
{
    static constexpr auto Color_Table = colors::make_table();

    static void static_test_color_table()
    {
        constexpr auto D = ColorId::make_default().value;

        static_assert(Color_Table.channels[0][D] == 0);
        static_assert(Color_Table.channels[1][D] == 0);
        static_assert(Color_Table.channels[2][D] == 0);
        static_assert(Color_Table.channels[3][D] == 0);
        static_assert(Color_Table.channels[4][D] == 0);
        static_assert(Color_Table.channels[5][D] == 0);
    }
}


/* proc */

namespace game_mbt
{
    void proc_copy(ColorMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
    {
        auto id_src = img::sub_view(mat.id_prev(), r_src);
        auto id_dst = img::sub_view(mat.id_curr(), r_dst);

        auto px_src = img::sub_view(mat.px_prev(), r_src);
        auto px_dst = img::sub_view(mat.px_curr(), r_dst);

        assert(id_src.width == id_dst.width);
        assert(id_src.height == id_dst.height);
        
        auto h = px_src.height;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(img::row_span(id_src, y), img::row_span(id_dst, y));
            span::copy(img::row_span(px_src, y), img::row_span(px_dst, y));
        }
    }


    void proc_mbt(ColorMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];
        
        auto dst = mat.px_curr();
        auto ids = mat.id_curr();

        auto w = dst.width;
        auto h = dst.height;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

        auto cy = cy_begin;
        auto cx = cx_begin;

        auto mbt_at = [&](u32 i)
        {
            auto y = i / w;
            auto x = i - (w * y);

            auto id = img::row_span(ids, y).data;
            auto px = img::row_span(dst, y).data;

            auto cy = cy_begin + y * delta.y;
            auto cx = cx_begin + x * delta.x;

            auto iter = mandelbrot_iter(cx, cy, limit);
            auto color_id = to_color_id(iter, limit);
            auto px_id = color_id.value;
            
            id[x] = color_id;
            px[x] = img::to_pixel(r[px_id], g[px_id], b[px_id]);
        };

        tbb::parallel_for(0u, w * h, mbt_at);
    }


    void proc_mbt_range(ColorMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto view = mat.px_curr();

        auto dst = img::sub_view(view, r_dst);
        auto ids = img::sub_view(mat.id_curr(), r_dst);

        auto w = dst.width;
        auto h = dst.height;

        auto x_begin = dst.x_begin;
        auto y_begin = dst.y_begin;

        auto stride = view.width;

        auto px_begin = dst.matrix_data_ + y_begin * stride + x_begin;
        auto id_begin = ids.matrix_data_ + y_begin * stride + x_begin;

        auto cy_begin = (fmbt)y_begin * delta.y + begin.y;
        auto cx_begin = (fmbt)x_begin * delta.x + begin.x;

        auto px = px_begin;
        auto id = id_begin;

        auto cy = cy_begin;
        auto cx = cx_begin;

        for (u32 y = 0; y < h; y++)
        {
            for (u32 x = 0; x < w; x++)
            {
                auto iter = mandelbrot_iter(cx, cy, limit);
                auto color_id = to_color_id(iter, limit);
                auto px_id = color_id.value;
                id[x] = color_id;
                px[x] = img::to_pixel(r[px_id], g[px_id], b[px_id]);

                cx += delta.x;
            }

            px += stride;
            id += stride;
            cy += delta.y;
            cx = cx_begin;
        }
    }

}