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

        auto rgb_src = mat.rgb_prev();
        auto rgb_dst = mat.rgb_curr();

        auto red_src   = img::sub_view(rgb_src.view_red(),   r_src);
        auto green_src = img::sub_view(rgb_src.view_green(), r_src);
        auto blue_src  = img::sub_view(rgb_src.view_blue(),  r_src);

        auto red_dst   = img::sub_view(rgb_dst.view_red(),   r_dst);
        auto green_dst = img::sub_view(rgb_dst.view_green(), r_dst);
        auto blue_dst  = img::sub_view(rgb_dst.view_blue(),  r_dst);

        assert(id_src.width == id_dst.width);
        assert(id_src.height == id_dst.height);
        
        auto h = rgb_src.height;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(img::row_span(id_src, y), img::row_span(id_dst, y));
            span::copy(img::row_span(red_src, y), img::row_span(red_dst, y));
            span::copy(img::row_span(green_src, y), img::row_span(green_dst, y));
            span::copy(img::row_span(blue_src, y), img::row_span(blue_dst, y));
        }
    }


    void proc_mbt(ColorMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto ids = mat.id_curr();
        auto rgb = mat.rgb_curr();

        auto w = rgb.width;
        auto h = rgb.height;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

        auto id_dst    = img::to_span(ids);        
        auto red_dst   = img::to_span(rgb.view_red());
        auto green_dst = img::to_span(rgb.view_green());
        auto blue_dst  = img::to_span(rgb.view_blue());

        auto mbt_at = [&](u32 i)
        {
            auto y = i / w;
            auto x = i - (w * y);

            auto cy = cy_begin + y * delta.y;
            auto cx = cx_begin + x * delta.x;

            auto iter = mandelbrot_iter(cx, cy, limit);
            auto color_id = to_color_id(iter, limit);
            auto px_id = color_id.value;

            id_dst.data[i]    = color_id;
            red_dst.data[i]   = r[px_id];
            green_dst.data[i] = g[px_id];
            blue_dst.data[i]  = b[px_id];
        };

        tbb::parallel_for(0u, w * h, mbt_at);
    }


    void proc_mbt_range(ColorMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto ids = img::sub_view(mat.id_curr(), r_dst);
        auto rgb = mat.rgb_curr();
        auto red =   img::sub_view(rgb.view_red(), r_dst);
        auto green = img::sub_view(rgb.view_green(), r_dst);
        auto blue =  img::sub_view(rgb.view_blue(), r_dst);

        auto w = ids.width;
        auto h = ids.height;

        auto x_begin = r_dst.x_begin;
        auto y_begin = r_dst.y_begin;

        auto cy_begin = (fmbt)y_begin * delta.y + begin.y;
        auto cx_begin = (fmbt)x_begin * delta.x + begin.x;

        auto cy = cy_begin;
        auto cx = cx_begin;

        auto mbt_row = [&](u32 y)
        {
            auto id_dst    = img::row_span(ids, y);
            auto red_dst   = img::row_span(red, y);
            auto green_dst = img::row_span(green, y);
            auto blue_dst  = img::row_span(blue, y);

            auto cx = cx_begin;
            auto cy = cy_begin + y * delta.y;
            
            for (u32 x = 0; x < w; x++)
            {
                auto iter = mandelbrot_iter(cx, cy, limit);
                auto color_id = to_color_id(iter, limit);
                auto px_id = color_id.value;

                id_dst.data[x]    = color_id;
                red_dst.data[x]   = r[px_id];
                green_dst.data[x] = g[px_id];
                blue_dst.data[x]  = b[px_id];

                cx += delta.x;
            }
        };

        tbb::parallel_for(0u, h, mbt_row);
    }


    void proc_render(ColorMatrix const& mat, img::ImageView const& screen)
    {
        auto rgb = mat.rgb_curr();
        auto red   = img::to_span(rgb.view_red());
        auto green = img::to_span(rgb.view_green());
        auto blue  = img::to_span(rgb.view_blue());

        auto dst = img::to_span(screen);

        auto pixel_at = [&](u32 i)
        {
            dst.data[i] = img::to_pixel(red.data[i], green.data[i], blue.data[i]);
        };

        tbb::parallel_for(0u, dst.length, pixel_at);
    }

}