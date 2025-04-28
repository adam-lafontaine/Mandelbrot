#include "mbt_process.hpp"

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
        auto red   = rgb.view_red();
        auto green = rgb.view_green();
        auto blue  = rgb.view_blue();

        auto w = rgb.width;
        auto h = rgb.height;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

        auto cy = cy_begin;
        auto cx = cx_begin;

        for (u32 y = 0; y < h; y++)
        {
            auto id_dst    = img::row_span(ids, y);
            auto red_dst   = img::row_span(red, y);
            auto green_dst = img::row_span(green, y);
            auto blue_dst  = img::row_span(blue, y);

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

            cy += delta.y;
            cx = cx_begin;
        }
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

        for (u32 y = 0; y < h; y++)
        {
            auto id_dst    = img::row_span(ids, y);
            auto red_dst   = img::row_span(red, y);
            auto green_dst = img::row_span(green, y);
            auto blue_dst  = img::row_span(blue, y);
            
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
            
            cy += delta.y;
            cx = cx_begin;
        }
    }


    void proc_render(ColorMatrix const& mat, img::ImageView const& screen)
    {
        auto rgb = mat.rgb_curr();
        auto red   = img::to_span(rgb.view_red());
        auto green = img::to_span(rgb.view_green());
        auto blue  = img::to_span(rgb.view_blue());

        auto dst = img::to_span(screen);

        for (u32 i = 0; i < dst.length; i++)
        {
            dst.data[i] = img::to_pixel(red.data[i], green.data[i], blue.data[i]);
        }
    }

}


/* by frame */

namespace game_mbt
{
    void proc_mbt_(ColorMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto ids = mat.id_curr();
        auto rgb = mat.rgb_curr();
        auto red   = rgb.view_red();
        auto green = rgb.view_green();
        auto blue  = rgb.view_blue();

        auto w = rgb.width;
        auto h = rgb.height;

        auto dy = delta.y;
        auto dx = delta.x;

        u32 n = 4;

        auto cy_begin = (begin.y * 2 + (n - 1) * dy) / 2;
        auto cx_begin = (begin.x * 2 + (n - 1) * dx) / 2;

        auto cy = cy_begin;
        auto cx = cx_begin;

        ColorId* id_dst[4] = {0};
        u8* r_dst[4] = {0};
        u8* g_dst[4] = {0};
        u8* b_dst[4] = {0};

        for (u32 y = 0; y < h; y += n)
        {
            for (u32 i = 0; i < n; i++)
            {
                id_dst[i] = img::row_span(ids, y + i).data;
                r_dst[i] = img::row_span(red, y + i).data;
                g_dst[i] = img::row_span(green, y + i).data;
                b_dst[i] = img::row_span(blue, y + i).data;
            }

            for (u32 x = 0; x < w; x += n)
            {
                auto iter = mandelbrot_iter(cx, cy, limit);
                auto id = to_color_id(iter, limit);
                auto px = id.value;

                auto rp = r[px];
                auto gp = g[px];
                auto bp = b[px];

                for (u32 i = 0; i < n; i++)
                {
                    for (u32 j = 0; j < n; j++)
                    {
                        id_dst[i][x + j] = id;
                        r_dst[i][x + j] = rp;
                        g_dst[i][x + j] = gp;
                        b_dst[i][x + j] = bp;
                    }
                }

                cx += n * dx;
            }

            cy += n * dy;
            cx = cx_begin;
        }
    }


    void proc_mbt_1(ColorMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto ids = mat.id_curr();
        auto rgb = mat.rgb_curr();
        auto red   = rgb.view_red();
        auto green = rgb.view_green();
        auto blue  = rgb.view_blue();

        auto w = rgb.width;
        auto h = rgb.height;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

        auto cy = cy_begin;
        auto cx = cx_begin;

        auto dy = delta.y;
        auto dx = delta.x;


    }
}


namespace game_mbt
{
    void proc_copy(MBTMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
    {
        auto mbt_src = mat.mbt_prev();
        auto mbt_dst = mat.mbt_curr();

        mbt_dst.limit = mbt_src.limit;

        auto cx_src = img::sub_view(mbt_src.view_cx(), r_src);
        auto cx_dst = img::sub_view(mbt_dst.view_cx(), r_dst);

        auto cy_src = img::sub_view(mbt_src.view_cy(), r_src);
        auto cy_dst = img::sub_view(mbt_dst.view_cy(), r_dst);

        auto mx_src = img::sub_view(mbt_src.view_mx(), r_src);
        auto mx_dst = img::sub_view(mbt_dst.view_mx(), r_dst);

        auto my_src = img::sub_view(mbt_src.view_my(), r_src);
        auto my_dst = img::sub_view(mbt_dst.view_my(), r_dst);

        auto mx2_src = img::sub_view(mbt_src.view_mx2(), r_src);
        auto mx2_dst = img::sub_view(mbt_dst.view_mx2(), r_dst);

        auto my2_src = img::sub_view(mbt_src.view_my2(), r_src);
        auto my2_dst = img::sub_view(mbt_dst.view_my2(), r_dst);

        auto iter_src = img::sub_view(mbt_src.view_iter(), r_src);
        auto iter_dst = img::sub_view(mbt_dst.view_iter(), r_dst);

        auto h = mbt_src.height;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(img::row_span(cx_src, y), img::row_span(cx_dst, y));
            span::copy(img::row_span(cy_src, y), img::row_span(cy_dst, y));

            span::copy(img::row_span(cy_src, y), img::row_span(cy_dst, y));
            span::copy(img::row_span(cy_src, y), img::row_span(cy_dst, y));

            span::copy(img::row_span(mx_src, y), img::row_span(mx_dst, y));
            span::copy(img::row_span(my_src, y), img::row_span(my_dst, y));

            span::copy(img::row_span(mx2_src, y), img::row_span(mx2_dst, y));
            span::copy(img::row_span(my2_src, y), img::row_span(my2_dst, y));

            span::copy(img::row_span(iter_src, y), img::row_span(iter_dst, y));
        }        
    }


    void proc_mbt(MBTMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto mbt = mat.mbt_curr();
        mbt.limit = limit;

        auto w = mbt.width;
        auto h = mbt.height;

        auto cpos = begin;

        for (u32 y = 0; y < h; y++)
        {            
            for (u32 x = 0; x < w; x++)
            {
                mandelbrot_pos_xy(mbt, cpos, x, y);
            }

            cpos.y += delta.y;
            cpos.x = begin.x;
        }
    }


    void proc_mbt_range(MBTMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto mbt = mat.mbt_curr();
        mbt.limit = limit;

        auto x_begin = r_dst.x_begin;
        auto y_begin = r_dst.y_begin;
        auto x_end = r_dst.x_end;
        auto y_end = r_dst.y_end;

        auto cy_begin = (fmbt)y_begin * delta.y + begin.y;
        auto cx_begin = (fmbt)x_begin * delta.x + begin.x;

        Vec2D<fmbt> cpos = { cx_begin, cy_begin };

        for (u32 y = y_begin; y < y_end; y++)
        {
            for (u32 x = x_begin; x < x_end; x++)
            {
                mandelbrot_pos_xy(mbt, cpos, x, y);
            }

            cpos.y += delta.y;
            cpos.x = begin.x;
        }
    }


    void proc_render(MBTMatrix const& mat, img::ImageView const& screen, ColorFormat format, u32 n_colors)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto mbt = mat.mbt_curr();

        auto src = img::to_span(mbt.view_iter());
        auto dst = img::to_span(screen);

        for (u32 i = 0; i < dst.length; i++)
        {
            auto id = to_color_id(src.data[i], mbt.limit, n_colors).value;
            dst.data[i] = img::to_pixel(r[id], g[id], b[id]);
        }
    }
}