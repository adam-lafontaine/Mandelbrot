#include "mbt_process.hpp"

#include <tbb/parallel_for.h>


namespace game_mbt
{
    void proc_copy(MBTMatrix& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
    {
        auto mbt_src = mat.mbt_prev();
        auto& mbt_dst = mat.mbt_curr();

        mbt_dst.limit = mbt_src.limit;

        auto iter_src = img::sub_view(mbt_src.view_iter(), r_src);
        auto iter_dst = img::sub_view(mbt_dst.view_iter(), r_dst);

        auto h = mbt_src.height;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(img::row_span(iter_src, y), img::row_span(iter_dst, y));
        }        
    }


    void proc_mbt(MBTMatrix& mat, u32 limit)
    {        
        auto src = mat.mbt_prev();
        auto& dst = mat.mbt_curr();
        dst.limit = limit;

        auto mbt_row = [&](u32 y) { mandelbrot_row(src, dst, y); };

        tbb::parallel_for(0u, src.height, mbt_row);
    }
    
    
    void proc_mbt(MBTMatrix& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto& dst = mat.mbt_curr();
        dst.limit = limit;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

        auto mbt_row = [&](u32 y)
        {
            Vec2D<fmbt> cpos = { cx_begin, cy_begin + y * delta.y };
            mandelbrot_row(dst, y, cpos, delta);
        };

        tbb::parallel_for(0u, dst.height, mbt_row);
    }


    void proc_mbt_range(MBTMatrix& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto& mbt = mat.mbt_curr();
        mbt.limit = limit;

        auto x_begin = r_dst.x_begin;
        auto y_begin = r_dst.y_begin;
        auto x_end = r_dst.x_end;
        auto y_end = r_dst.y_end;

        auto cy_begin = (fmbt)y_begin * delta.y + begin.y;
        auto cx_begin = (fmbt)x_begin * delta.x + begin.x;

        auto mbt_row = [&](u32 y)
        {
            Vec2D<fmbt> cpos = { cx_begin, cy_begin + y * delta.y };
            mandelbrot_span(mbt, y, x_begin, x_end, cpos, delta);
        };

        tbb::parallel_for(y_begin, y_end, mbt_row);
    }


    void proc_render(MBTMatrix const& mat, img::ImageView const& screen, ColorFormat format, u32 n_colors)
    {
        auto r = Color_Table.channels[format.R];
        auto g = Color_Table.channels[format.G];
        auto b = Color_Table.channels[format.B];

        auto mbt = mat.mbt_curr();

        auto src = img::to_span(mbt.view_iter());
        auto dst = img::to_span(screen);

        auto render_at = [&](u32 i)
        {
            auto id = to_color_id(src.data[i], mbt.limit, n_colors).value;
            dst.data[i] = img::to_pixel(r[id], g[id], b[id]);
        };

        tbb::parallel_for(0u, dst.length, render_at);
    }
}