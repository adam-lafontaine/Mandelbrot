#include "mbt_process.hpp"


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
        
        auto h = src.height;

        for (u32 y = 0; y < h; y++)
        {            
            mandelbrot_row(src, dst, y);
        }
    }
    
    
    void proc_mbt(MBTMatrix& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto& dst = mat.mbt_curr();
        dst.limit = limit;

        auto h = dst.height;

        auto cy_begin = begin.y;
        auto cx_begin = begin.x;

        Vec2D<fmbt> cpos = { cx_begin, cy_begin };

        for (u32 y = 0; y < h; y++)
        {   
            mandelbrot_row(dst, y, cpos, delta);
            cpos.y += delta.y;
        }
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

        Vec2D<fmbt> cpos = { cx_begin, cy_begin };

        for (u32 y = y_begin; y < y_end; y++)
        {
            mandelbrot_span(mbt, y, x_begin, x_end, cpos, delta);
            cpos.y += delta.y;
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