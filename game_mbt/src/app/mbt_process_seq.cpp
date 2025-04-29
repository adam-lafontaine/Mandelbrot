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


namespace game_mbt
{
    class SpanViewMBT
    {
    public:
        fmbt* cx = 0;
        fmbt* cy = 0;

        fmbt* mx = 0;
        fmbt* my = 0;

        fmbt* mx2 = 0;
        fmbt* my2 = 0;

        u32* iter = 0;

        u32 length = 0;
    };


    SpanViewMBT row_span(MatrixViewMBT const& mbt, u32 y)
    {
        SpanViewMBT span{};

        span.cx = img::row_span(mbt.view_cx(), y).data;
        span.cy = img::row_span(mbt.view_cy(), y).data;
        span.mx = img::row_span(mbt.view_mx(), y).data;
        span.my = img::row_span(mbt.view_my(), y).data;
        span.mx2 = img::row_span(mbt.view_mx2(), y).data;
        span.my2 = img::row_span(mbt.view_my2(), y).data;
        span.iter = img::row_span(mbt.view_iter(), y).data;

        span.length = mbt.width;

        return span;
    }    


    SpanViewMBT row_sub_span(MatrixViewMBT const& mbt, u32 y, u32 x_begin, u32 x_end)
    {
        SpanViewMBT span{};

        span.cx = img::row_span(mbt.view_cx(), y).data + x_begin;
        span.cy = img::row_span(mbt.view_cy(), y).data + x_begin;
        span.mx = img::row_span(mbt.view_mx(), y).data + x_begin;
        span.my = img::row_span(mbt.view_my(), y).data + x_begin;
        span.mx2 = img::row_span(mbt.view_mx2(), y).data + x_begin;
        span.my2 = img::row_span(mbt.view_my2(), y).data + x_begin;
        span.iter = img::row_span(mbt.view_iter(), y).data + x_begin;

        span.length = x_end - x_begin;;

        return span;
    }


    static void mandelbrot_row(MatrixViewMBT const& dst, u32 y, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta)
    {
        auto row = row_span(dst, y);

        MBTData mdata{};

        mdata.cx = begin.x;
        mdata.cy = begin.y;
        mdata.limit = dst.limit;

        auto len = row.length;

        for (u32 i = 0; i < len; i++)
        {
            zero_iter(mdata);
            mandelbrot_iter(mdata);

            row.cx[i] = mdata.cx;
            row.cy[i] = mdata.cy;
            row.mx[i] = mdata.mx;
            row.my[i] = mdata.my;
            row.mx2[i] = mdata.mx2;
            row.my2[i] = mdata.my2;
            row.iter[i] = mdata.iter;

            mdata.cx += delta.x;
        }
    }


    static void mandelbrot_row(MatrixViewMBT const& src, MatrixViewMBT const& dst, u32 y)
    {
        auto row_src = row_span(src, y);
        auto row_dst = row_span(dst, y);

        MBTData mdata{};
        mdata.limit = dst.limit;

        auto len = row_dst.length;

        for (u32 i = 0; i < len; i++)
        {
            mdata.cx = row_src.cx[i];
            mdata.cy = row_src.cy[i];

            mdata.mx = row_src.mx[i];
            mdata.my = row_src.my[i];
            mdata.mx2 = row_src.mx2[i];
            mdata.my2 = row_src.my2[i];
            mdata.iter = row_src.iter[i];

            mandelbrot_iter(mdata);
            
            row_dst.cx[i] = mdata.cx;
            row_dst.cy[i] = mdata.cy;
            
            row_dst.mx[i] = mdata.mx;
            row_dst.my[i] = mdata.my;
            row_dst.mx2[i] = mdata.mx2;
            row_dst.my2[i] = mdata.my2;
            row_dst.iter[i] = mdata.iter;
        }
    }


    static void mandelbrot_span(MatrixViewMBT const& dst, u32 y, u32 x_begin, u32 x_end, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta)
    {
        auto row = row_sub_span(dst, y, x_begin, x_end);

        MBTData mdata{};

        mdata.cx = begin.x;
        mdata.cy = begin.y;
        mdata.limit = dst.limit;

        auto len = row.length;

        for (u32 i = 0; i < len; i++)
        {
            zero_iter(mdata);
            mandelbrot_iter(mdata);

            row.cx[i] = mdata.cx;
            row.cy[i] = mdata.cy;
            row.mx[i] = mdata.mx;
            row.my[i] = mdata.my;
            row.mx2[i] = mdata.mx2;
            row.my2[i] = mdata.my2;
            row.iter[i] = mdata.iter;

            mdata.cx += delta.x;
        }
    }
}


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
            cpos.x = cx_begin;
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