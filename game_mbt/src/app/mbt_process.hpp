#pragma once


/* basic mandelbrot kernel */

namespace game_mbt
{
    // The mandelbrot iteration loop
    static inline u32 mandelbrot_iter(fmbt cx, fmbt cy, u32 iter_limit)
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


/* data matrix */

namespace game_mbt
{

    class MatrixViewMBT
    {
    public:
        fmbt* cx_ = 0;
        fmbt* cy_ = 0;

        fmbt* mx_ = 0;
        fmbt* my_ = 0;

        fmbt* mx2_ = 0;
        fmbt* my2_ = 0;

        u32* iter_ = 0;

        u32 width = 0;
        u32 height = 0;

        u32 limit = 0;


        auto view_cx() { return img::make_view(cx_, width, height); }
        auto view_cy() { return img::make_view(cy_, width, height); }
        auto view_mx() { return img::make_view(mx_, width, height); }
        auto view_my() { return img::make_view(my_, width, height); }
        auto view_mx2() { return img::make_view(mx2_, width, height); }
        auto view_my2() { return img::make_view(my2_, width, height); }
        auto view_iter() { return img::make_view(iter_, width, height); }

        auto view_cx() const { return img::make_view(cx_, width, height); }
        auto view_cy() const { return img::make_view(cy_, width, height); }
        auto view_mx() const { return img::make_view(mx_, width, height); }
        auto view_my() const { return img::make_view(my_, width, height); }
        auto view_mx2() const { return img::make_view(mx2_, width, height); }
        auto view_my2() const { return img::make_view(my2_, width, height); }
        auto view_iter() const { return img::make_view(iter_, width, height); }
    };


    class MBTMatrix
    {
    private:
        u8 p = 1;
        u8 c = 0;

    public:
        MatrixViewMBT mbt_data_[2];

        MatrixViewMBT mbt_prev() const { return mbt_data_[p]; }
        MatrixViewMBT& mbt_curr() { return mbt_data_[c]; }

        MatrixViewMBT mbt_curr() const { return mbt_data_[c]; }

        void swap() { p = c; c = !p; }

        MemoryBuffer<fmbt> buffer64;
        MemoryBuffer<u32> buffer32;
    };


    inline bool create_mbt(MBTMatrix& mat, u32 width, u32 height)
    {
        auto len = width * height;

        auto n_mbt = len * 6 * 2;
        if (!mb::create_buffer(mat.buffer64, n_mbt, "mbt buffer64"))
        {
            return false;
        }

        auto n_32 = len * 2;
        if (!mb::create_buffer(mat.buffer32, n_32, "mbt buffer 32"))
        {
            return false;
        }

        for (u32 i = 0; i < 2; i++)
        {
            auto& mbt = mat.mbt_data_[i];

            mbt.cx_ = mb::push_elements(mat.buffer64, len);
            mbt.cy_ = mb::push_elements(mat.buffer64, len);

            mbt.mx_ = mb::push_elements(mat.buffer64, len);
            mbt.my_ = mb::push_elements(mat.buffer64, len);

            mbt.mx2_ = mb::push_elements(mat.buffer64, len);
            mbt.my2_ = mb::push_elements(mat.buffer64, len);

            mbt.iter_ = mb::push_elements(mat.buffer32, len);

            mbt.width = width;
            mbt.height = height;

            mbt.limit = 32;
        }
        
        return true;
    }


    void destroy_mbt(MBTMatrix& mat)
    {
        mb::destroy_buffer(mat.buffer64);
        mb::destroy_buffer(mat.buffer32);
    }





    static inline void mandelbrot_at(MatrixViewMBT const& mbt, u32 i)
    {
        auto const cx = mbt.cx_[i];
        auto const cy = mbt.cy_[i];

        auto& mx = mbt.mx_[i];
        auto& my = mbt.my_[i];
        auto& mx2 = mbt.mx2_[i];
        auto& my2 = mbt.my2_[i];

        auto& iter = mbt.iter_[i];

        while (iter < mbt.limit && mx2 + my2 <= 4.0)
        {
            ++iter;
    
            my = (mx + mx) * my + cy;
            mx = mx2 - my2 + cx;
            my2 = my * my;
            mx2 = mx * mx;
        }
    }


    static inline void mandelbrot_xy(MatrixViewMBT const& mbt, u32 x, u32 y)
    {
        auto i = mbt.width * y + x;

        mandelbrot_at(mbt, i);
    }


    static inline void mandelbrot_pos_xy(MatrixViewMBT const& mbt, Vec2D<fmbt> const& cpos, u32 x, u32 y)
    {
        auto i = mbt.width * y + x;

        mbt.cx_[i] = cpos.x;
        mbt.cy_[i] = cpos.y;

        mbt.mx_[i] = 0;
        mbt.my_[i] = 0;
        mbt.mx2_[i] = 0;
        mbt.my2_[i] = 0;

        mbt.iter_[i] = 0;        

        mandelbrot_at(mbt, i);
    }
}


/* span */

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
}


/* mandelbrot processing */

namespace game_mbt
{
    class MBTData
    {
    public:
        fmbt cx = 0.0;
        fmbt cy = 0.0;
        
        fmbt mx = 0.0;
        fmbt my = 0.0;
        fmbt mx2 = 0.0;
        fmbt my2 = 0.0;

        u32 iter = 0;
        u32 limit = 32;
    };


    inline void zero_iter(MBTData& mdata)
    {
        mdata.mx = 0.0;
        mdata.my = 0.0;
        mdata.mx2 = 0.0;
        mdata.my2 = 0.0;
        mdata.iter = 0;
    }


    inline void mandelbrot_iter(MBTData& mdata)
    {
        while (mdata.iter < mdata.limit && mdata.mx2 + mdata.my2 <= 4.0)
        {
            ++mdata.iter;
    
            mdata.my = (mdata.mx + mdata.mx) * mdata.my + mdata.cy;
            //mdata.my = num::fma((mdata.mx + mdata.mx), mdata.my, mdata.cy); slower
            mdata.mx = mdata.mx2 - mdata.my2 + mdata.cx;
            mdata.my2 = mdata.my * mdata.my;
            mdata.mx2 = mdata.mx * mdata.mx;
        }
    }

}


/* load/store */

namespace game_mbt
{
    inline void load_iter_at(SpanViewMBT const& src, MBTData& mdata, u32 i)
    {
        mdata.cx = src.cx[i];
        mdata.cy = src.cy[i];

        mdata.mx = src.mx[i];
        mdata.my = src.my[i];
        mdata.mx2 = src.mx2[i];
        mdata.my2 = src.my2[i];
        mdata.iter = src.iter[i];
    }


    inline void store_iter_at(MBTData const& mdata, SpanViewMBT const& dst, u32 i)
    {
        dst.cx[i] = mdata.cx;
        dst.cy[i] = mdata.cy;
        
        dst.mx[i] = mdata.mx;
        dst.my[i] = mdata.my;
        dst.mx2[i] = mdata.mx2;
        dst.my2[i] = mdata.my2;
        dst.iter[i] = mdata.iter;
    }

} // game_mbt


/* mandelbrot row */

namespace game_mbt
{
    static void mandelbrot_row(MatrixViewMBT const& dst, u32 y, fmbt bx, fmbt by, fmbt dx)
    {
        auto row = row_span(dst, y);

        MBTData mdata{};

        mdata.cx = bx;
        mdata.cy = by;
        mdata.limit = dst.limit;

        auto len = row.length;

        for (u32 i = 0; i < len; i++)
        {
            zero_iter(mdata);
            mandelbrot_iter(mdata);
            store_iter_at(mdata, row, i);

            mdata.cx += dx;
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
            load_iter_at(row_src, mdata, i);
            mandelbrot_iter(mdata);            
            store_iter_at(mdata, row_dst, i);
        }
    }


    static void mandelbrot_span(MatrixViewMBT const& dst, u32 y, u32 x_begin, u32 x_end, fmbt bx, fmbt by, fmbt dx)
    {
        auto row = row_sub_span(dst, y, x_begin, x_end);

        MBTData mdata{};

        mdata.cx = bx;
        mdata.cy = by;
        mdata.limit = dst.limit;

        auto len = row.length;

        for (u32 i = 0; i < len; i++)
        {
            zero_iter(mdata);
            mandelbrot_iter(mdata);
            store_iter_at(mdata, row, i);

            mdata.cx += dx;
        }
    }

}


/* processing api */

namespace game_mbt
{
    void proc_copy(MBTMatrix& mat, Rect2Du32 r_src, Rect2Du32 r_dst);

    void proc_mbt(MBTMatrix& mat, u32 limit);

    void proc_mbt(MBTMatrix& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_mbt_range(MBTMatrix& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_render(MBTMatrix const& mat, img::ImageView const& screen, ColorFormat format);
    
}