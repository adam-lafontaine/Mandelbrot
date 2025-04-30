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


#include <immintrin.h>

/* mandelbrot kernel simd */

namespace game_mbt
{



} // game_mbt



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


/* simd avx2 */

namespace game_mbt
{

/* mandelbrot processing */

namespace avx2
{
    using d256 = __m256d;
    using i256 = __m256i;


    class MBTData256
    {
    public:
        d256 cx;
        d256 cy;
        
        d256 mx;
        d256 my;
        d256 mx2;
        d256 my2;

        i256 iter;

        u32 limit = 32;
    };


    inline void zero_iter(MBTData256& mdata)
    {
        mdata.mx = _mm256_setzero_pd();
        mdata.my = _mm256_setzero_pd();
        mdata.mx2 = _mm256_setzero_pd();
        mdata.my2 = _mm256_setzero_pd();
        mdata.iter = _mm256_setzero_si256();
    }


    inline void mandelbrot_iter(MBTData256& mdata)
    {
        auto active = _mm256_set1_epi64x(1); 

        auto test_active = [&]()
        {
            // mdata.iter < mdata.limit
            auto cmp_iter = _mm256_cmpgt_epi64(_mm256_set1_epi64x(mdata.limit) , mdata.iter);
            active = _mm256_andnot_si256(cmp_iter, active);

            // mdata.mx2 + mdata.my2 <= 4.0
            auto sum = _mm256_add_pd(mdata.mx2, mdata.my2);
            auto val = _mm256_cmp_pd(sum, _mm256_set1_pd(4.0), _CMP_LE_OQ);
            auto cmp_sum = _mm256_castpd_si256(val);
            active = _mm256_andnot_si256(cmp_sum, active);

            return _mm256_testz_si256(active, active);
        };

        auto inc_iter = [&]()
        {
            // ++mdata.iter;
            auto inc = _mm256_and_si256(active, _mm256_set1_epi64x(1));
            mdata.iter = _mm256_add_epi64(mdata.iter, inc);
        };        

        while (test_active())
        {   
            inc_iter();

            // mdata.my = (mdata.mx + mdata.mx) * mdata.my + mdata.cy;
            auto val = _mm256_add_pd(mdata.mx, mdata.mx);
            val = _mm256_fmadd_pd(val, mdata.my, mdata.cy);

            // mdata.mx = mdata.mx2 - mdata.my2 + mdata.cx;
            val = _mm256_sub_pd(mdata.mx2, mdata.my2);
            mdata.mx = _mm256_add_pd(val, mdata.cx);

            // mdata.my2 = mdata.my * mdata.my;
            // mdata.mx2 = mdata.mx * mdata.mx;
            mdata.mx2 = _mm256_mul_pd(mdata.mx, mdata.mx);
            mdata.my2 = _mm256_mul_pd(mdata.my, mdata.my);
        }
    }
    
} // avx2


/* load/store */

namespace avx2
{
    inline void load_iter_at(SpanViewMBT const& src, MBTData256& mdata, u32 i)
    {
        mdata.cx = _mm256_load_pd(src.cx + i);
        mdata.cy = _mm256_load_pd(src.cy + i);

        mdata.mx = _mm256_load_pd(src.mx + i);
        mdata.my = _mm256_load_pd(src.my + i);
        mdata.mx2 = _mm256_load_pd(src.mx2 + i);
        mdata.my2 = _mm256_load_pd(src.my2 + i);

        mdata.iter = _mm256_set_epi64x(src.iter[i], src.iter[i + 1], src.iter[i + 2], src.iter[i + 3]);
    }


    inline void store_iter_at(MBTData256 const& mdata, SpanViewMBT const& dst, u32 i)
    {
        _mm256_store_pd(dst.cx + i, mdata.cx);
        _mm256_store_pd(dst.cy + i, mdata.cy);

        _mm256_store_pd(dst.mx + i, mdata.mx);
        _mm256_store_pd(dst.my + i, mdata.my);
        _mm256_store_pd(dst.mx2 + i, mdata.mx2);
        _mm256_store_pd(dst.my2 + i, mdata.my2);

        i64 iter[4] = { dst.iter[i], dst.iter[i + 1], dst.iter[i + 2], dst.iter[i + 3] };
        auto mem256 = (i256*)iter;
        _mm256_storeu_si256(mem256, mdata.iter);
    }
    
} // avx2


/* mandelbrot row */

namespace avx2
{
    static void mandelbrot_row(MatrixViewMBT const& dst, u32 y, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta)
    {
        /*uto row = row_span(dst, y);

        auto bx = begin.x;
        auto dx = delta.x;

        f64 cx64[4] = { bx, bx + dx, bx + 2 * dx, bx + 3 * dx };

        MBTData256 mdata{};
        
        mdata.cx = _mm256_load_pd(cx64);
        mdata.cy = begin.y;
        mdata.limit = dst.limit;

        auto len = row.length;

        for (u32 i = 0; i < len; i++)
        {
            zero_iter(mdata);
            mandelbrot_iter(mdata);
            store_iter_at(mdata, row, i);

            mdata.cx += delta.x;
        }*/
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
            store_iter_at(mdata, row, i);

            mdata.cx += delta.x;
        }
    }
    
} // avx2

} // game_game_mbt


/* processing api */

namespace game_mbt
{
    void proc_copy(MBTMatrix& mat, Rect2Du32 r_src, Rect2Du32 r_dst);

    void proc_mbt(MBTMatrix& mat, u32 limit);

    void proc_mbt(MBTMatrix& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_mbt_range(MBTMatrix& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_render(MBTMatrix const& mat, img::ImageView const& screen, ColorFormat format);
    
}