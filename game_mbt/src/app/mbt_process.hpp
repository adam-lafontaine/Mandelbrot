#pragma once


/* proc */

namespace game_mbt
{
    void proc_copy(ColorMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst);
    
    void proc_mbt(ColorMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format);

    void proc_mbt_range(ColorMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit, ColorFormat format);

    void proc_render(ColorMatrix const& mat, img::ImageView const& screen);

}


/* mandelbrot */

namespace game_mbt
{
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


    class MatrixViewMBT
    {
    public:
        fmbt* cx = 0;
        fmbt* cy = 0;

        fmbt* mx = 0;
        fmbt* my = 0;

        fmbt* mx2 = 0;
        fmbt* my2 = 0;

        u32* iter = 0;

        u32 width = 0;
        u32 height = 0;

        u32 limit = 0;


        auto view_cx() { return img::make_view(cx, width, height); }
        auto view_cy() { return img::make_view(cy, width, height); }
        auto view_mx() { return img::make_view(mx, width, height); }
        auto view_my() { return img::make_view(my, width, height); }
        auto view_mx2() { return img::make_view(mx2, width, height); }
        auto view_my2() { return img::make_view(my2, width, height); }
        auto view_iter() { return img::make_view(iter, width, height); }
    };


    class MBTMatrix
    {
    private:
        u8 p = 1;
        u8 c = 0;

    public:
        MatrixViewMBT mbt_data_[2];

        MatrixViewMBT mbt_prev() const { return mbt_data_[p]; }
        MatrixViewMBT mbt_curr() const { return mbt_data_[c]; }

        void swap() { p = c; c = !p; }
    };


    static inline void mandelbrot_at(MatrixViewMBT const& mbt, u32 i)
    {
        auto const cx = mbt.cx[i];
        auto const cy = mbt.cy[i];

        auto& mx = mbt.mx[i];
        auto& my = mbt.my[i];
        auto& mx2 = mbt.mx2[i];
        auto& my2 = mbt.my2[i];

        auto& iter = mbt.iter[i];

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

        mbt.cx[i] = cpos.x;
        mbt.cy[i] = cpos.y;

        mbt.mx[i] = 0;
        mbt.my[i] = 0;
        mbt.mx2[i] = 0;
        mbt.my2[i] = 0;

        mbt.iter[i] = 0;        

        mandelbrot_at(mbt, i);
    }
}