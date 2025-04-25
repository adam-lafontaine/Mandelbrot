#pragma once


/* proc */

namespace game_mbt
{
    void proc_copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst);

    void proc_mbt(ColorIdMatrix const& mat, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_mbt_range(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format);
}


/* mandelbrot */

namespace game_mbt
{
    static inline ColorId to_color_id(u32 iter, u32 iter_limit)
    {
        constexpr auto DEF = ColorId::make_default();

        return iter >= iter_limit ? DEF : ColorId::make(iter % colors::N_COLORS);
    }


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