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
    static inline ColorId to_color_id(u32 iter, u32 iter_limit)
    {   
        constexpr auto N = colors::N_COLORS;
        constexpr auto DEF = ColorId::make_default();

        if (iter >= iter_limit)
        {
            return DEF;
        }

        u32 limit = num::min(iter_limit, N / 2);
        
        u32 d = 48;// num::max(32u, limit / 2);
        u32 n = iter % d;

        return ColorId::make(N * n / d);
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