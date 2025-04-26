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
        constexpr auto DEF = ColorId::make_default();
        constexpr auto N = colors::N_COLORS;

        static_assert(num::cxpr::is_power_of_2(N));

        constexpr u32 level_base[9] = 
        {
            0,
            colors::n_level_colors(1),
            colors::n_level_colors(2),
            colors::n_level_colors(3),
            colors::n_level_colors(4),
            colors::n_level_colors(5),
            colors::n_level_colors(6),
            colors::n_level_colors(7),
            colors::n_level_colors(8),
        };

        if (iter >= iter_limit)
        {
            return DEF;
        }

        u32 n = 0;
        u32 d = 1;

        for (u32 i = 1; i < 9; i++)
        {
            if (iter < level_base[i])
            {
                n = iter - level_base[i - 1];
                d = level_base[i] - level_base[i - 1];
                break;
            }
        }

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