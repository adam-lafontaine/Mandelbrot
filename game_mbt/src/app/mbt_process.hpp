#pragma once













/* proc */

namespace game_mbt
{
    void proc_copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst);

    void proc_mbt(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit);

    void proc_render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format);
}