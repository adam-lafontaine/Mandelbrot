#pragma once

#include "../image/image.hpp"
#include "../span/span.hpp"
#include "../util/numeric.hpp"

namespace num = numeric;
namespace img = image;


namespace ascii
{
    enum class Font : int
    {
        Ascii5,
        Joystick8
    };
    

    void render_text(StringView const& text, img::SubView const& dst, Font font, u32 desired_height, img::Pixel color);

    void render_text(StringView const& text, img::SubView const& dst, Font font, img::Pixel color);

    u32 font_height(Font font);

    u32 text_width(StringView const& text, Font font);

    u32 text_width(StringView const& text, Font font, u32 scale);

    Vec2Du32 text_dimensions(StringView const& text, Font font);    


    inline void render_text(cstr text, img::SubView const& dst, Font font, u32 desired_height, img::Pixel color)
    {
        render_text(span::to_string_view(text), dst, font, desired_height, color);
    }


    inline void render_text(cstr text, img::SubView const& dst, Font font, img::Pixel color)
    {
        render_text(span::to_string_view(text), dst, font, color);
    }

    
    inline u32 text_width(cstr text, Font font)
    {
        return text_width(span::to_string_view(text), font);
    }


    inline Vec2Du32 text_dimensions(cstr text, Font font)
    {
        return text_dimensions(span::to_string_view(text), font);
    }
    

}


namespace ascii
{
    /*inline Rect2Du32 rect_fit_w()
    {
        auto dims = ascii::text_dimensions(text, font);
        auto scale = (view.width - 2 * pad) / dims.x;
        auto width = dims.x * scale;
        auto height = dims.y * scale;
        auto x = (view.width - width) / 2; // centered
        auto y = (view.height - height) / 2;
    }*/

    
    inline Vec2Du32 text_dimensions(auto const& text, Font font, u32 scale)
    {
        auto dims = text_dimensions(text, font);
        dims.x *= scale;
        dims.y *= scale;

        return dims;
    }
    
    
    inline Vec2Du32 text_dimensions_max(auto text, auto const& view, Font font, u32 pad)
    {
        auto dims = text_dimensions(text, font);
        auto w = view.width - 2 * pad;
        auto h = view.height - 2 * pad;

        assert(w >= dims.x);
        assert(h >= dims.y);

        auto scale = num::min(w / dims.x, h / dims.y);
        dims.x *= scale;
        dims.y *= scale;

        return dims;
    }
    
    
    inline void render_text_fit(auto text, auto const& view, Font font, u32 pad, img::Pixel color)
    {
        auto dims = text_dimensions(text, font);
        auto scale = (view.width - 2 * pad) / dims.x;
        auto width = dims.x * scale;
        auto height = dims.y * scale;
        auto x = (view.width - width) / 2; // centered
        auto y = (view.height - height) / 2;
        auto dst = img::sub_view(view, img::make_rect(x, y, width, height));
        render_text(text, dst, font, height, color);
    }
}