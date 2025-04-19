#pragma once

#include "ascii_image.hpp"


//#include <cstring>



using MaskView = img::GrayView;


namespace font_ascii_5
{
#include "ascii_5.cpp"
}


namespace font_joystick_8
{
#include "ascii_joystick_8.cpp"
}


namespace
{
    template <class ASCII>
    static MaskView make_char_mask(ASCII const& ascii, char c)
    {        
        auto id = c - ' ';

        MaskView view{};
        view.width = (u32)ascii.widths[id];
        view.height = (u32)ascii.height;
        view.matrix_data_ = (u8*)ascii.u8_pixel_data[id];

        return view;
    }


    template <class ASCII>
    static u32 count_text_width(ASCII const& ascii, StringView const& text)
    {      
        u32 width = 0;

        for (u32 i = 0; i < text.length; i++)
        {
            auto id = text.data[i] - ' ';
            width += (u32)ascii.widths[id];
        }

        return width;
    }


    static MaskView make_ascii_5_mask(char c)
    {
        return make_char_mask(font_ascii_5::ascii_chars, c);
    }


    static MaskView make_joystick_8_mask(char c)
    {
        return make_char_mask(font_joystick_8::ascii_chars, c);
    }


    
}


namespace font_ascii_factory
{
    static inline u32 font_height(ascii::Font font)
    {
        switch (font)
        {
        case ascii::Font::Ascii5:
            return font_ascii_5::ascii_chars.height;

        case ascii::Font::Joystick8:
            return font_joystick_8::ascii_chars.height;

        default:
            return 0;
        }
    }


    static inline u32 text_width(StringView const& text, ascii::Font font)
    {
        switch (font)
        {
        case ascii::Font::Ascii5:
            return count_text_width(font_ascii_5::ascii_chars, text);

        case ascii::Font::Joystick8:
            return count_text_width(font_joystick_8::ascii_chars, text);

        default:
            return 0;
        }
    }


    static inline Vec2Du32 text_dimensions(StringView const& text, ascii::Font font)
    {
        Vec2Du32 dims{};
        switch (font)
        {
        case ascii::Font::Ascii5:
            dims.x = count_text_width(font_ascii_5::ascii_chars, text);
            dims.y = font_ascii_5::ascii_chars.height;

        case ascii::Font::Joystick8:
            dims.x = count_text_width(font_joystick_8::ascii_chars, text);
            dims.y = font_joystick_8::ascii_chars.height;

        default:
            break;
        }

        return dims;
    }


    static inline auto mask_fn(ascii::Font font)
    {
        auto make_mask = make_ascii_5_mask; // default?

        switch (font)
        {
        case ascii::Font::Ascii5:
            make_mask = make_ascii_5_mask;
            break;

        case ascii::Font::Joystick8:
            make_mask = make_joystick_8_mask;
            break;

        default:
            break;
        }

        return make_mask;
    }
}


namespace ascii
{
    void render_text(StringView const& text, img::SubView const& dst, Font font, img::Pixel color)
    {
        auto mask_fn = font_ascii_factory::mask_fn(font);        

        u32 width = 0;
        u32 const height = num::min(font_ascii_factory::font_height(font), dst.height);
        i32 w_remaining = dst.width;

        Rect2Du32 d_range = {};
        d_range.y_begin = 0;
        d_range.y_end = height;

        auto const mask_char = [color](u8 mask, img::Pixel p)
        {
            return (mask) ? color : p;
        };

        for (u32 i = 0; i < text.length && w_remaining > 0; i++)
        {
            auto mask = mask_fn(text.data[i]);

            width = num::min(mask.width, (u32)w_remaining);
            d_range.x_end += width;

            auto s_view = img::sub_view(mask, img::make_rect(width, height));
            auto d_view = img::sub_view(dst, d_range);

            img::transform(s_view, d_view, mask_char);

            d_range.x_begin = d_range.x_end;
            w_remaining -= width;
        }
    }


    void render_text(StringView const& text, img::SubView const& dst, Font font, u32 desired_height, img::Pixel color)
    {
        u32 const base_height = font_ascii_factory::font_height(font);
        auto mask_fn = font_ascii_factory::mask_fn(font);

        auto const scale = num::max(desired_height / base_height, 1U);

        u32 s_width = 0;
        u32 d_width = 0;
        u32 const s_height = base_height;
        u32 const d_height = num::min(s_height * scale, dst.height);
        i32 w_remaining = (i32)dst.width;

        Rect2Du32 d_range{};
        d_range.y_begin = 0;
        d_range.y_end = d_height;

        auto const mask_char = [color](u8 mask, img::Pixel p)
        {
            return (mask) ? color : p;
        };

        for (u32 i = 0; i < text.length && w_remaining > 0; i++)
        {
            auto mask = mask_fn(text.data[i]);

            s_width = num::min(mask.width, (u32)w_remaining / scale);
            d_width = s_width * scale;

            d_range.x_end += d_width;

            auto s_view = img::sub_view(mask, img::make_rect(s_width, s_height));
            auto d_view = img::sub_view(dst, d_range);

            img::transform_scale_up(s_view, d_view, scale, mask_char);

            d_range.x_begin = d_range.x_end;
            w_remaining -= d_width;
        }


    }


    u32 font_height(Font font)
    {
        return font_ascii_factory::font_height(font);
    }


    u32 text_width(StringView const& text, Font font)
    {
        return font_ascii_factory::text_width(text, font);
    }


    u32 text_width(StringView const& text, Font font, u32 scale)
    {
        return font_ascii_factory::text_width(text, font) * scale;
    }


    Vec2Du32 text_dimensions(StringView const& text, Font font)
    {
        return font_ascii_factory::text_dimensions(text, font);
    }
}