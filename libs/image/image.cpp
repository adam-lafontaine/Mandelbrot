#pragma once

#include "image.hpp"
#include "../util/numeric.hpp"

#include "../stb_libs/stb_image_options.hpp"

namespace image
{
    namespace sp = span;
    namespace num = numeric;
}


/* create/destroy */

namespace image
{
    bool create_image(Image& image, u32 width, u32 height)
    {
        auto data = mem::alloc<Pixel>(width * height, "create_image");
        if (!data)
        {
            return false;
        }

        image.data_ = data;
        image.width = width;
        image.height = height;

        return true;
    }

    
    void destroy_image(Image& image)
    {
        if (image.data_)
		{
			mem::free(image.data_);
			image.data_ = nullptr;
		}

		image.width = 0;
		image.height = 0;
    }


    bool create_image(ImageGray& image, u32 width, u32 height)
    {
        auto data = mem::alloc<u8>(width * height, "create_image");
        if (!data)
        {
            return false;
        }

        image.data_ = data;
        image.width = width;
        image.height = height;

        return true;
    }

    
    void destroy_image(ImageGray& image)
    {
        if (image.data_)
		{
			mem::free(image.data_);
			image.data_ = nullptr;
		}

		image.width = 0;
		image.height = 0;
    }
}


/* row_begin */

namespace image
{
    template <typename T>
    static inline T* row_begin(MatrixView2D<T> const& view, u32 y)
    {
        return view.matrix_data_ + (u64)y * view.width;
    }


    template <typename T>
    static inline T* row_begin(MatrixSubView2D<T> const& view, u32 y)
    {
        return view.matrix_data_ + (u64)(view.y_begin + y) * view.matrix_width + view.x_begin;
    }
}


/* xy_at */

namespace image
{
    template <typename T>
    static inline T* xy_at(MatrixView2D<T> const& view, u32 x, u32 y)
    {
        return row_begin(view, y) + x;
    }


    template <typename T>
    static inline T* xy_at(MatrixSubView2D<T> const& view, u32 x, u32 y)
    {
        return row_begin(view, y) + x;
    }
}


/* row_span */

namespace image
{
    template <typename T>
	static inline SpanView<T> row_span(MatrixView2D<T> const& view, u32 y)
	{
        SpanView<T> span{};

        span.data = view.matrix_data_ + (u64)y * view.width;
        span.length = view.width;

        return span;
	}


    template <typename T>
    static inline SpanView<T> row_span(MatrixSubView2D<T> const& view, u32 y)
    {
        SpanView<T> span{};

        span.data = view.matrix_data_ + (u64)(view.y_begin + y) * view.matrix_width + view.x_begin;
        span.length = view.width;

        return span;
    }


    template <typename T>
    static inline SpanView<T> sub_span(MatrixView2D<T> const& view, u32 y, u32 x_begin, u32 x_end)
    {
        SpanView<T> span{};

        span.data = view.matrix_data_ + (u64)(y * view.width) + x_begin;
        span.length = x_end - x_begin;

        return span;
    }


    template <typename T>
    static inline SpanView<T> sub_span(MatrixSubView2D<T> const& view, u32 y, u32 x_begin, u32 x_end)
    {
        SpanView<T> span{};

        span.data = view.matrix_data_ + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin) + x_begin;
        span.length = x_end - x_begin;

        return span;
    }
}


/* make_view */

namespace image
{
    ImageView make_view(u32 width, u32 height, Pixel* data)
    {
        ImageView view{};

        view.width = width;
        view.height = height;
        view.matrix_data_ = data;

        return view;
    }


    GrayView make_view(u32 width, u32 height, u8* data)
    {
        GrayView view{};

        view.width = width;
        view.height = height;
        view.matrix_data_ = data;

        return view;
    }


    ImageView make_view(Image const& image)
    {
        return make_view(image.width, image.height, image.data_);
    }


    GrayView make_view(ImageGray const& image)
    {
        return make_view(image.width, image.height, image.data_);
    }


    ImageView make_view(u32 width, u32 height, Buffer32& buffer)
    {
        ImageView view{};

        view.matrix_data_ = mb::push_elements(buffer, width * height);
        if (view.matrix_data_)
        {
            view.width = width;
            view.height = height;
        }

        return view;
    }


    GrayView make_view(u32 width, u32 height, Buffer8& buffer)
    {
        GrayView view{};

        view.matrix_data_ = mb::push_elements(buffer, width * height);
        if (view.matrix_data_)
        {
            view.width = width;
            view.height = height;
        }

        return view;
    }
}


/* pixel_at */

namespace image
{
    Pixel pixel_at(ImageView const& view, u32 x, u32 y)
    {
        return *xy_at(view, x, y);
    }


    u8 pixel_at(GrayView const& view, u32 x, u32 y)
    {
        return *xy_at(view, x, y);
    }


    u8 pixel_at(GraySubView const& view, u32 x, u32 y)
    {
        return *xy_at(view, x, y);
    }
}


/* alpha_blend */

namespace image
{  
    static inline void alpha_blend(Pixel src, Pixel* dst, f32 alpha)
    {        
        auto const i = 1.0f - alpha;

        auto& d = *dst;
        d.red = num::round_to_unsigned<u8>(num::fmaf(alpha, src.red, i * d.red) /* a * s.red + i * d.red */);
        d.green = num::round_to_unsigned<u8>(num::fmaf(alpha, src.green, i * d.green) /* a * s.green + i * d.green */);
        d.blue = num::round_to_unsigned<u8>(num::fmaf(alpha, src.blue, i * d.blue) /* a * s.blue + i * d.blue */);
    }


    static inline void alpha_blend(Pixel* src, Pixel* dst, f32 alpha)
    {
        alpha_blend(*src, dst, alpha);
    }


    static void alpha_blend(Pixel s, Pixel* dst)
    {
        constexpr auto scale = 1.0f / 255.0f;

        f32 a = 1.0f;

        switch (s.alpha)
        {
        case 0: 
            return;

        case 255: 
            *dst = s;            
            return;

        case 126:
        case 127:
        case 128:
        case 129:
            a = 0.5f;
            break;

        default: 
            a = s.alpha * scale;
            break;
        }

        alpha_blend(s, dst, a);
    }    


    static void alpha_blend(Pixel* src, Pixel* dst)
    {
        alpha_blend(*src, dst);
    }


    static void alpha_blend_span(SpanView<Pixel> const& src, SpanView<Pixel> const& dst)
    {
        for (u32 i = 0; i < dst.length; ++i) // TODO: simd
        {
            alpha_blend(src.data + i, dst.data + i);
        }
    }


    static void alpha_blend_span(SpanView<Pixel> const& src, SpanView<Pixel> const& dst, f32 alpha)
    {
        for (u32 i = 0; i < dst.length; ++i) // TODO: simd
        {
            alpha_blend(src.data + i, dst.data + i, alpha);
        }
    }
}


/* fill */

namespace image
{
    template <typename T>
    static void fill_span_if(SpanView<T> const& dst, u8 value, fn<bool(T)> const& pred)
    {
        T* d = dst.data;
        for (u32 i = 0; i < dst.length; ++i)
		{
			d[i] = pred(d[i]) ? value : d[i];
		}
    }


    static void fill_span_blend(SpanView<Pixel> const& dst, Pixel value)
    {
        Pixel* d = dst.data;

        for (u32 i = 0; i < dst.length; ++i)
        {
            alpha_blend(value, d + i); // TODO: simd
        }
    }


    void fill(ImageView const& view, Pixel color)
    {
        assert(view.matrix_data_);

        sp::fill_32(to_span(view), color);
    }


    void fill(SubView const& view, Pixel color)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);

        for (u32 y = 0; y < view.height; y++)
        {
            sp::fill_32(row_span(view, y), color);
        }
    }


    void fill_if(GraySubView const& view, u8 gray, fn<bool(u8)> const& pred)
    {
        assert(view.matrix_data_);

        for (u32 y = 0; y < view.height; y++)
        {
            fill_span_if(row_span(view, y), gray, pred);
        }
    }


    void fill_blend(SubView const& view, Pixel color)
    {
        for (u32 y = 0; y < view.height; y++)
        {
            fill_span_blend(row_span(view, y), color);
        }
    }


    void fill_row(SubView const& view, u32 y, Pixel color)
    {
        sp::fill_32(row_span(view, y), color);
    }


    void fill_row_blend(SubView const& view, u32 y, Pixel color)
    {
        fill_span_blend(row_span(view, y), color);
    }
}


/* copy */

namespace image
{
    template <class VIEW_S, class VIEW_D>
    static void copy_view(VIEW_S const& src, VIEW_D const& dst)
    {
        sp::copy(to_span(src), to_span(dst));
    }


    template <class VIEW_S, class VIEW_D>
    static void copy_sub_view(VIEW_S const& src, VIEW_D const& dst)
    {
        for (u32 y = 0; y < src.height; y++)
        {
            sp::copy(row_span(src, y), row_span(dst, y));
        }
    }


    template <typename V_SRC, typename V_DST>
    static void copy_blend_sub_view(V_SRC const& src, V_DST const& dst)
    {
        for (u32 y = 0; y < src.height; y++)
        {
            alpha_blend_span(row_span(src, y), row_span(dst, y));
        }
    }


    void copy(ImageView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_view(src, dst);
    }    


    void copy(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_sub_view(src, dst);
    }


    void copy(SubView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_sub_view(src, dst);
    }


    void copy(SubView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_sub_view(src, dst);
    }


    void copy_blend(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_blend_sub_view(src, dst);
    }


    void copy_blend(SubView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_blend_sub_view(src, dst);
    }


    void copy_blend(SubView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        copy_blend_sub_view(src, dst);
    }


    void copy_blend(ImageView const& src, SubView const& dst, u8 alpha)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        constexpr auto scale = 1.0f / 255.0f;

        f32 a = 1.0f;

        switch (alpha)
        {
        case 0: 
            return;

        case 255:
            copy(src, dst);
            return;

        case 127:
        case 128:
            a = 0.5f;
            break;

        default: 
            a = alpha * scale;
            break;
        }

        for (u32 y = 0; y < src.height; y++)
        {
            alpha_blend_span(row_span(src, y), row_span(dst, y), a);
        }
    }
}


/* for_each_pixel */

namespace image
{   
    template <typename T>
    static inline void for_each_in_span(SpanView<T> const& span, fn<void(T)> const& func)
    {
        for (u32 i = 0; i < span.length; i++)
        {
            func(span.data[i]);
        }
    }


    void for_each_pixel(ImageView const& view, fn<void(Pixel)> const& func)
    {
        assert(view.matrix_data_);

        for_each_in_span(to_span(view), func);
    }


    void for_each_pixel(SubView const& view, fn<void(Pixel)> const& func)
    {
        assert(view.matrix_data_);

        for (u32 y = 0; y < view.height; y++)
        {
            for_each_in_span(row_span(view, y), func);
        }
    }
}


/* for_each_xy */

namespace image
{
    template <class VIEW, class FN>
    static void for_each_xy_write(VIEW const& view, FN const& xy_func)
    {
        for (u32 y = 0; y < view.height; ++y)
        {
            auto d = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
            {
                d[x] = xy_func(x, y);
            }
        }
    }


    template <class VIEW, class FN>
    static void for_each_xy_exe(VIEW const& view, FN const& xy_func)
    {
        for (u32 y = 0; y < view.height; ++y)
        {
			for (u32 x = 0; x < view.width; ++x)
            {
                xy_func(x, y);
            }
        }
    }


    void for_each_xy(ImageView const& view, fn<Pixel(u32 x, u32 y)> const& xy_func)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        
        for_each_xy_write(view, xy_func);
    }


    void for_each_xy(SubView const& view, fn<Pixel(u32 x, u32 y)> const& xy_func)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        
        for_each_xy_write(view, xy_func);
    }


    void for_each_xy(ImageView const& view, fn<void(u32 x, u32 y)> const& xy_func)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        
        for_each_xy_exe(view, xy_func);
    }


    void for_each_xy(GrayView const& view, fn<u8(u32 x, u32 y)> const& xy_func)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        
        for_each_xy_write(view, xy_func);
    }
}


/* transform static */

namespace image
{
    template <typename P_SRC, typename P_DST>
    static void transform_span(SpanView<P_SRC> const& src, SpanView<P_DST> const& dst, fn<P_DST(P_SRC, P_DST)> const& func)
    {
        auto s = src.data;
        auto d = dst.data;

        for (u32 i = 0; i < src.length; i++)
        {
            d[i] = func(s[i], d[i]);
        }
    }
    

    template <typename P_SRC, typename P_DST>
    static void transform_span(SpanView<P_SRC> const& src, SpanView<P_DST> const& dst, fn<P_DST(P_SRC)> const& func)
    {
        auto s = src.data;
        auto d = dst.data;

        for (u32 i = 0; i < src.length; i++)
        {
            d[i] = func(s[i]);
        }
    }


    template <class V_SRC, class V_DST, class FUNC>
    static void transform_view(V_SRC const& src, V_DST const& dst, FUNC const& func)
    {
        transform_span(to_span(src), to_span(dst), func);
    }


    template <class V_SRC, class V_DST, class FUNC>
    static void transform_sub_view(V_SRC const& src, V_DST const& dst, FUNC const& func)
    {
        for (u32 y = 0; y < src.height; y++)
        {
            transform_span(row_span(src, y), row_span(dst, y), func);
        }
    }

}


/* transform api */

namespace image
{   
    
    void transform(ImageView const& src, SubView const& dst, fn<Pixel(Pixel, Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform(SubView const& src, SubView const& dst, fn<Pixel(Pixel, Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform(GrayView const& src, SubView const& dst, fn<Pixel(u8, Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform(GraySubView const& src, SubView const& dst, fn<Pixel(u8, Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform(ImageView const& src, GrayView const& dst, fn<u8(Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_view(src, dst, func);
    }


    void transform(ImageView const& src, GraySubView const& dst, fn<u8(Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }    


    void transform(GrayView const& src, ImageView const& dst, fn<Pixel(u8)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_view(src, dst, func);
    }


    void transform(GrayView const& src, SubView const& dst, fn<Pixel(u8)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform(GraySubView const& src, ImageView const& dst, fn<Pixel(u8)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform(GraySubView const& src, SubView const& dst, fn<Pixel(u8)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        transform_sub_view(src, dst, func);
    }


    void transform_scale_up(GraySubView const& src, SubView const& dst, u32 scale, fn<Pixel(u8, Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width * scale);
        assert(dst.height == src.height * scale);

        for (u32 src_y = 0; src_y < src.height; src_y++)
        {
            auto s = row_begin(src, src_y);

            for (u32 src_x = 0; src_x < src.width; src_x++)
            {
                auto const src_value = s[src_x];

                auto dst_y = src_y * scale;
                for (u32 offset_y = 0; offset_y < scale; offset_y++, dst_y++)
                {
                    auto d = row_begin(dst, dst_y);

                    auto dst_x = src_x * scale;
                    for (u32 offset_x = 0; offset_x < scale; offset_x++, dst_x++)
                    {
                        d[dst_x] = func(src_value, d[dst_x]);
                    }                    
                }
            }
        }
    }


    void transform_scale_up(ImageView const& src, GrayView const& dst, u32 scale, fn<u8(Pixel)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width * scale);
        assert(dst.height == src.height * scale);

        for (u32 src_y = 0; src_y < src.height; src_y++)
        {
            auto s = row_begin(src, src_y);

            for (u32 src_x = 0; src_x < src.width; src_x++)
            {
                auto const value = func(s[src_x]);

                auto dst_y = src_y * scale;
                for (u32 offset_y = 0; offset_y < scale; offset_y++, dst_y++)
                {
                    auto d = row_begin(dst, dst_y);

                    auto dst_x = src_x * scale;
                    for (u32 offset_x = 0; offset_x < scale; offset_x++, dst_x++)
                    {
                        d[dst_x] = value;
                    }
                }
            }
        }
    }
}


/* circle unsafe */

namespace image
{
namespace draw
{
    template <class VIEW, typename P>
    static void circle_outline_unsafe(VIEW const& view, Point2Di32 center, u32 radius, P color)
    {
        static_assert(sizeof(P) == sizeof(view.matrix_data_[0]));

        i32 const cx = center.x;
        i32 const cy = center.y;

        assert(view.width >= 2 * radius);
        assert(view.height >= 2 * radius);
        assert(cx >= radius);
        assert(cx <= view.width - radius);
        assert(cy >= radius);
        assert(cy <= view.height - radius);

        i32 ox = radius;
        i32 oy = 0;
        i32 dy = -2;
        i32 dx = 4 * (radius - 1);
        i32 D = 2 * radius - 1;
        i32 mask = 0;

        i32 y = 0;
        i32 x_min = 0;
        i32 x_max = 0;

        auto row = row_span(view, 0);

        auto const write = [&]()
        {
            row = row_span(view, y);
            row.data[x_min] = row.data[x_max] = color;
        };

        while (oy <= ox)
        {         
            y = cy - oy;
            x_min = cx - ox;
            x_max = cx + ox;
            write();

            y = cy + oy;
            write();

            y = cy - ox;
            x_min = cx - oy;
            x_max = cx + oy;
            write();

            y = cy + ox;
            write();

            D += dy;
            dy -= 4;
            ++oy;

            mask = (D >> 31);
            D += dx & mask;
            dx -= 4 & mask;
            ox += mask;
        }
    }


    template <class VIEW, typename P>
    static void circle_fill_unsafe(VIEW const& view, Point2Di32 center, u32 radius, P color)
    {
        static_assert(sizeof(P) == sizeof(view.matrix_data_[0]));

        i32 const cx = center.x;
        i32 const cy = center.y;

        assert(view.width >= 2 * radius);
        assert(view.height >= 2 * radius);
        assert(cx >= radius);
        assert(cx <= view.width - radius);
        assert(cy >= radius);
        assert(cy <= view.height - radius);

        i32 ox = radius;
        i32 oy = 0;
        i32 dy = -2;
        i32 dx = 4 * (radius - 1);
        i32 D = 2 * radius - 1;
        i32 mask = 0;

        i32 y = 0;
        i32 x_begin = 0;
        i32 x_end = 0;

        while (oy <= ox)
        {      
            y = cy - oy;
            x_begin = cx - ox;
            x_end = cx + ox + 1;
            fill_span(sub_span(view, y, x_begin, x_end), color);

            y = cy + oy;
            fill_span(sub_span(view, y, x_begin, x_end), color);

            y = cy - ox;
            x_begin = cx - oy;
            x_end = cx + oy + 1;
            fill_span(sub_span(view, y, x_begin, x_end), color);

            y = cy + ox;
            fill_span(sub_span(view, y, x_begin, x_end), color);

            D += dy;
            dy -= 4;
            ++oy;

            mask = (D >> 31);
            D += dx & mask;
            dx -= 4 & mask;
            ox += mask;
        }
    }
}
}


/* circle */

namespace image
{
namespace draw
{
    template <class VIEW, typename P>
    static void circle_outline_safe(VIEW const& view, Point2Di32 center, u32 radius, P color)
    {
        static_assert(sizeof(P) == sizeof(view.matrix_data_[0]));

        constexpr i32 oD = 0; // offset to correct diameter

        i32 const cx = center.x;
        i32 const cy = center.y;

        i32 const width = view.width;
        i32 const height = view.height;

        i32 ox = radius;
        i32 oy = 0;
        i32 dy = -2;
        i32 dx = 4 * (radius - 1);
        i32 D = 2 * radius - 1;
        i32 mask = 0;

        i32 y = 0;
        i32 x_min = 0;
        i32 x_max = 0;       

        auto row = row_span(view, 0);

        auto const write_at_x = [&](i32 x)
        {
            if (x >= 0 && x < width)
            {
                row.data[x] = color;
            }
        };

        auto const write = [&]()
        {
            if (y >= 0 & y < height)
            {
                row = row_span(view, y);
                write_at_x(x_min);
                write_at_x(x_max);
            }
        };

        while (oy <= ox)
        {         
            y = cy - oy;
            x_min = cx - ox;
            x_max = cx + ox + oD;
            write();

            y = cy + oy + oD;
            write();

            y = cy - ox;
            x_min = cx - oy;
            x_max = cx + oy + oD;
            write();

            y = cy + ox + oD;
            write();

            D += dy;
            dy -= 4;
            ++oy;

            mask = (D >> 31);
            D += dx & mask;
            dx -= 4 & mask;
            ox += mask;
        }
    }


    template <class VIEW, typename P>
    static void circle_fill_safe(VIEW const& view, Point2Di32 center, u32 radius, P color)
    {
        static_assert(sizeof(P) == sizeof(view.matrix_data_[0]));

        constexpr i32 oD = 0; // offset to correct diameter

        i32 const cx = center.x;
        i32 const cy = center.y;

        i32 const width = view.width;
        i32 const height = view.height;

        i32 ox = radius;
        i32 oy = 0;
        i32 dy = -2;
        i32 dx = 4 * (radius - 1);
        i32 D = 2 * radius - 1;
        i32 mask = 0;

        i32 y = 0;
        i32 x_begin = 0;
        i32 x_end = 0;

        auto const write = [&]()
        {
            if (y >= 0 & y < height && x_end > 0 && x_begin < width)
            {
                x_begin = num::clamp(x_begin, 0, width - 1);
                x_end = num::clamp(x_end, 1, width);
                sp::fill(sub_span(view, y, x_begin, x_end), color);
            }
        };

        while (oy <= ox)
        {           
            y = cy - oy;
            x_begin = cx - ox;
            x_end = cx + ox + 1 + oD;
            write();

            y = cy + oy + oD;
            write();

            y = cy - ox;
            x_begin = cx - oy;
            x_end = cx + oy + 1 + oD;
            write();

            y = cy + ox + oD;
            write();

            D += dy;
            dy -= 4;
            ++oy;

            mask = (D >> 31);
            D += dx & mask;
            dx -= 4 & mask;
            ox += mask;
        }
    }


    void circle_outline(ImageView const& view, Point2Di32 center, u32 radius, Pixel color)
    {
        circle_outline_safe(view, center, radius, color);
    }


    void circle_outline(SubView const& view, Point2Di32 center, u32 radius, Pixel color)
    {
        circle_outline_safe(view, center, radius, color);
    }


    void circle_fill(ImageView const& view, Point2Di32 center, u32 radius, Pixel color)
    {
        circle_fill_safe(view, center, radius, color);
    }


    void circle_fill(SubView const& view, Point2Di32 center, u32 radius, Pixel color)
    {
        circle_fill_safe(view, center, radius, color);
    }
}
}


/* rotate static */

namespace image
{
    template <class VIEW_S, class VIEW_D>
    static void rotate_view_90(VIEW_S const& src, VIEW_D const& dst)
    {
        auto const dw = dst.width;
        auto const dh = dst.height;

        for (u32 dy = 0; dy < dh; dy++)
        {
            auto const sx = dy;

            auto d = row_begin(dst, dy);            
            for (u32 dx = 0; dx < dw; dx++)
            {
                auto sy = dw - 1 - dx;
                d[dx] = *xy_at(src, sx, sy);
            }
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void rotate_view_180(VIEW_S const& src, VIEW_D const& dst)
    {
        auto const dw = dst.width;
        auto const dh = dst.height;

        for (u32 dy = 0; dy < dh; dy++)
        {
            auto const sy = dh - 1 - dy;

            auto d = row_begin(dst, dy);
            auto s = row_begin(src, sy);
            for (u32 dx = 0; dx < dw; dx++)
            {
                auto sx = dw - 1 - dx;
                d[dx] = s[sx];
            }
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void rotate_view_270(VIEW_S const& src, VIEW_D const& dst)
    {
        auto const dw = dst.width;
        auto const dh = dst.height;

        for (u32 dy = 0; dy < dh; dy++)
        {
            auto const sx = dh - 1 - dy;

            auto d = row_begin(dst, dy);
            for (u32 dx = 0; dx < dw; dx++)
            {
                auto sy = dx;
                d[dx] = *xy_at(src, sx, sy);
            }
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void rotate_view_any(VIEW_S const& src, VIEW_D const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, f32 cos, f32 sin)
    {
        u32 const sw = src.width;
        u32 const sh = src.height;

        i32 const spx = src_pivot.x;
        i32 const spy = src_pivot.y;

        i32 const dpx = dst_pivot.x;
        i32 const dpy = dst_pivot.y;
        
        f32 dysin = spx - dpy * sin - dpx * cos - cos;
        f32 dycos = spy - dpy * cos + dpx * sin + sin;

        u32 sx = 0;
        u32 sy = 0;

        f32 sxf = 0.0f;
        f32 syf = 0.0f;

        for (u32 y = 0; y < dst.height; y++)
        {
            auto d = row_begin(dst, y);            
            
            sxf = dysin;
            syf = dycos;

            for (u32 x = 0; x < dst.width; x++)
            { 
                sxf += cos;
                syf -= sin;

                if (sxf < 0.0f || syf < 0.0f)
                {
                    continue;
                }

                sx = num::round_to_unsigned<u32>(sxf);
                sy = num::round_to_unsigned<u32>(syf);

                if (sx >= sw || sy >= sh)
                {
                    continue;
                }

                d[x] = *xy_at(src, sx, sy);
            }

            dysin += sin;
            dycos += cos;
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void rotate_view_any(VIEW_S const& src, VIEW_D const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, uangle rot)
    {
        auto const cos = num::cos(rot);
        auto const sin = num::sin(rot);

        rotate_view_any(src, dst, src_pivot, dst_pivot, cos, sin);
    }    
}


/* rotate_blend static */

namespace image
{
    template <class VIEW_S, class VIEW_D>
    static void rotate_blend_any(VIEW_S const& src, VIEW_D const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, f32 cos, f32 sin)
    {
        u32 const sw = src.width;
        u32 const sh = src.height;

        i32 const spx = src_pivot.x;
        i32 const spy = src_pivot.y;

        i32 const dpx = dst_pivot.x;
        i32 const dpy = dst_pivot.y;
        
        f32 dysin = spx - dpy * sin - dpx * cos - cos;
        f32 dycos = spy - dpy * cos + dpx * sin + sin;
        
        u32 sx = 0;
        u32 sy = 0;

        f32 sxf = 0.0f;
        f32 syf = 0.0f;

        for (u32 y = 0; y < dst.height; y++)
        {
            auto d = row_begin(dst, y);
            
            sxf = dysin;
            syf = dycos;
            
            for (u32 x = 0; x < dst.width; x++)
            {
                sxf += cos;
                syf -= sin;

                if (sxf < 0.0f || syf < 0.0f)
                {
                    continue;
                }

                sx = num::round_to_unsigned<u32>(sxf);
                sy = num::round_to_unsigned<u32>(syf);

                if (sx >= sw || sy >= sh)
                {
                    continue;
                }

                alpha_blend(xy_at(src, sx, sy), d + x);
            }

            dysin += sin;
            dycos += cos;
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void rotate_blend_any(VIEW_S const& src, VIEW_D const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, uangle rot)
    {
        auto const cos = num::cos(rot);
        auto const sin = num::sin(rot);

        rotate_blend_any(src, dst, src_pivot, dst_pivot, cos, sin);
    }


    template <class VIEW_S, class VIEW_D, class FUNC>
    static void rotate_blend_transform_any(
        VIEW_S const& src, VIEW_D const& dst, 
        Point2Di32 src_pivot, Point2Di32 dst_pivot, 
        f32 cos, f32 sin, 
        FUNC const& func)
    {
        u32 const sw = src.width;
        u32 const sh = src.height;

        i32 const spx = src_pivot.x;
        i32 const spy = src_pivot.y;

        i32 const dpx = dst_pivot.x;
        i32 const dpy = dst_pivot.y;
        
        f32 dysin = spx - dpy * sin - dpx * cos - cos;
        f32 dycos = spy - dpy * cos + dpx * sin + sin;
        
        u32 sx = 0;
        u32 sy = 0;

        f32 sxf = 0.0f;
        f32 syf = 0.0f;

        for (u32 y = 0; y < dst.height; y++)
        {
            auto d = row_begin(dst, y);
            
            sxf = dysin;
            syf = dycos;
            
            for (u32 x = 0; x < dst.width; x++)
            {
                sxf += cos;
                syf -= sin;

                if (sxf < 0.0f || syf < 0.0f)
                {
                    continue;
                }

                sx = num::round_to_unsigned<u32>(sxf);
                sy = num::round_to_unsigned<u32>(syf);

                if (sx >= sw || sy >= sh)
                {
                    continue;
                }

                auto sp = *xy_at(src, sx, sy);
                auto fp = func(sp);

                alpha_blend(fp, d + x);
            }

            dysin += sin;
            dycos += cos;
        }
    }
}


/* rotate 90 */

namespace image
{
    void rotate_90(ImageView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.height);
        assert(src.height == dst.width);

        rotate_view_90(src, dst);
    }


    void rotate_90(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.height);
        assert(src.height == dst.width);

        rotate_view_90(src, dst);
    }


    void rotate_90(SubView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.height);
        assert(src.height == dst.width);

        rotate_view_90(src, dst);
    }


    void rotate_180(ImageView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        rotate_view_180(src, dst);
    }


    void rotate_180(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        rotate_view_180(src, dst);
    }


    void rotate_180(SubView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width);
        assert(dst.height == src.height);

        rotate_view_180(src, dst);
    }


    void rotate_270(ImageView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.height);
        assert(src.height == dst.width);

        rotate_view_270(src, dst);
    }


    void rotate_270(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.height);
        assert(src.height == dst.width);

        rotate_view_270(src, dst);
    }


    void rotate_270(SubView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.height);
        assert(src.height == dst.width);

        rotate_view_270(src, dst);
    }
    
}


/* rotate */

namespace image
{
    void rotate(GrayView const& src, GrayView const& dst, Point2Du32 src_pivot, Point2Du32 dst_pivot, uangle rot)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);

        Point2Di32 sp = { (i32)src_pivot.x, (i32)src_pivot.y };
        Point2Di32 dp = { (i32)dst_pivot.x, (i32)dst_pivot.y };

        rotate_view_any(src, dst, sp, dp, rot);
    }


    void rotate(ImageView const& src, ImageView const& dst, Point2Du32 src_pivot, Point2Du32 dst_pivot, uangle rot)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);

        Point2Di32 sp = { (i32)src_pivot.x, (i32)src_pivot.y };
        Point2Di32 dp = { (i32)dst_pivot.x, (i32)dst_pivot.y };

        rotate_view_any(src, dst, sp, dp, rot);
    }


    void rotate_blend(ImageView const& src, ImageView const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, uangle rot)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);

        rotate_blend_any(src, dst, src_pivot, dst_pivot, rot);
    }


    void rotate_blend(ImageView const& src, SubView const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, uangle rot)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);

        rotate_blend_any(src, dst, src_pivot, dst_pivot, rot);
    }


    void rotate_blend_transform(GraySubView const& src, SubView const& dst, Vec2Df32 cos_sin, fn<Pixel(u8)> const& func)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);

        Point2Di32 sp = { (i32)src.width / 2, (i32)src.height / 2 };
        Point2Di32 dp = { (i32)dst.width / 2, (i32)dst.height / 2 };

        rotate_blend_transform_any(src, dst, sp, dp, cos_sin.x, cos_sin.y, func);
    }
}


/* flip */

namespace image
{
    template <class VIEW_S, class VIEW_D>
    static void flip_view_h(VIEW_S const& src, VIEW_D const& dst)
    {
        auto const w = src.width;
        auto const h = src.height;
        auto const w2 = w / 2;

        for (u32 y = 0; y < h; y++)
        {
            auto s = row_begin(src, y);
            auto d = row_begin(dst, y);
            for (u32 x = 0; x <= w2; x++)
            {
                auto end = w - 1 - x;
                d[x] = s[end];
                d[end] = s[x];
            }
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void flip_view_h_blend(VIEW_S const& src, VIEW_D const& dst)
    {
        auto const w = src.width;
        auto const h = src.height;
        auto const w2 = w / 2;

        for (u32 y = 0; y < h; y++)
        {
            auto s = row_begin(src, y);
            auto d = row_begin(dst, y);
            for (u32 x = 0; x <= w2; x++)
            {
                auto end = w - 1 - x;
                alpha_blend(s[end], d + x);
                alpha_blend(s[x], d + end);
            }
        }
    }


    template <class VIEW_S, class VIEW_D>
    static void flip_view_v(VIEW_S const& src, VIEW_D const& dst)
    {
        auto const w = src.width;
        auto const h = src.height;

        for (u32 y = 0; y < h; y++)
        {
            sp::copy(row_span(src, y), row_span(dst, h - y - 1));
        }
    }


    void flip_h(ImageView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.width);
        assert(src.height == dst.height);

        flip_view_h(src, dst);
    }


    void flip_h(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.width);
        assert(src.height == dst.height);

        flip_view_h(src, dst);
    }


    void flip_h(SubView const& src, ImageView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.width);
        assert(src.height == dst.height);

        flip_view_h(src, dst);
    }


    void flip_h(SubView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.width);
        assert(src.height == dst.height);

        flip_view_h(src, dst);
    }


    void flip_v(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.width);
        assert(src.height == dst.height);

        flip_view_v(src, dst);
    }


    void flip_h_blend(ImageView const& src, SubView const& dst)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == dst.width);
        assert(src.height == dst.height);

        flip_view_h_blend(src, dst);
    }
}


/* centroid */

namespace image
{
    Point2Du32 centroid(ImageView const& view)
	{		
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        
		f64 total = 0.0;
		f64 x_total = 0.0;
		f64 y_total = 0.0;

        auto const accumulate = [&](u32 x, u32 y)
        {
            auto p = *xy_at(view, x, y);
            u64 val = p.alpha ? 1 : 0;
            total += val;
            x_total += x * val;
            y_total += y * val;
        };
        
        for_each_xy(view, accumulate);

		Point2Du32 pt{};

        if (total > 0.0)
        {
            pt.x = (u32)(x_total / total);
			pt.y = (u32)(y_total / total);
        }
		else
		{
			pt.x = view.width / 2;
			pt.y = view.height / 2;
		}

		return pt;
	}
}


/* resize */

namespace image
{
    void scale_down(ImageView const& src, ImageView const& dst, u32 scale)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(src.width == scale * dst.width);
        assert(src.height == scale * dst.height);

        if (scale == 1)
        {
            copy(src, dst);
            return;
        }
        
        f32 const i_scale = 1.0f / (scale * scale);

        f32 red = 0.0f;
        f32 green = 0.0f;
        f32 blue = 0.0f;
        f32 alpha = 0.0f;

        for (u32 yd = 0; yd < dst.height; yd++)
        {
            auto ys = scale * yd;

            auto rd = row_begin(dst, yd);            

            for (u32 xd = 0; xd < dst.width; xd++)
            {
                auto xs = scale * xd;

                red   = 0.0f;
                green = 0.0f;
                blue  = 0.0f;
                alpha = 0.0f;

                for (u32 v = 0; v < scale; v++)
                {
                    auto rs = row_begin(src, ys + v) + xs;
                    for (u32 u = 0; u < scale; u++)
                    {
                        auto s = rs[u];
                        red   += i_scale * s.red;
                        green += i_scale * s.green;
                        blue  += i_scale * s.blue;
                        alpha += i_scale * s.alpha;
                    }
                }

                auto& d = rd[xd];
                d.red   = (u8)red;
                d.green = (u8)green;
                d.blue  = (u8)blue;
                d.alpha = (u8)alpha;
            }
        }
    }


    void scale_up(ImageView const& src, ImageView const& dst, u32 scale)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width * scale);
        assert(dst.height == src.height * scale);

        for (u32 ys = 0; ys < src.height; ys++)
        {
            auto yd = scale * ys;
            auto rs = row_begin(src, ys);

            for (u32 xs = 0; xs < src.width; xs++)
            {
                auto xd = scale * xs;

                auto p = rs[xs];

                for (u32 v = 0; v < scale; v++)
                {
                    auto rd = row_begin(dst, yd + v) + xd;
                    for (u32 u = 0; u < scale; u++)
                    {
                        rd[u] = p;
                    }
                }
            }
        }
    }


    void scale_up(ImageView const& src, SubView const& dst, u32 scale)
    {
        assert(src.matrix_data_);
        assert(dst.matrix_data_);
        assert(dst.width == src.width * scale);
        assert(dst.height == src.height * scale);

        for (u32 ys = 0; ys < src.height; ys++)
        {
            auto yd = scale * ys;
            auto rs = row_begin(src, ys);

            for (u32 xs = 0; xs < src.width; xs++)
            {
                auto xd = scale * xs;

                auto p = rs[xs];

                for (u32 v = 0; v < scale; v++)
                {
                    auto rd = row_begin(dst, yd + v) + xd;
                    for (u32 u = 0; u < scale; u++)
                    {
                        rd[u] = p;
                    }
                }
            }
        }
    }


    bool resize(ImageView const& src, ImageView const& dst)
    {
#ifdef IMAGE_RESIZE
        assert(src.width);
		assert(src.height);
		assert(src.matrix_data_);
		assert(dst.width);
		assert(dst.height);
        assert(dst.matrix_data_);

		int channels = 4;
        auto layout = stbir_pixel_layout::STBIR_RGBA;

		int width_src = (int)(src.width);
		int height_src = (int)(src.height);
		int stride_bytes_src = width_src * channels;
        u8* data_src = (u8*)src.matrix_data_;

		int width_dst = (int)(dst.width);
		int height_dst = (int)(dst.height);
		int stride_bytes_dst = width_dst * channels;
        u8* data_dst = (u8*)dst.matrix_data_;

        auto data = stbir_resize_uint8_linear(
			data_src, width_src, height_src, stride_bytes_src,
			data_dst, width_dst, height_dst, stride_bytes_dst,
			layout);

		assert(data && " *** stbir_resize_uint8_linear() failed *** ");

        if (!data)
        {
            return false;
        }

		return true;
#else

        assert(false && " *** IMAGE_RESIZE not enabled *** ");
        return false;

#endif
    }
}


/* read write */

namespace image
{
    static bool has_extension(const char* filename, const char* ext)
    {
        auto file_length = sp::strlen(filename);
        auto ext_length = sp::strlen(ext);

        return !sp::strcmp(&filename[file_length - ext_length], ext);
    }


    static bool is_bmp(const char* filename)
    {
        return has_extension(filename, ".bmp") || has_extension(filename, ".BMP");
    }


    static bool is_png(const char* filename)
    {
        return has_extension(filename, ".png") || has_extension(filename, ".PNG");
    }


    static bool is_valid_image_file(const char* filename)
    {
        return 
            has_extension(filename, ".bmp") || 
            has_extension(filename, ".BMP") ||
            has_extension(filename, ".png")||
            has_extension(filename, ".PNG");
    }


    bool read_image_from_file(const char* img_path_src, Image& image_dst)
	{
    #ifdef IMAGE_READ
        auto is_valid_file = is_valid_image_file(img_path_src);
        assert(is_valid_file && "invalid image file");

        if (!is_valid_file)
        {
            return false;
        }

		int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 4;

		auto data = stbi_load(img_path_src, &width, &height, &image_channels, desired_channels);

		assert(data && "stbi_load() - no image data");
		assert(width && "stbi_load() - no image width");
		assert(height && "stbi_load() - no image height");

		if (!data)
		{
			return false;
		}

		image_dst.data_ = (Pixel*)data;
		image_dst.width = width;
		image_dst.height = height;

        mem::tag(data, image_dst.width * image_dst.height, "stbi_load");

		return true;
    #else

        assert(false && " *** IMAGE_READ not enabled *** ");
        return false;

    #endif
	}


    bool read_image_from_memory(ByteView const& bytes, Image& image_dst)
    {
    #ifdef IMAGE_READ

        auto buffer = (unsigned char*)bytes.data;
        int len = (int)bytes.length;
        int width = 0;
		int height = 0;
		int image_channels = 0;
		int desired_channels = 4;

        auto data = stbi_load_from_memory(buffer, len, &width, &height, &image_channels, desired_channels);

        assert(data && "stbi_load_from_memory() - no image data");
		assert(width && "stbi_load_from_memory() - no image width");
		assert(height && "stbi_load_from_memory() - no image height");

		if (!data)
		{
			return false;
		}

		image_dst.data_ = (Pixel*)data;
		image_dst.width = width;
		image_dst.height = height;

        mem::tag(data, image_dst.width * image_dst.height, "stbi_load");

		return true;

    #else

        assert(false && " *** IMAGE_READ not enabled *** ");
        return false;

    #endif
    }


    bool write_image(Image const& image_src, const char* file_path_dst)
	{
#ifdef IMAGE_WRITE
		assert(image_src.width);
		assert(image_src.height);
		assert(image_src.data_);

		int width = (int)(image_src.width);
		int height = (int)(image_src.height);
		int channels = 4;
		auto const data = image_src.data_;

		int result = 0;

		if(is_bmp(file_path_dst))
		{
			result = stbi_write_bmp(file_path_dst, width, height, channels, data);
			assert(result);
		}
		else if(is_png(file_path_dst))
		{
			int stride_in_bytes = width * channels;

			result = stbi_write_png(file_path_dst, width, height, channels, data, stride_in_bytes);
			assert(result);
		}
		else
		{
			assert(false && " *** bad image file type *** ");
            return false;
		}

		return (bool)result;
#else

        assert(false && " *** IMAGE_WRITE not enabled *** ");
        return false;

#endif
	}

}