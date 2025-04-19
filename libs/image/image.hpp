#pragma once

#include "../span/span.hpp"


/*  image basic */

namespace image
{
    namespace mb = memory_buffer;


    class RGBAu8
    {
    public:
        u8 red;
        u8 green;
        u8 blue;
        u8 alpha;
    };

    using Pixel = RGBAu8;
    using Image = Matrix2D<Pixel>;
    using ImageView = MatrixView2D<Pixel>;    
    using ImageGray = Matrix2D<u8>;
    using GrayView = MatrixView2D<u8>;


    bool create_image(Image& image, u32 width, u32 height);

    void destroy_image(Image& image);


    bool create_image(ImageGray& image, u32 width, u32 height);

    void destroy_image(ImageGray& image);


    inline u32 as_u32(Pixel p)
    {
        return  *((u32*)(&p));
    }


    inline Image as_image(ImageView const& view)
    {
        Image image;
        image.width = view.width;
        image.height = view.height;
        image.data_ = view.matrix_data_;

        return image;
    }
}


namespace image
{
    template <typename T>
    class MatrixSubView2D
    {
    public:
        T*  matrix_data_;
        u32 matrix_width;

        u32 x_begin;
        u32 y_begin;

        u32 width;
        u32 height;
    };


    using SubView = MatrixSubView2D<Pixel>;    
    using GraySubView = MatrixSubView2D<u8>;
}


namespace image
{
    using Buffer8 = MemoryBuffer<u8>;
    using Buffer32 = MemoryBuffer<Pixel>;


    constexpr inline Pixel to_pixel(u8 red, u8 green, u8 blue, u8 alpha)
    {
        Pixel p{};
        p.red = red;
        p.green = green;
        p.blue = blue;
        p.alpha = alpha;

        return p;
    }


    constexpr inline Pixel to_pixel(u8 red, u8 green, u8 blue)
    {
        return to_pixel(red, green, blue, 255);
    }


    constexpr inline Pixel to_pixel(u8 gray)
    {
        return to_pixel(gray, gray, gray);
    } 


    inline Buffer8 create_buffer8(u32 n_pixels, cstr tag)
	{
		Buffer8 buffer;
		mb::create_buffer(buffer, n_pixels, tag);
		return buffer;
	}


    inline Buffer32 create_buffer32(u32 n_pixels, cstr tag)
	{
		Buffer32 buffer;
		mb::create_buffer(buffer, n_pixels, tag);
		return buffer;
	}


    inline constexpr Rect2Du32 make_rect(u32 width, u32 height)
    {
        Rect2Du32 range{};
        range.x_begin = 0;
        range.x_end = width;
        range.y_begin = 0;
        range.y_end = height;

        return range;
    }


    inline constexpr Rect2Du32 make_rect(u32 x_begin, u32 y_begin, u32 width, u32 height)
    {
        Rect2Du32 range{};
        range.x_begin = x_begin;
        range.x_end = x_begin + width;
        range.y_begin = y_begin;
        range.y_end = y_begin + height;

        return range;
    }
}


/* make_view */

namespace image
{
    ImageView make_view(u32 width, u32 height, Pixel* data);

    GrayView make_view(u32 width, u32 height, u8* data);
    
    ImageView make_view(Image const& image);

    GrayView make_view(ImageGray const& image);

    ImageView make_view(u32 width, u32 height, Buffer32& buffer);

    GrayView make_view(u32 width, u32 height, Buffer8& buffer);
}


/* sub_view */

namespace image
{
    template <typename T>
    inline MatrixSubView2D<T> sub_view(MatrixView2D<T> const& view, Rect2Du32 const& range)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        assert(range.x_end > 0);
        assert(range.x_begin < view.width);
        assert(range.x_end <= view.width);
        assert(range.x_begin < range.x_end);
        assert(range.x_end - range.x_begin <= view.width);
        assert(range.y_end > 0);
        assert(range.y_begin < view.height);
        assert(range.y_end <= view.height);
        assert(range.y_begin < range.y_end);
        assert(range.y_end - range.y_begin <= view.height);

        MatrixSubView2D<T> sub_view{};

        sub_view.matrix_data_ = view.matrix_data_;
        sub_view.matrix_width = view.width;
        sub_view.x_begin = range.x_begin;
        sub_view.y_begin = range.y_begin;
        sub_view.width = range.x_end - range.x_begin;
        sub_view.height = range.y_end - range.y_begin;

        return sub_view;
    }


    template <typename T>
    inline MatrixSubView2D<T> sub_view(MatrixSubView2D<T> const& view, Rect2Du32 const& range)
    {
        assert(view.matrix_data_);
        assert(view.width);
        assert(view.height);
        assert(range.x_end > 0);
        assert(range.x_begin < view.width);
        assert(range.x_end <= view.width);        
        assert(range.x_begin < range.x_end);
        assert(range.x_end - range.x_begin <= view.width);
        assert(range.y_end > 0);
        assert(range.y_begin < view.height);
        assert(range.y_end <= view.height);
        assert(range.y_begin < range.y_end);
        assert(range.y_end - range.y_begin <= view.height);
        
        MatrixSubView2D<T> sub_view{};

        sub_view.matrix_data_ = view.matrix_data_;
        sub_view.matrix_width = view.matrix_width;

        sub_view.x_begin = range.x_begin + view.x_begin;
		sub_view.y_begin = range.y_begin + view.y_begin;

		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

        return sub_view;
    }


    template <typename T>
    inline MatrixSubView2D<T> sub_view(MatrixView2D<T> const& view)
    {
        auto range = make_range(view.width, view.height);
        return sub_view(view, range);
    }


    inline Point2Du32 to_point(u32 x, u32 y)
    {
        Point2Du32 pt{};
        pt.x = x;
        pt.y = y;

        return pt;
    }


    template <typename T>
    static inline SpanView<T> to_span(Matrix2D<T> const& view)
    {
        SpanView<T> span{};

        span.data = view.matrix_data_;
        span.length = view.width * view.height;

        return span;
    }
    

    template <typename T>
    static inline SpanView<T> to_span(MatrixView2D<T> const& view)
    {
        SpanView<T> span{};

        span.data = view.matrix_data_;
        span.length = view.width * view.height;

        return span;
    }
}


/* pixel_at */

namespace image
{
    Pixel pixel_at(ImageView const& view, u32 x, u32 y);

    u8 pixel_at(GrayView const& view, u32 x, u32 y);

    u8 pixel_at(GraySubView const& view, u32 x, u32 y);
}


/* fill */

namespace image
{
    void fill(ImageView const& view, Pixel color);

    void fill(SubView const& view, Pixel color);

    void fill_if(GraySubView const& view, u8 gray, fn<bool(u8)> const& pred);

    void fill_blend(SubView const& view, Pixel color);


    void fill_row(SubView const& view, u32 y, Pixel color);

    void fill_row_blend(SubView const& view, u32 y, Pixel color);
}


/* copy */

namespace image
{
    void copy(ImageView const& src, ImageView const& dst);

    void copy(ImageView const& src, SubView const& dst);

    void copy(SubView const& src, ImageView const& dst);

    void copy(SubView const& src, SubView const& dst);


    void copy_blend(ImageView const& src, SubView const& dst);

    void copy_blend(SubView const& src, ImageView const& dst);

    void copy_blend(SubView const& src, SubView const& dst);
}


/* for_each_pixel */

namespace image
{  
    void for_each_pixel(ImageView const& view, fn<void(Pixel)> const& func);

    void for_each_pixel(SubView const& view, fn<void(Pixel)> const& func);
}


/* for_each_xy */

namespace image
{
    void for_each_xy(ImageView const& view, fn<Pixel(u32 x, u32 y)> const& xy_func);    

    void for_each_xy(SubView const& view, fn<Pixel(u32 x, u32 y)> const& xy_func);

    void for_each_xy(ImageView const& view, fn<void(u32 x, u32 y)> const& xy_func);


    void for_each_xy(GrayView const& view, fn<u8(u32 x, u32 y)> const& xy_func);
}


/* transform */

namespace image
{
    void transform(ImageView const& src, SubView const& dst, fn<Pixel(Pixel, Pixel)> const& func);

    void transform(SubView const& src, SubView const& dst, fn<Pixel(Pixel, Pixel)> const& func);

    void transform(GrayView const& src, SubView const& dst, fn<Pixel(u8, Pixel)> const& func);

    void transform(GraySubView const& src, SubView const& dst, fn<Pixel(u8, Pixel)> const& func);


    void transform(ImageView const& src, GrayView const& dst, fn<u8(Pixel)> const& func);

    void transform(ImageView const& src, GraySubView const& dst, fn<u8(Pixel)> const& func);


    void transform(GrayView const& src, ImageView const& dst, fn<Pixel(u8)> const& func);

    void transform(GrayView const& src, SubView const& dst, fn<Pixel(u8)> const& func);

    void transform(GraySubView const& src, ImageView const& dst, fn<Pixel(u8)> const& func);

    void transform(GraySubView const& src, SubView const& dst, fn<Pixel(u8)> const& func);


    void transform_scale_up(GraySubView const& src, SubView const& dst, u32 scale, fn<Pixel(u8, Pixel)> const& func);

    void transform_scale_up(ImageView const& src, GrayView const& dst, u32 scale, fn<u8(Pixel)> const& func);


    // todo delete
    inline void transform_stretch(GraySubView const& src, SubView const& dst, u32 scale, fn<Pixel(u8, Pixel)> const& func)
    {
        transform_scale_up(src, dst, scale, func);
    }


    inline void transform_stretch(ImageView const& src, GrayView const& dst, u32 scale, fn<u8(Pixel)> const& func)
    {
        transform_scale_up(src, dst, scale, func);
    }
}


/* circle */

namespace image
{
namespace draw
{
    void circle_outline(ImageView const& view, Point2Di32 center, u32 radius, Pixel color);

    void circle_outline(SubView const& view, Point2Di32 center, u32 radius, Pixel color);

    void circle_fill(ImageView const& view, Point2Di32 center, u32 radius, Pixel color);

    void circle_fill(SubView const& view, Point2Di32 center, u32 radius, Pixel color);
}
}


/* rotate */

namespace image
{
    void rotate_90(ImageView const& src, ImageView const& dst);

    void rotate_90(ImageView const& src, SubView const& dst);

    void rotate_90(SubView const& src, SubView const& dst);


    void rotate_180(ImageView const& src, ImageView const& dst);

    void rotate_180(ImageView const& src, SubView const& dst);

    void rotate_180(SubView const& src, SubView const& dst);


    void rotate_270(ImageView const& src, ImageView const& dst);

    void rotate_270(ImageView const& src, SubView const& dst);

    void rotate_270(SubView const& src, SubView const& dst);


    void rotate(GrayView const& src, GrayView const& dst, Point2Du32 src_pivot, Point2Du32 dst_pivot, uangle rot);
    
    void rotate(ImageView const& src, ImageView const& dst, Point2Du32 src_pivot, Point2Du32 dst_pivot, uangle rot);

    void rotate_blend(ImageView const& src, ImageView const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, uangle rot);

    void rotate_blend(ImageView const& src, SubView const& dst, Point2Di32 src_pivot, Point2Di32 dst_pivot, uangle rot);


    void rotate_blend_transform(GraySubView const& src, SubView const& dst, Vec2Df32 cos_sin, fn<Pixel(u8)> const& func);
}


/* flip */

namespace image
{
    void flip_h(ImageView const& src, ImageView const& dst);

    void flip_h(ImageView const& src, SubView const& dst);

    void flip_h(SubView const& src, ImageView const& dst);

    void flip_h(SubView const& src, SubView const& dst);


    void flip_v(ImageView const& src, SubView const& dst);
    

    void flip_h_blend(ImageView const& src, SubView const& dst);
}


namespace image
{
    Point2Du32 centroid(ImageView const& view);
}


/* resize */

namespace image
{
    void scale_down(ImageView const& src, ImageView const& dst, u32 scale);

    void scale_up(ImageView const& src, ImageView const& dst, u32 scale);

    void scale_up(ImageView const& src, SubView const& dst, u32 scale);

    bool resize(ImageView const& src, ImageView const& dst);
}


/* read write */

namespace image
{
    bool read_image_from_file(const char* img_path_src, Image& image_dst);

    bool read_image_from_memory(ByteView const& data, Image& image_dst);

    bool write_image(Image const& image_src, const char* file_path_dst);
}