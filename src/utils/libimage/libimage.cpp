/*

Copyright (c) 2021 Adam Lafontaine

*/

#include "libimage.hpp"




namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void make_image(image_t& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (pixel_t*)malloc(sizeof(pixel_t) * width * height);

		assert(image_dst.data);
	}


	view_t make_view(image_t const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		view_t view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	view_t make_view(image_t& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	view_t sub_view(image_t const& image, pixel_range_t const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		view_t sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	view_t sub_view(view_t const& view, pixel_range_t const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin >= view.x_begin);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin >= view.y_begin);
		assert(range.y_end <= view.y_end);

		view_t sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	view_t row_view(image_t const& image, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = image.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	view_t row_view(view_t const& view, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = view.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	view_t column_view(image_t const& image, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = image.height;

		return sub_view(image, range);
	}


	view_t column_view(view_t const& view, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = view.height;

		return sub_view(view, range);
	}


	view_t row_view(image_t const& image, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	view_t row_view(view_t const& view, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	view_t column_view(image_t const& image, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(image, range);
	}


	view_t column_view(view_t const& view, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(view, range);
	}


	void for_each_pixel(image_t const& image, std::function<void(pixel_t& p)> const& func)
	{
		u32 size = image.width * image.height;
		for (u32 i = 0; i < size; ++i)
		{
			func(image.data[i]);
		}
	}


	void for_each_pixel(view_t const& view, std::function<void(pixel_t& p)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto row = view.row_begin(y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(row[x]);
			}
		}
	}


	void for_each_xy(image_t const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			for (u32 x = 0; x < image.width; ++x)
			{
				func(x, y);
			}
		}
	}


	void for_each_xy(view_t const& view, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			for (u32 x = 0; x < view.width; ++x)
			{
				func(x, y);
			}
		}
	}

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

	void make_image(gray::image_t& image_dst, u32 width, u32 height)
	{
		assert(width);
		assert(height);

		image_dst.width = width;
		image_dst.height = height;
		image_dst.data = (gray::pixel_t*)malloc(sizeof(gray::pixel_t) * width * height);

		assert(image_dst.data);
	}


	gray::view_t make_view(gray::image_t const& img)
	{
		assert(img.width);
		assert(img.height);
		assert(img.data);

		gray::view_t view;

		view.image_data = img.data;
		view.image_width = img.width;
		view.x_begin = 0;
		view.y_begin = 0;
		view.x_end = img.width;
		view.y_end = img.height;
		view.width = img.width;
		view.height = img.height;

		return view;
	}


	gray::view_t make_view(gray::image_t& image, u32 width, u32 height)
	{
		make_image(image, width, height);
		return make_view(image);
	}


	gray::view_t sub_view(gray::image_t const& image, pixel_range_t const& range)
	{
		assert(image.width);
		assert(image.height);
		assert(image.data);

		gray::view_t sub_view;

		sub_view.image_data = image.data;
		sub_view.image_width = image.width;
		sub_view.x_begin = range.x_begin;
		sub_view.y_begin = range.y_begin;
		sub_view.x_end = range.x_end;
		sub_view.y_end = range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::view_t sub_view(gray::view_t const& view, pixel_range_t const& range)
	{
		assert(view.width);
		assert(view.height);
		assert(view.image_data);

		assert(range.x_begin < view.x_end);
		assert(range.x_end <= view.x_end);
		assert(range.y_begin < view.y_end);
		assert(range.y_end <= view.y_end);

		gray::view_t sub_view;

		sub_view.image_data = view.image_data;
		sub_view.image_width = view.image_width;
		sub_view.x_begin = view.x_begin + range.x_begin;
		sub_view.y_begin = view.y_begin + range.y_begin;
		sub_view.x_end = view.x_begin + range.x_end;
		sub_view.y_end = view.y_begin + range.y_end;
		sub_view.width = range.x_end - range.x_begin;
		sub_view.height = range.y_end - range.y_begin;

		assert(sub_view.width);
		assert(sub_view.height);

		return sub_view;
	}


	gray::view_t row_view(gray::image_t const& image, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = image.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	gray::view_t row_view(gray::view_t const& view, u32 y)
	{
		pixel_range_t range;
		range.x_begin = 0;
		range.x_end = view.width;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	gray::view_t column_view(gray::image_t const& image, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = image.height;

		return sub_view(image, range);
	}


	gray::view_t column_view(gray::view_t const& view, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = 0;
		range.y_end = view.height;

		return sub_view(view, range);
	}


	gray::view_t row_view(gray::image_t const& image, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(image, range);
	}


	gray::view_t row_view(gray::view_t const& view, u32 x_begin, u32 x_end, u32 y)
	{
		pixel_range_t range;
		range.x_begin = x_begin;
		range.x_end = x_end;
		range.y_begin = y;
		range.y_end = y + 1;

		return sub_view(view, range);
	}


	gray::view_t column_view(gray::image_t const& image, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(image, range);
	}


	gray::view_t column_view(gray::view_t const& view, u32 y_begin, u32 y_end, u32 x)
	{
		pixel_range_t range;
		range.x_begin = x;
		range.x_end = x + 1;
		range.y_begin = y_begin;
		range.y_end = y_end;

		return sub_view(view, range);
	}


	void for_each_pixel(gray::image_t const& image, std::function<void(gray::pixel_t& p)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			auto row = image.row_begin(y);
			for (u32 x = 0; x < image.width; ++x)
			{
				func(row[x]);
			}
		}
	}


	void for_each_pixel(gray::view_t const& view, std::function<void(gray::pixel_t& p)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto row = view.row_begin(y);
			for (u32 x = 0; x < view.width; ++x)
			{
				func(row[x]);
			}
		}
	}


	void for_each_xy(gray::image_t const& image, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < image.height; ++y)
		{
			for (u32 x = 0; x < image.width; ++x)
			{
				func(x, y);
			}
		}
	}


	void for_each_xy(gray::view_t const& view, std::function<void(u32 x, u32 y)> const& func)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			for (u32 x = 0; x < view.width; ++x)
			{
				func(x, y);
			}
		}
	}

#endif // !LIBIMAGE_NO_GRAYSCALE

}

