#pragma once
/*

Copyright (c) 2021 Adam Lafontaine

*/

//#define LIBIMAGE_NO_COLOR
#define LIBIMAGE_NO_GRAYSCALE
#define LIBIMAGE_NO_READ
#define LIBIMAGE_NO_WRITE
#define LIBIMAGE_NO_RESIZE
//#define LIBIMAGE_NO_PARALLEL
//#define LIBIMAGE_NO_FILESYSTEM


// jetson nano
//#define LIBIMAGE_NO_CPP17
//#define LIBIMAGE_NO_SIMD


#ifdef LIBIMAGE_NO_CPP17

#define LIBIMAGE_NO_PARALLEL
#define LIBIMAGE_NO_FILESYSTEM

#endif // LIBIMAGE_NO_CPP17

#ifndef LIBIMAGE_NO_COLOR
#include "rgba.hpp"
#endif

#ifndef LIBIMAGE_NO_GRAYSCALE
#include "gray.hpp"
#endif

#include <functional>
#include <string>




namespace libimage
{

#ifndef LIBIMAGE_NO_COLOR

	void make_image(image_t& image_dst, u32 width, u32 height);

	view_t make_view(image_t const& image);

	view_t make_view(image_t& image, u32 width, u32 height);

	view_t sub_view(image_t const& image, pixel_range_t const& range);

	view_t sub_view(view_t const& view, pixel_range_t const& range);

	view_t row_view(image_t const& image, u32 y);

	view_t row_view(view_t const& view, u32 y);

	view_t column_view(image_t const& image, u32 x);

	view_t column_view(view_t const& view, u32 x);

	view_t row_view(image_t const& image, u32 x_begin, u32 x_end, u32 y);

	view_t row_view(view_t const& view, u32 x_begin, u32 x_end, u32 y);

	view_t column_view(image_t const& image, u32 y_begin, u32 y_end, u32 x);

	view_t column_view(view_t const& view, u32 y_begin, u32 y_end, u32 x);

	void for_each_pixel(image_t const& image, std::function<void(pixel_t& p)> const& func);

	void for_each_pixel(view_t const& view, std::function<void(pixel_t& p)> const& func);

	void for_each_xy(image_t const& image, std::function<void(u32 x, u32 y)> const& func);

	void for_each_xy(view_t const& view, std::function<void(u32 x, u32 y)> const& func);

#ifndef LIBIMAGE_NO_READ

	void read_image_from_file(const char* img_path_src, image_t& image_dst);

#endif // !LIBIMAGE_NO_READ

#ifndef LIBIMAGE_NO_WRITE

	void write_image(image_t const& image_src, const char* file_path_dst);

	void write_view(view_t const& view_src, const char* file_path_dst);
	

#endif // !LIBIMAGE_NO_WRITE

#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(image_t const& image_src, image_t& image_dst);

	view_t make_resized_view(image_t const& image_src, image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_COLOR


#ifndef LIBIMAGE_NO_GRAYSCALE




	void make_image(gray::image_t& image_dst, u32 width, u32 height);

	gray::view_t make_view(gray::image_t const& image);

	gray::view_t make_view(gray::image_t& image, u32 width, u32 height);

	gray::view_t sub_view(gray::image_t const& image, pixel_range_t const& range);

	gray::view_t sub_view(gray::view_t const& view, pixel_range_t const& range);

	gray::view_t row_view(gray::image_t const& image, u32 y);

	gray::view_t row_view(gray::view_t const& view, u32 y);

	gray::view_t column_view(gray::image_t const& image, u32 y);

	gray::view_t column_view(gray::view_t const& view, u32 y);

	gray::view_t row_view(gray::image_t const& image, u32 x_begin, u32 x_end, u32 y);

	gray::view_t row_view(gray::view_t const& view, u32 x_begin, u32 x_end, u32 y);

	gray::view_t column_view(gray::image_t const& image, u32 y_begin, u32 y_end, u32 x);

	gray::view_t column_view(gray::view_t const& view, u32 y_begin, u32 y_end, u32 x);

	void for_each_pixel(gray::image_t const& image, std::function<void(gray::pixel_t& p)> const& func);

	void for_each_pixel(gray::view_t const& view, std::function<void(gray::pixel_t& p)> const& func);

	void for_each_xy(gray::image_t const& image, std::function<void(u32 x, u32 y)> const& func);

	void for_each_xy(gray::view_t const& view, std::function<void(u32 x, u32 y)> const& func);


#ifndef LIBIMAGE_NO_READ

	void read_image_from_file(const char* file_path_src, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_READ
	

#ifndef LIBIMAGE_NO_WRITE

	void write_image(gray::image_t const& image_src, const char* file_path_dst);

	void write_view(gray::view_t const& view_src, const char* file_path_dst);
	

#endif // !LIBIMAGE_NO_WRITE

#ifndef LIBIMAGE_NO_RESIZE

	void resize_image(gray::image_t const& img_src, gray::image_t& img_dst);

	gray::view_t make_resized_view(gray::image_t const& image_src, gray::image_t& image_dst);

#endif // !LIBIMAGE_NO_RESIZE

#endif // !LIBIMAGE_NO_GRAYSCALE

}


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_READ

	inline void read_image_from_file(std::string const& img_path_src, image_t& image_dst)
	{
		read_image_from_file(img_path_src.c_str(), image_dst);
	}

#endif // !LIBIMAGE_NO_READ

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(image_t const& image_src, std::string const& file_path)
	{
		write_image(image_src, file_path.c_str());
	}

	inline void write_view(view_t const& view_src, std::string const& file_path)
	{
		write_view(view_src, file_path.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_READ

	inline void read_image_from_file(std::string const& img_path_src, gray::image_t& image_dst)
	{
		read_image_from_file(img_path_src.c_str(), image_dst);
	}

#endif // !LIBIMAGE_NO_READ

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(gray::image_t const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}

	inline void write_view(gray::view_t const& view_src, std::string const& file_path_dst)
	{
		write_view(view_src, file_path_dst.c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_GRAYSCALE
}


#ifndef LIBIMAGE_NO_FILESYSTEM

#include <filesystem>
namespace fs = std::filesystem;


namespace libimage
{
#ifndef LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_READ

	inline void read_image_from_file(fs::path const& img_path_src, image_t& image_dst)
	{
		read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#endif // !LIBIMAGE_NO_READ

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(image_t const& image_src, fs::path const& file_path)
	{
		write_image(image_src, file_path.string().c_str());
	}

	inline void write_view(view_t const& view_src, fs::path const& file_path)
	{
		write_view(view_src, file_path.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_COLOR

#ifndef LIBIMAGE_NO_GRAYSCALE

#ifndef LIBIMAGE_NO_READ

	inline void read_image_from_file(fs::path const& img_path_src, gray::image_t& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#endif // !LIBIMAGE_NO_READ

#ifndef LIBIMAGE_NO_WRITE

	inline void write_image(gray::image_t const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}

	inline void write_view(gray::view_t const& view_src, fs::path const& file_path_dst)
	{
		write_view(view_src, file_path_dst.string().c_str());
	}

#endif // !LIBIMAGE_NO_WRITE

#endif // !LIBIMAGE_NO_GRAYSCALE
}

#endif // !LIBIMAGE_NO_FILESYSTEM