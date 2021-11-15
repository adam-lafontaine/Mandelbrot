#include "app.hpp"
#include "../utils/index_range.hpp"
#include "../utils/image.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <limits>
#include <execution>
#include <cmath>
#include <functional>


constexpr r64 MBT_MIN_X = -2.0;
constexpr r64 MBT_MAX_X = 0.7;
constexpr r64 MBT_MIN_Y = -1.2;
constexpr r64 MBT_MAX_Y = 1.2;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;

constexpr u32 NUM_SHADES = 256;

constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = 50;
constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = 50000;
constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
constexpr r64 ZOOM_SPEED_LOWER_LIMIT = 1.0;

class Point2Dr64
{
public:
	r64 x;
	r64 y;
};


class Point2Di32
{
public:
	i32 x;
	i32 y;
};

using Vec2Di32 = Point2Di32;


class Range2Du32
{
public:
	u32 x_begin;
	u32 x_end;
	u32 y_begin;
	u32 y_end;
};


class AppState
{
public:
	bool render_new;

	Point2Dr64 screen_pos;
	Vec2Di32 pixel_shift;

	r64 zoom_level;
	r64 zoom_speed;
	
	u32 rgb_option;

	u32 max_iter;
	mat_u32_t iterations;
};


static r64 screen_width(AppState const& state)
{
	return MBT_WIDTH / state.zoom_level;
}


static r64 screen_height(AppState const& state)
{
	return MBT_HEIGHT / state.zoom_level;
}


static image_t make_buffer_image(app::PixelBuffer const& buffer)
{
	assert(buffer.bytes_per_pixel == RGBA_CHANNELS);

	image_t image{};

	image.width = buffer.width;
	image.height = buffer.height;
	image.data = (pixel_t*)buffer.memory;

	return image;
}


static Range2Du32 get_range(image_t const& img)
{
	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = img.width;
	r.y_begin = 0;
	r.y_end = img.height;

	return r;
}


static Range2Du32 get_range(mat_u32_t const& mat)
{
	Range2Du32 r{};
	r.x_begin = 0;
	r.x_end = mat.width;
	r.y_begin = 0;
	r.y_end = mat.height;

	return r;
}


static pixel_t to_platform_pixel(u8 red, u8 green, u8 blue)
{
	pixel_t p = {};
	p.value = platform_to_color_32(red, green, blue);

	return p;
}


static pixel_t to_platform_pixel(u8 gray)
{
	pixel_t p = {};
	p.value = platform_to_color_32(gray, gray, gray);

	return p;
}


static pixel_t to_platform_pixel(pixel_t const& p)
{
	return to_platform_pixel(p.red, p.green, p.blue);
}


constexpr pixel_t to_pixel(u8 red, u8 green, u8 blue)
{
	pixel_t p = {};
	p.red = red;
	p.green = green;
	p.blue = blue;

	return p;
}


static void fill(image_t const& dst, pixel_t const& p)
{
	auto const platform_p = to_platform_pixel(p);
	std::fill(dst.begin(), dst.end(), platform_p);
}




using gray_palette_t = std::array<u8, NUM_SHADES * 2 - 1>;


constexpr gray_palette_t make_gray_palette()
{
	gray_palette_t palette = {};

	for (u32 i = 0; i < NUM_SHADES; ++i)
	{
		palette[i] = static_cast<u8>(i);
	}

	for (u32 i = NUM_SHADES; i < palette.size(); ++i)
	{
		u8 c = static_cast<u8>(palette.size() - 1 - i);
		palette[i] = c;
	}

	return palette;
}


static pixel_t to_gray(r64 ratio)
{
	assert(ratio >= 0.0);
	assert(ratio <= 1.0);

	auto const p = ratio;
	auto const q = 1.0 - p;
	auto const p2 = p * p;
	auto const p3 = p2 * p;
	auto const q2 = q * q;
	auto const q3 = q2 * q;

	constexpr auto palette = make_gray_palette();
	constexpr auto n_colors = static_cast<u32>(palette.size());

	auto const r = 16.0 * q2 * p2;

	auto g = static_cast<u32>(r * (n_colors - 1));

	if (g >= n_colors)
	{
		g = n_colors - 1;
	}

	auto const shade = palette[g];

	return to_pixel(shade, shade, shade);
}


using color_palette_t = std::array<u8, NUM_SHADES>;


constexpr color_palette_t make_color_palette()
{
	color_palette_t palette = {};

	for (u32 i = 0; i < NUM_SHADES; ++i)
	{
		palette[i] = static_cast<u8>(i);
	}

	return palette;
}


static pixel_t to_rgb(r64 ratio, u32 rgb_option)
{
	assert(ratio >= 0.0);
	assert(ratio <= 1.0);

	auto const p = ratio;
	auto const q = 1.0 - p;
	auto const p2 = p * p;
	auto const p3 = p2 * p;
	auto const q2 = q * q;
	auto const q3 = q2 * q;

	constexpr auto palette = make_color_palette();
	constexpr auto n_colors = static_cast<u32>(palette.size());

	auto const max_low = 9.4 * q3 * p;
	auto const max_high = 9.45 * q * p3;
	auto const max_center = 16.0 * q2 * p2;
	/*auto const low_to_high = p;
	auto const high_to_low = q;
	auto const high_to_low2 = q2;*/

	r64 color_map[] = { max_high, max_low, max_center };
	u32 rgb_idx[] = { 0, 1, 2 };

	switch (rgb_option)
	{
	case 1:
		rgb_idx[0] = 0;
		rgb_idx[1] = 2;
		rgb_idx[2] = 1;		
		break;
	case 2:
		rgb_idx[0] = 2;
		rgb_idx[1] = 0;
		rgb_idx[2] = 1;
		break;
	case 3:
		rgb_idx[0] = 0;
		rgb_idx[1] = 1;
		rgb_idx[2] = 2;		
		break;
	case 4:
		rgb_idx[0] = 2;
		rgb_idx[1] = 1;
		rgb_idx[2] = 0;		
		break;
	case 5:
		rgb_idx[0] = 1;
		rgb_idx[1] = 2;
		rgb_idx[2] = 0;
		break;
	case 6:
		rgb_idx[0] = 1;
		rgb_idx[1] = 0;
		rgb_idx[2] = 2;
		break;
	}

	auto r = static_cast<u32>(color_map[rgb_idx[0]] * (n_colors - 1));
	if (r >= n_colors)
	{
		r = n_colors - 1;
	}
	
	auto g = static_cast<u32>(color_map[rgb_idx[1]] * (n_colors - 1));
	if (g >= n_colors)
	{
		g = n_colors - 1;
	}
	
	auto b = static_cast<u32>(color_map[rgb_idx[2]] * (n_colors - 1));
	if (b >= n_colors)
	{
		b = n_colors - 1;
	}

	u8 const red = palette[r];
	u8 const green = palette[g];
	u8 const blue = palette[b];

	return to_platform_pixel(red, green, blue);
}


constexpr pixel_t to_rgb2(r64 ratio)
{
	constexpr u32 N = 16;
	std::array<pixel_t, N> palette = 
	{
		to_pixel(66, 30, 15),
		to_pixel(25, 7, 26),
		to_pixel(9, 1, 47),
		to_pixel(4, 4, 73),
		to_pixel(0, 7, 100),
		to_pixel(12, 44, 138),
		to_pixel(24, 82, 177),
		to_pixel(57, 125, 209),
		to_pixel(134, 181, 229),
		to_pixel(211, 236, 248),
		to_pixel(241, 233, 191),
		to_pixel(248, 201, 95),
		to_pixel(255, 170, 0),
		to_pixel(204, 128, 0),
		to_pixel(153, 87, 0),
		to_pixel(106, 52, 3),
	};

	auto n = ratio * (N - 1);
	if (n < 0.0)
	{
		return palette[0];
	}

	auto low = static_cast<u32>(n);
	if (low == 0)
	{
		return palette[0];
	}

	if (low >= N - 1)
	{
		return palette[N - 1];
	}

	auto high = low + 1;

	auto t = (n - low);

	auto p_low = palette[low];
	auto p_high = palette[high];

	auto const lerp_channel = [&](u32 c) 
	{
		auto const diff = static_cast<r32>(p_high.channels[c]) - p_low.channels[c];
		return static_cast<u8>(p_low.channels[c] + t * diff);
	};

	auto red = lerp_channel(0);
	auto green = lerp_channel(1);
	auto blue = lerp_channel(2);

	return to_pixel(red, green, blue);
}


static void draw(image_t const& dst, AppState const& state)
{
	auto& mat = state.iterations;

	auto [mat_min, mat_max] = std::minmax_element(mat.begin(), mat.end());
	auto min = static_cast<r64>(*mat_min);
	auto max = static_cast<r64>(*mat_max);

	auto diff = max - min;

	/*auto const f = 0.03;

	min += f * diff;
	max -= f * diff;*/

	diff = max - min;

	auto const to_platform_color = [&](u32 i) 
	{
		auto r = (i - min) / diff;

		if (r < 0.0)
		{
			r = 0.0;
		}
		else if (r > 1.0)
		{
			r = 1.0;
		}

		r = std::sqrt(r);

		//return to_gray(r);
		return to_rgb(r, state.rgb_option);
	};

	std::transform(std::execution::par, mat.begin(), mat.end(), dst.begin(), to_platform_color);
	//std::transform(mat.begin(), mat.end(), dst.begin(), to_platform_color);
}


static void for_each_row(image_t const& img, std::function<void(u32 y)> const& func)
{
	UnsignedRange y_ids(0u, img.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	std::for_each(std::execution::par, y_id_begin, y_id_end, func);
}


static void for_each_row(mat_u32_t const& mat, std::function<void(u32 y)> const& func)
{
	UnsignedRange y_ids(0u, mat.height);
	auto const y_id_begin = y_ids.begin();
	auto const y_id_end = y_ids.end();

	std::for_each(std::execution::par, y_id_begin, y_id_end, func);
}


static void copy_left(mat_u32_t const& mat, u32 n_cols)
{
	u32 const x_len = mat.width - n_cols;

	auto const copy_row_part = [&](u32 y)
	{
		auto row = mat.row_begin(y);
		for (u32 ix = 0; ix < x_len; ++ix)
		{
			auto src_x = ix + n_cols;
			auto dst_x = src_x - n_cols;
			row[dst_x] = row[src_x];
		}
	};

	for_each_row(mat, copy_row_part);
}


static void copy_right(mat_u32_t const& mat, u32 n_cols)
{
	u32 const x_len = mat.width - n_cols;

	auto const copy_row_part = [&](u32 y)
	{
		auto row = mat.row_begin(y);
		for (u32 ix = 0; ix < x_len; ++ix)
		{
			auto src_x = x_len - 1 - ix;
			auto dst_x = src_x + n_cols;
			row[dst_x] = row[src_x];
		}
	};

	for_each_row(mat, copy_row_part);
}


static void copy(mat_u32_t const& mat, Vec2Di32 const& direction)
{
	auto up = direction.y < 0;
	auto right = direction.x > 0;

	auto const n_cols = static_cast<u32>(std::abs(direction.x));
	auto const n_rows = static_cast<u32>(std::abs(direction.y));

	if (n_cols == 0 && n_rows == 0)
	{
		return;
	}

	if (n_rows == 0)
	{
		if (right)
		{
			copy_right(mat, n_cols);
		}
		else
		{
			copy_left(mat, n_cols);
		}

		return;
	}

	u32 const x_len = mat.width - n_cols;
	u32 const y_len = mat.height - n_rows;

	u32 src_x_begin = right ? 0 : n_cols;
	u32 dst_x_begin = src_x_begin + direction.x;

	u32 src_y_begin = 0;
	i32 dy = 0;
	if (up)
	{
		src_y_begin = n_rows;
		dy = 1;
	}
	else
	{
		src_y_begin = y_len - 1;
		dy = -1;
	}

	for (u32 iy = 0; iy < y_len; ++iy)
	{
		auto src_y = src_y_begin + dy * iy;
		auto dst_y = src_y + direction.y;

		auto src_begin = mat.row_begin(src_y) + src_x_begin;
		auto dst_begin = mat.row_begin(dst_y) + dst_x_begin;

		auto src_end = src_begin + x_len;

		std::copy(std::execution::par, src_begin, src_end, dst_begin);
	}
}


static void mandelbrot(AppState& state)
{
	auto& dst = state.iterations;

	auto const max_iter = state.max_iter;
	constexpr r64 limit = 4.0;

	auto const x_pos = state.screen_pos.x;
	auto const y_pos = state.screen_pos.y;

	auto const min_re = MBT_MIN_X + x_pos;
	auto const min_im = MBT_MIN_Y + y_pos;

	auto const re_step = screen_width(state) / dst.width;
	auto const im_step = screen_height(state) / dst.height;

	auto const do_mandelbrot = [&](Range2Du32 const& range)
	{
		auto x_ids = UnsignedRange(range.x_begin, range.x_end);
		auto y_ids = UnsignedRange(range.y_begin, range.y_end);

		auto y_id_begin = y_ids.begin();
		auto y_id_end = y_ids.end();
		auto x_id_begin = x_ids.begin();
		auto x_id_end = x_ids.end();

		auto const do_row = [&](u32 y)
		{
			r64 const ci = min_im + y * im_step;
			auto row = dst.row_begin(y);

			auto const do_x = [&](u32 x)
			{
				u32 iter = 0;
				r64 const cr = min_re + x * re_step;

				r64 re = 0.0;
				r64 im = 0.0;
				r64 re2 = 0.0;
				r64 im2 = 0.0;

				while (iter < max_iter && re2 + im2 <= limit)
				{
					im = (re + re) * im + ci;
					re = re2 - im2 + cr;
					im2 = im * im;
					re2 = re * re;

					++iter;
				}

				--iter;

				row[x] = iter;
			};

			std::for_each(std::execution::par, x_id_begin, x_id_end, do_x);
		};

		std::for_each(y_id_begin, y_id_end, do_row);
	};

	auto& shift = state.pixel_shift;

	auto do_left = shift.x > 0;
	auto do_top = shift.y > 0;
	//auto do_right = shift.x < 0;	
	//auto do_bottom = shift.y < 0;

	auto const n_cols = static_cast<u32>(std::abs(shift.x));
	auto const n_rows = static_cast<u32>(std::abs(shift.y));

	auto no_horizontal = n_cols == 0;
	auto no_vertical = n_rows == 0;

	auto r = get_range(dst);

	if (no_horizontal && no_vertical)
	{
		do_mandelbrot(r);
		return;
	}

	if (no_horizontal)
	{
		if (do_top)
		{
			r.y_end = n_rows;
		}
		else // if (do_bottom)
		{
			r.y_begin = dst.height - 1 - n_rows;
		}

		do_mandelbrot(r);
		return;
	}

	if (no_vertical)
	{
		if (do_left)
		{
			r.x_end = n_cols;
		}
		else // if (do_right)
		{
			r.x_begin = dst.width - 1 - n_cols;
		}

		do_mandelbrot(r);
		return;
	}

	auto r2 = r;

	if (do_top)
	{
		r.y_end = n_rows;
		r2.y_begin = n_rows;
	}
	else // if (do_bottom)
	{
		r.y_begin = dst.height - 1 - n_rows;
		r2.y_end = dst.height - 1 - n_rows;
	}

	if (do_left)
	{
		r2.x_end = n_cols;
	}
	else // if (do_right)
	{
		r2.x_begin = dst.width - 1 - n_cols;
	}

	do_mandelbrot(r);
	do_mandelbrot(r2);
}


static void render(image_t const& dst, AppState& state)
{	
	copy(state.iterations, state.pixel_shift);
	mandelbrot(state);

	draw(dst, state);
}


namespace app
{
	static void initialize_memory(AppState& state, PixelBuffer const& buffer)
	{
		state.render_new = true;

		state.screen_pos.x = 0.0;
		state.screen_pos.y = 0.0;

		state.zoom_level = 1.0;
		state.zoom_speed = ZOOM_SPEED_LOWER_LIMIT;

		state.max_iter = MAX_ITERATIONS_START;

		state.rgb_option = 1;

		auto const width = buffer.width;
		auto const height = buffer.height;

		state.iterations.width = width;
		state.iterations.height = height;
		state.iterations.data = (u32*)((u8*)(&state) + sizeof(u32) * width * height);
	}


	static void process_input(Input const& input, AppState& state)
	{
		constexpr r64 zoom_speed_factor_per_second = 0.1;
		constexpr r64 iteration_adjustment_factor = 0.02;

		auto zoom_speed_factor = 1.0 + zoom_speed_factor_per_second * input.dt_frame;

		state.pixel_shift = { 0, 0 };

		i32 const pixel_shift = static_cast<i32>(std::round(PIXELS_PER_SECOND * input.dt_frame));

		r64 const zoom_per_second = 0.5;
		auto const zoom = [&]() { return state.zoom_speed * (1.0 + zoom_per_second * input.dt_frame); };

		auto direction = false;

		// pan image with arrow keys
		if (input.keyboard.right_key.is_down)
		{
			auto distance_per_pixel = screen_width(state) / BUFFER_WIDTH;

			state.pixel_shift.x -= pixel_shift;
			state.screen_pos.x += pixel_shift * distance_per_pixel;

			direction = true;
			state.render_new = true;
		}
		if (input.keyboard.left_key.is_down)
		{
			auto distance_per_pixel = screen_width(state) / BUFFER_WIDTH;

			state.pixel_shift.x += pixel_shift;
			state.screen_pos.x -= pixel_shift * distance_per_pixel;

			direction = true;
			state.render_new = true;
		}
		if (input.keyboard.up_key.is_down)
		{
			auto distance_per_pixel = screen_height(state) / BUFFER_HEIGHT;

			state.pixel_shift.y += pixel_shift;
			state.screen_pos.y -= pixel_shift * distance_per_pixel;

			direction = true;
			state.render_new = true;
		}
		if (input.keyboard.down_key.is_down)
		{
			auto distance_per_pixel = screen_height(state) / BUFFER_HEIGHT;

			state.pixel_shift.y -= pixel_shift;
			state.screen_pos.y += pixel_shift * distance_per_pixel;

			direction = true;
			state.render_new = true;
		}

		// zoom speed with *, /
		if (input.keyboard.mult_key.is_down)
		{
			state.zoom_speed *= zoom_speed_factor;
			state.render_new = true;
		}
		if (input.keyboard.div_key.is_down)
		{
			state.zoom_speed = std::max(state.zoom_speed / zoom_speed_factor, ZOOM_SPEED_LOWER_LIMIT);

			state.render_new = true;
		}

		// zoom in/out with +, -
		if (input.keyboard.plus_key.is_down && !direction)
		{
			auto old_w = screen_width(state);
			auto old_h = screen_height(state);
			state.zoom_level *= zoom();
			auto new_w = screen_width(state);
			auto new_h = screen_height(state);

			state.screen_pos.x += 0.5 * (old_w - new_w);
			state.screen_pos.y += 0.5 * (old_h - new_h);

			state.render_new = true;
		}
		if (input.keyboard.minus_key.is_down && !direction)
		{
			auto old_w = screen_width(state);
			auto old_h = screen_height(state);
			state.zoom_level /= zoom();
			auto new_w = screen_width(state);
			auto new_h = screen_height(state);

			state.screen_pos.x += 0.5 * (old_w - new_w);
			state.screen_pos.y += 0.5 * (old_h - new_h);

			state.render_new = true;
		}

		// resolution with F, D
		if (input.keyboard.f_key.is_down)
		{
			u32 adj = static_cast<u32>(iteration_adjustment_factor * state.max_iter);
			adj = std::max(adj, 5u);

			state.max_iter = std::min(state.max_iter + adj, MAX_ITERATIONS_UPPER_LIMIT);
			state.render_new = true;
		}
		if (input.keyboard.d_key.is_down)
		{
			u32 adj = static_cast<u32>(iteration_adjustment_factor * state.max_iter);
			adj = std::max(adj, 5u);

			state.max_iter = std::max(state.max_iter - adj, MAX_ITERTAIONS_LOWER_LIMIT);
			state.render_new = true;
		}

		// color scheme with 1 - 6
		if (input.keyboard.one_key.is_down && state.rgb_option != 1)
		{			
			state.rgb_option = 1;
			state.render_new = true;
		}
		if (input.keyboard.two_key.is_down && state.rgb_option != 2)
		{
			state.rgb_option = 2;
			state.render_new = true;
		}
		if (input.keyboard.three_key.is_down && state.rgb_option != 3)
		{
			state.rgb_option = 3;
			state.render_new = true;
		}
		if (input.keyboard.four_key.is_down && state.rgb_option != 4)
		{
			state.rgb_option = 4;
			state.render_new = true;
		}
		if (input.keyboard.five_key.is_down && state.rgb_option != 5)
		{
			state.rgb_option = 5;
			state.render_new = true;
		}
		if (input.keyboard.six_key.is_down && state.rgb_option != 6)
		{
			state.rgb_option = 6;
			state.render_new = true;
		}
	}


	void update_and_render(AppMemory& memory, Input const& input, PixelBuffer const& buffer)
	{
		auto const state_sz = sizeof(AppState);
		auto const iter_sz = sizeof(u32) * buffer.width * buffer.height;

		auto const required_sz = state_sz + iter_sz;

		assert(required_sz <= memory.permanent_storage_size);

		auto& state = *(AppState*)memory.permanent_storage;		

		if (!memory.is_app_initialized)
		{
			initialize_memory(state, buffer);
			memory.is_app_initialized = true;
		}

		process_input(input, state);

		if (!state.render_new)
		{
			return;
		}

		auto buffer_image = make_buffer_image(buffer);
		render(buffer_image, state);

		state.render_new = false;
	}


	void end_program()
	{
		
	}
}