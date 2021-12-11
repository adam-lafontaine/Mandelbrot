#include "app.hpp"
#include "../utils/index_range.hpp"
#include "../utils/image.hpp"

#include <cassert>
#include <algorithm>
#include <array>
#include <execution>
#include <cmath>
#include <functional>


constexpr r64 MBT_MIN_X = -2.0;
constexpr r64 MBT_MAX_X = 0.7;
constexpr r64 MBT_MIN_Y = -1.2;
constexpr r64 MBT_MAX_Y = 1.2;
constexpr r64 MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
constexpr r64 MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;

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

	image_t buffer_image;

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


static pixel_t to_platform_pixel(pixel_t const& p)
{
	return to_platform_pixel(p.red, p.green, p.blue);
}


constexpr std::array< std::array<u8, 16>, 3> palettes16 = 
{ {
	{ 66, 25,  9,  4,   0,  12,  24,  57, 134, 211, 241, 248, 255, 204, 153, 106 },
	{ 30,  7,  1,  4,   7,  44,  82, 125, 181, 236, 233, 201, 170, 128,  87,  57 },
	{ 15, 26, 47, 73, 100, 138, 177, 209, 229, 248, 191,  95,   0,   0,   0,   3 }
} };


static constexpr std::array<u8, 32> make_palette32(u32 c)
{
	assert(c < 3);

	auto& palette16 = palettes16[c];

	auto const lerp = [](u8 a, u8 b, r64 t)
	{
		return static_cast<u8>(a + t * (b - a));
	};

	std::array<u8, 32> palette = {};

	for (u32 i = 0; i < 15; ++i)
	{
		u32 p = 2 * i;
		u32 p1 = p + 1;
		u32 i1 = i + 1;
		palette[p] = palette16[i];
		palette[p1] = lerp(palette16[i], palette16[i1], 0.5);
	}

	palette[30] = palette16[15];
	palette[31] = lerp(palette16[15], palette16[0], 0.5);

	return palette;
}


constexpr std::array< std::array<u8, 32>, 3> palettes32 =
{ {
	make_palette32(0),
	make_palette32(1),
	make_palette32(2)
} };


static constexpr std::array<u8, 64> make_palette64(u32 c)
{
	assert(c < 3);

	auto& palette16 = palettes16[c];

	auto const lerp = [](u8 a, u8 b, r64 t)
	{
		return static_cast<u8>(a + t * (b - a));
	};

	std::array<u8, 64> palette = {};

	for (u32 i = 0; i < 15; ++i)
	{
		u32 p = 2 * i;
		u32 p1 = p + 1;
		u32 p2 = p1 + 1;
		u32 p3 = p2 + 1;
		u32 i1 = i + 1;
		palette[p] = palette16[i];
		palette[p1] = lerp(palette16[i], palette16[i1], 0.25);
		palette[p2] = lerp(palette16[i], palette16[i1], 0.5);
		palette[p3] = lerp(palette16[i], palette16[i1], 0.75);
	}

	palette[60] = palette16[15];
	palette[61] = lerp(palette16[15], palette16[0], 0.25);
	palette[62] = lerp(palette16[15], palette16[0], 0.5);
	palette[63] = lerp(palette16[15], palette16[0], 0.75);

	return palette;
}


constexpr std::array< std::array<u8, 64>, 3> palettes64 =
{ {
	make_palette64(0),
	make_palette64(1),
	make_palette64(2)
} };


static pixel_t to_rgb_64(u32 iterations, u32 max_iter, u32 rgb_option)
{
	if (iterations >= max_iter)
	{
		return to_platform_pixel(0, 0, 0);
	}

	auto i = iterations % 64;

	u8 color_map[] = { palettes64[0][i], palettes64[1][i], palettes64[2][i] };

	u32 c1 = 0;
	u32 c2 = 0;
	u32 c3 = 0;

	switch (rgb_option)
	{
	case 1:
		c1 = 0;
		c2 = 1;
		c3 = 2;
		break;
	case 2:
		c1 = 0;
		c2 = 2;
		c3 = 1;
		break;
	case 3:
		c1 = 1;
		c2 = 0;
		c3 = 2;
		break;
	case 4:
		c1 = 1;
		c2 = 2;
		c3 = 0;
		break;
	case 5:
		c1 = 2;
		c2 = 0;
		c3 = 1;
		break;
	case 6:
		c1 = 2;
		c2 = 1;
		c3 = 0;
		break;
	}

	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_32(u32 iterations, u32 max_iter, u32 rgb_option)
{
	if (iterations >= max_iter)
	{
		return to_platform_pixel(0, 0, 0);
	}

	auto i = iterations % 32;

	u8 color_map[] = { palettes32[0][i], palettes32[1][i], palettes32[2][i] };

	u32 c1 = 0;
	u32 c2 = 0;
	u32 c3 = 0;

	switch (rgb_option)
	{
	case 1:
		c1 = 0;
		c2 = 1;
		c3 = 2;
		break;
	case 2:
		c1 = 0;
		c2 = 2;
		c3 = 1;
		break;
	case 3:
		c1 = 1;
		c2 = 0;
		c3 = 2;
		break;
	case 4:
		c1 = 1;
		c2 = 2;
		c3 = 0;
		break;
	case 5:
		c1 = 2;
		c2 = 0;
		c3 = 1;
		break;
	case 6:
		c1 = 2;
		c2 = 1;
		c3 = 0;
		break;
	}

	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static pixel_t to_rgb_16(u32 iterations, u32 max_iter, u32 rgb_option)
{
	if (iterations >= max_iter)
	{
		return to_platform_pixel(0, 0, 0);
	}

	auto i = iterations % 16;

	u8 color_map[] = { palettes16[0][i], palettes16[1][i], palettes16[2][i] };

	u32 c1 = 0;
	u32 c2 = 0;
	u32 c3 = 0;

	switch (rgb_option)
	{
	case 1:
		c1 = 0;
		c2 = 1;
		c3 = 2;
		break;
	case 2:
		c1 = 0;
		c2 = 2;
		c3 = 1;
		break;
	case 3:
		c1 = 1;
		c2 = 0;
		c3 = 2;
		break;
	case 4:
		c1 = 1;
		c2 = 2;
		c3 = 0;
		break;
	case 5:
		c1 = 2;
		c2 = 0;
		c3 = 1;
		break;
	case 6:
		c1 = 2;
		c2 = 1;
		c3 = 0;
		break;
	case 7:
		r64 r = (r64)iterations / max_iter;
		u8 p = std::sqrt(r) > 0.7 ? 0 : 255;
		return to_platform_pixel(p, p, p);
	}

	return to_platform_pixel(color_map[c1], color_map[c2], color_map[c3]);
}


static void draw(image_t const& dst, AppState const& state)
{
	auto& mat = state.iterations;

	auto [mat_min, mat_max] = std::minmax_element(mat.begin(), mat.end());
	auto min = *mat_min;
	auto max = *mat_max;

	auto diff = max - min;

	auto const to_platform_color = [&](u32 i) 
	{		
		return to_rgb_64(i - min, diff, state.rgb_option);
	};

	std::transform(std::execution::par, mat.begin(), mat.end(), dst.begin(), to_platform_color);
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
				r64 re2_old = 0.0;
				r64 im2_old = 0.0;

				while (iter < max_iter && re2 + im2 <= limit)
				{
					re2_old = re2;
					im2_old = im2;
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

		std::for_each(std::execution::par, y_id_begin, y_id_end, do_row);
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

		state.buffer_image = make_buffer_image(buffer);

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
		if (input.keyboard.seven_key.is_down && state.rgb_option != 7)
		{
			state.rgb_option = 7;
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

		render(state.buffer_image, state);

		state.render_new = false;
	}


	void end_program()
	{
		
	}
}