#include "render.hpp"
#include "render_include.hpp"
#include "colors.hpp"
#include "range_list.hpp"

#include <cassert>
#include <algorithm>
#include <functional>
#include <cmath>


#ifndef NO_CPP17

#include <execution>
// -ltbb
// sudo apt install libtbb-dev


template <class LIST_T, class FUNC_T>
static void do_for_each(LIST_T const& list, FUNC_T const& func)
{
	std::for_each(std::execution::par, list.begin(), list.end(), func);
}

#else

template <class LIST_T, class FUNC_T>
static void do_for_each(LIST_T const& list, FUNC_T const& func)
{
	std::for_each(list.begin(), list.end(), func);
}

#endif // !NO_CPP17


constexpr u32 N_THREADS = 16;


using id_func_t = std::function<void(u32)>;


class ThreadProcess
{
public:
	u32 thread_id = 0;
	id_func_t process;
};


using ProcList = std::array<ThreadProcess, N_THREADS>;


static ProcList make_proc_list(id_func_t const& id_func)
{
	ProcList list = { 0 };

	for (u32 i = 0; i < N_THREADS; ++i)
	{
		list[i] = { i, id_func };
	}

	return list;
}


static void execute_procs(ProcList const& list)
{
	auto const func = [](ThreadProcess const& t) { t.process(t.thread_id); };

	do_for_each(list, func);
}


static void process_rows(u32 height, id_func_t const& row_func)
{
	auto const rows_per_thread = height / N_THREADS;

	auto const thread_proc = [&](u32 id)
	{
		auto y_begin = id * rows_per_thread;
		auto y_end = (id == N_THREADS - 1 ? height : (id + 1) * rows_per_thread);

		for (u32 y = y_begin; y < y_end; ++y)
		{
			row_func(y);
		}
	};

	execute_procs(make_proc_list(thread_proc));
}


namespace mat
{
	template <typename T>
	class View
	{
	public:

		T* matrix_data = nullptr;
		u32 matrix_width = 0;

		u32 x_begin = 0;
		u32 x_end = 0;
		u32 y_begin = 0;
		u32 y_end = 0;

		u32 width = 0;
		u32 height = 0;
	};


	template <typename T>
	static T* row_begin(Matrix<T> const& mat, u32 y)
	{
		assert(y < mat.height);

		auto offset = y * mat.width;

		auto ptr = mat.data + (u64)(offset);
		assert(ptr);

		return ptr;
	}


	template <typename T>
	static View<T> sub_view(Matrix<T> const& matrix, Range2Du32 const& range)
	{
		assert(matrix.width);
		assert(matrix.height);
		assert(matrix.data);

		View<T> sub_view{};

		sub_view.matrix_data = matrix.data;
		sub_view.matrix_width = matrix.width;
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


	template <typename T>
	static void copy(View<T> const& src, View<T> const& dst)
	{
		assert(src.matrix_data);
		assert(dst.matrix_data);
		assert(src.width == dst.width);
		assert(src.height == dst.height);

		auto const row_func = [&](u32 y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x];
			}
		};

		process_rows(src.height, row_func);
	}


	template <typename T>
	static void copy_xy(Matrix<T> const& src, Matrix<T> const& dst, Range2Du32 const& r_src, Range2Du32 const & r_dst, u32 dst_x, u32 dst_y)
	{
		auto x_offset = dst_x - r_dst.x_begin;
		auto y_offset = dst_y - r_dst.y_begin;

		auto src_x = r_src.x_begin + x_offset;
		auto src_y = r_src.y_begin + y_offset;

		auto src_i = src_y * src.width + src_x;
		auto dst_i = dst_y * src.width + dst_x;    

		dst.data[dst_i] = src.data[src_i];
	}
}


static Pixel to_pixel(u8 r, u8 g, u8 b)
{
    Pixel p{};
    p.red = r;
    p.green = g;
    p.blue = b;
    p.alpha = 255;

    return p;
}


static bool in_range(u32 x, u32 y, Range2Du32 const& range)
{
    return 
        range.x_begin <= x &&
        x < range.x_end &&
        range.y_begin <= y &&
        y < range.y_end;
}


static u32 mandelbrot_iter(r64 cx, r64 cy, u32 iter_limit)
{
	u32 iter = 0;

    r64 mx = 0.0;
    r64 my = 0.0;
    r64 mx2 = 0.0;
    r64 my2 = 0.0;

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


static i32 color_index(u32 iter, u32 iter_limit)
{
	if (iter >= iter_limit)
	{
		return -1;
	}	

	u32 min = 0;
	u32 max = 0;
	u32 n_colors = 8;

	for (auto i : color_levels)
	{
		n_colors *= 2;
		min = max;
		max = i;

		if (iter < max)
		{
			return (iter - min) % n_colors * (N_COLORS / n_colors);
		}
	}	

	min = max;
	
	return (iter - min) % N_COLORS;
}




/*
static void copy_range(Mat2Di32 const& src, Mat2Di32 const& dst, Range2Du32 const& r_src, Range2Du32 const & r_dst)
{
	auto const r_width = r_src.x_end - r_src.x_begin;
	auto const r_height = r_src.y_end - r_src.y_begin;

	auto src_begin = src.data + r_src.y_begin * src.width + r_src.x_begin;
	auto dst_begin = dst.data + r_dst.y_begin * dst.width + r_dst.x_begin;

	auto const copy_row = [&](u32 y)
	{
		auto s = src_begin + y * src.width;
		auto d = dst_begin + y * dst.width;
		for(u32 x = 0; x < r_width; ++x)
		{
			d[x] = s[x];
		}
	};

	process_rows(r_height, copy_row);
}


static void mbt_range()
{

}
*/
/*
static void draw_xy(Mat2Di32 const& src, Image const& dst, ChannelOptions const& options, u32 x, u32 y)
{
	auto pixel_index = y * dst.width + x;
	auto color_id = src.data[pixel_index];

	if(color_id < 0)
    {
        dst.data[pixel_index] = to_pixel(0, 0, 0); // TODO: platform pixel
    }
    else
    {
        u8 color_map[] = { palettes[0][color_id], palettes[1][color_id], palettes[2][color_id] };
        dst.data[pixel_index] = to_pixel(color_map[options.channel1], color_map[options.channel2], color_map[options.channel3]);
    }
}
*/

static void draw_xy(Mat2Du32 const& iter_src, Image const& dst, ChannelOptions const& options, u32 iter_limit, u32 x, u32 y)
{
	auto iter = mat::row_begin(iter_src, y)[x];
	auto d = mat::row_begin(dst, y);

	if (iter >= iter_limit)
	{
		d[x] = to_pixel(0, 0, 0);
	}
	else
	{
		auto color_id = color_index(iter, iter_limit);
		u8 color_map[] = { palettes[0][color_id], palettes[1][color_id], palettes[2][color_id] };
		d[x] = to_pixel(color_map[options.channel1], color_map[options.channel2], color_map[options.channel3]);
	}
}


static void process_and_draw(AppState const& state)
{
	auto& iters = state.iterations[state.current_id];
	auto& pixels = state.screen_buffer;
	auto& options = state.channel_options;
	auto iter_limit = state.app_input.iter_limit;
	
	auto& prev_iters = state.iterations[state.prev_id];

	auto width = pixels.width;
	auto height = pixels.height;

	auto const mbt_row = [&](u32 y) 
	{
		auto iter_row = mat::row_begin(iters, y);
		r64 cy = std::fma((r64)y, state.my_step, state.min_my); // state.min_my + y * state.my_step;

		for(u32 x = 0; x < width; ++x)
		{
			if(in_range(x, y, state.copy_dst))
			{
				mat::copy_xy(prev_iters, iters, state.copy_src, state.copy_dst, x, y);
			}
			else
			{
				r64 cx = std::fma((r64)x, state.mx_step, state.min_mx); // state.min_mx + x * state.mx_step;
				auto iter = mandelbrot_iter(cx, cy, state.app_input.iter_limit);

				iter_row[x] = iter;
			}

			draw_xy(iters, pixels, options, iter_limit, x, y);
		}
	};

	process_rows(height, mbt_row);
}


static void draw(AppState const& state)
{
	auto& iters = state.iterations[state.current_id];
	auto& pixels = state.screen_buffer;
	auto& options = state.channel_options;
	auto iter_limit = state.app_input.iter_limit;

	auto const draw_row = [&](u32 y)
	{
		for(u32 x = 0; x < pixels.width; ++x)
		{
			draw_xy(iters, pixels, options, iter_limit, x, y);
		}
	};

	process_rows(pixels.height, draw_row);
}


void render(AppState& state)
{
	auto width = state.screen_buffer.width;
	auto height = state.screen_buffer.height;

	if(!state.app_input.render_new && !state.app_input.draw_new)
    {
        return;
    }

    set_rgb_channels(state.channel_options, state.app_input.rgb_option);

	if(state.app_input.render_new)
	{
		state.current_id = state.prev_id;
		state.prev_id = !state.prev_id;
		
		auto ranges = get_ranges(make_range(width, height), state.app_input.pixel_shift);
		
		state.min_mx = MBT_MIN_X + state.app_input.mbt_pos.x;
		state.min_my = MBT_MIN_Y + state.app_input.mbt_pos.y;
		state.mx_step = state.app_input.mbt_screen_width / width;
		state.my_step = state.app_input.mbt_screen_height / height;

		state.copy_src = ranges.copy_src;
        state.copy_dst = ranges.copy_dst;

		process_and_draw(state);

		state.app_input.draw_new = false;
	}
    
    if(state.app_input.draw_new)
    {       
		draw(state);
    }
}


u32 get_rgb_combo_qty()
{
	constexpr auto n = num_rgb_combinations();

	return n;
}