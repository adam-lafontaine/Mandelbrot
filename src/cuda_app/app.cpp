#include "../app/app.hpp"
#include "../app/render.hpp"
#include "cuda_types.hpp"

#include <cmath>


constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = 50;
constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = 50000;
constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
constexpr r64 ZOOM_SPEED_LOWER_LIMIT = 1.0;


static void process_input(Input const& input, AppState& state)
{
	constexpr r64 zoom_speed_factor_per_second = 0.1;
	constexpr r64 iteration_adjustment_factor = 0.02;

	auto zoom_speed_factor = 1.0 + zoom_speed_factor_per_second * input.dt_frame;

	state.pixel_shift = { 0, 0 };

	i32 const pixel_shift = static_cast<i32>(std::round(app::PIXELS_PER_SECOND * input.dt_frame));

	r64 const zoom_per_second = 0.5;
	auto const zoom = [&]() { return state.zoom_speed * (1.0 + zoom_per_second * input.dt_frame); };

	auto direction = false;

	// pan image with arrow keys
	if (input.keyboard.right_key.is_down)
	{
		auto distance_per_pixel = screen_width(state) / app::BUFFER_WIDTH;

		state.pixel_shift.x -= pixel_shift;
		state.screen_pos.x += pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
	if (input.keyboard.left_key.is_down)
	{
		auto distance_per_pixel = screen_width(state) / app::BUFFER_WIDTH;

		state.pixel_shift.x += pixel_shift;
		state.screen_pos.x -= pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
	if (input.keyboard.up_key.is_down)
	{
		auto distance_per_pixel = screen_height(state) / app::BUFFER_HEIGHT;

		state.pixel_shift.y += pixel_shift;
		state.screen_pos.y -= pixel_shift * distance_per_pixel;

		direction = true;
		state.render_new = true;
	}
	if (input.keyboard.down_key.is_down)
	{
		auto distance_per_pixel = screen_height(state) / app::BUFFER_HEIGHT;

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


namespace app
{
	static image_t make_buffer_image(ScreenBuffer const& buffer)
	{
		assert(buffer.bytes_per_pixel == RGBA_CHANNELS);

		image_t image{};

		image.width = buffer.width;
		image.height = buffer.height;
		image.data = (pixel_t*)buffer.memory;

		return image;
	}


	static AppState& get_state(AppMemory& memory, ScreenBuffer const& buffer)
	{
		auto const state_sz = sizeof(AppState);
		auto const iter_sz = sizeof(u32) * buffer.width * buffer.height;

		auto const required_sz = state_sz + iter_sz;

		assert(required_sz <= memory.permanent_storage_size);

		auto& state = *(AppState*)memory.permanent_storage;

		return state;
	}


    static DeviceState& get_device_memory(AppMemory& memory, ScreenBuffer const& buffer)
    {
        auto const state_sz = sizeof(DeviceState);
        auto const iter_sz = sizeof(u32) * buffer.width * buffer.height;

        auto const required_sz = state_sz + iter_sz;
    }


	void initialize_memory(AppMemory& memory, ScreenBuffer const& buffer)
	{
		auto& state = get_state(memory, buffer);

		state.render_new = true;

		state.screen_buffer = make_buffer_image(buffer);

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

		memory.is_app_initialized = true;
	}


	void update_and_render(AppMemory& memory, Input const& input, ScreenBuffer const& buffer)
	{
		if (!memory.is_app_initialized)
		{
			return;
		}

		auto& state = get_state(memory, buffer);

		process_input(input, state);

		if (!state.render_new)
		{
			return;
		}

		render(state.screen_buffer, state);

		state.render_new = false;
	}


	void end_program()
	{
		
	}
}