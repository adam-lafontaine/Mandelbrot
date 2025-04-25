#include "app_include.hpp"


/* state data */

namespace game_mbt
{
    class StateData
    {
    public:

        Vec2Du32 screen_dims;

        Stopwatch frame_sw;
        f64 dt_frame;
        InputCommand in_cmd;

        ColorIdMatrix color_ids;

        ColorFormat format;

        f32 zoom_rate = 1.0f;
        f32 zoom = 1.0f;

        u32 iter_limit = MAX_ITERATIONS_START;

        Vec2D<fmbt> mbt_scale;
        Vec2D<fmbt> mbt_pos;
        Vec2D<fmbt> mbt_delta;

        Vec2D<i16> pixel_shift;

        u8 n_copy = 0;
        Rect2Du32 copy_src;
        Rect2Du32 copy_dst;

        u8 n_proc = 0;
        Rect2Du32 proc_dst[2];        

        bool render_new = false;
        bool enabled = true;
    };


    static void reset_state_data(StateData& data)
    {
        auto w = data.screen_dims.x;
        auto h = data.screen_dims.y;

        data.dt_frame = 1.0 / 60;

        data.format = colors::make_color_format();

        data.zoom_rate = ZOOM_RATE_LOWER_LIMIT;
        data.zoom = 1.0f;

        data.iter_limit = MAX_ITERATIONS_START;

        data.mbt_scale = mbt_screen_dims(data.zoom);

        data.mbt_pos = { MBT_MIN_X, MBT_MIN_Y };

        data.mbt_delta = {
            data.mbt_scale.x / w,
            data.mbt_scale.y / h
        };

        data.pixel_shift = { 0 };

        auto full_rect = img::make_rect(w, h);

        data.n_copy = 0;
        data.copy_src = full_rect;
        data.copy_dst = full_rect;

        data.n_proc = 1;
        data.proc_dst[0] = full_rect;
        data.proc_dst[1] = full_rect;        

        data.render_new = true;
        data.enabled = true;
    }


    static inline StateData& get_data(AppState const& state)
    {
        return *state.data_;
    }


    static void destroy_state_data(AppState& state)
    {
        if (!state.data_)
        {
            return;
        }

        auto& data = get_data(state);

        destroy_color_ids(data.color_ids);

        mem::free(state.data_);
        state.data_ = 0;
    }


    static bool create_state_data(AppState& state)
    {
        auto data_p = mem::alloc<StateData>("StateData");
        if (!data_p)
        {
            return false;
        }

        state.data_ = data_p;

        auto& data = get_data(state);

        data = {}; // important!

        return true;
    }
}


/* update */

namespace game_mbt
{

namespace ns_update_state
{
    static void update_color_format(b8 change, StateData& data)
    {
        if (!change)
        {
            return;
        }
        
        data.format = colors::make_color_format();
    }


    static void update_zoom_rate(i8 idelta, StateData& data)
    {
        constexpr f32 factor_per_second = 0.1f;

        auto factor = 1.0 + idelta * factor_per_second * data.dt_frame;

        auto rate = data.zoom_rate * (f32)factor;

        data.zoom_rate = num::max(rate, ZOOM_RATE_LOWER_LIMIT);
    }


    static void update_zoom(i8 idelta, StateData& data)
    {
        constexpr f32 zoom_per_second = 0.5f;

        auto factor = 1.0 + idelta * zoom_per_second * data.dt_frame;

        data.zoom *= (f32)factor;

        auto old_dims = data.mbt_scale;

        data.mbt_scale = mbt_screen_dims(data.zoom);

        data.mbt_pos.x += 0.5 * (old_dims.x - data.mbt_scale.x);
        data.mbt_pos.y += 0.5 * (old_dims.y - data.mbt_scale.y);

        auto w = (fmbt)data.screen_dims.x;
        auto h = (fmbt)data.screen_dims.y;

        data.mbt_delta = {
            data.mbt_scale.x / w,
            data.mbt_scale.y / h
        };
    }


    static void update_mbt_pos(Vec2D<i8> ishift, StateData& data)
    {
        auto n_px = PIXELS_PER_SECOND * data.dt_frame;

        data.pixel_shift = {
            num::round_to_signed<i16>(ishift.x * n_px),
            num::round_to_signed<i16>(ishift.y * n_px),
        };

        auto dx = data.mbt_delta.x * data.pixel_shift.x;
        auto dy = data.mbt_delta.y * data.pixel_shift.y;

        data.mbt_pos.x += dx;
        data.mbt_pos.y += dy;
    }


    static void update_iter_limit(i8 idelta, StateData& data)
    {
        constexpr f32 factor = 0.005f;

        auto min = (f32)MAX_ITERTAIONS_LOWER_LIMIT;
        auto max = (f32)MAX_ITERATIONS_UPPER_LIMIT;

        auto delta = idelta * num::max(factor * data.iter_limit, 5.0f);

        auto limit = num::clamp(data.iter_limit + delta, min, max);

        data.iter_limit = num::round_to_unsigned<u32>(limit);
    }


    static void update_ranges(StateData& data)
    {
        data.n_copy = 0;
        data.n_proc = 0;

        if (!data.render_new)
        {
            return;
        }

        auto w = data.screen_dims.x;
        auto h = data.screen_dims.y;

        auto full_range = img::make_rect(w, h);

        auto shift = data.pixel_shift;

        if (!shift.x && !shift.y)
        {
            data.n_copy = 0;
            data.n_proc = 1;
            data.proc_dst[0] = full_range;

            return;
        }

        auto copy_right = shift.x < 0;
        auto copy_left = shift.x > 0;
        auto copy_down = shift.y < 0;
        auto copy_up = shift.y > 0;

        auto const n_cols = num::abs(shift.x);
        auto const n_rows = num::abs(shift.y);

        auto& src = data.copy_src;
        auto& dst = data.copy_dst;

        src = full_range;
        dst = full_range;

        data.n_copy = 1;
        data.n_proc = 0;

        if (copy_up || copy_down)
        {
            data.n_proc++;
            auto& proc = data.proc_dst[data.n_proc - 1];
            proc = full_range;

            if(copy_up)
            {
                src.y_begin = n_rows;
                dst.y_end -= n_rows;

                proc.y_begin = dst.y_end;
            }
            else if (copy_down)
            {
                dst.y_begin = n_rows;
                src.y_end -= n_rows;

                proc.y_end = dst.y_begin;
            }
        }

        if (copy_right || copy_left)
        {
            data.n_proc++;
            auto& proc = data.proc_dst[data.n_proc - 1];
            proc = full_range;

            if(copy_left)
            {
                src.x_begin = n_cols;
                dst.x_end -= n_cols;

                proc.x_begin = dst.x_end;                
            }
            else if(copy_right)
            {
                dst.x_begin = n_cols;
                src.x_end -= n_cols;

                proc.x_end = dst.x_begin;
            }
        }
    }
}


    static void update_state(InputCommand const& cmd, StateData& data)
    {
        namespace ns = ns_update_state;

        static_assert(sizeof(cmd) == sizeof(cmd.any));

        data.in_cmd = cmd;

        ns::update_color_format(cmd.change_color, data);
        ns::update_zoom_rate(cmd.zoom_rate, data);

        if (!cmd.direction)
        {
            ns::update_zoom(cmd.zoom, data);
        }

        ns::update_mbt_pos(cmd.shift, data);
        ns::update_iter_limit(cmd.resolution, data);        

        ns::update_ranges(data);
    }


    static void update_color_ids(StateData& data)
    {
        if (!data.enabled || !data.render_new)
        {
            return;
        }

        if (data.n_copy)
        {
            proc_copy(data.color_ids, data.copy_src, data.copy_dst);
        }

        for (u32 i = 0; i < data.n_proc; i++)
        {            
            proc_mbt(data.color_ids, data.proc_dst[i], data.mbt_pos, data.mbt_delta, data.iter_limit);
        }        
    }

}


/* api */

namespace game_mbt
{
    AppResult init(AppState& state)
    {
        AppResult result;

        if (!create_state_data(state))
        {
            result.error_code = 1;
            return result;
        }

        result.app_dimensions = { 
            BUFFER_WIDTH,
            BUFFER_HEIGHT
         };

        result.success = true;
        result.error_code = 0;

        return result;
    }


    AppResult init(AppState& state, Vec2Du32 available_dims)
    {
        AppResult result;

        result.app_dimensions = available_dims;
        result.success = true;

        return result;
    }


    bool set_screen_memory(AppState& state, image::ImageView screen)
    {
        state.screen = screen;

        auto& data = get_data(state);

        auto w = state.screen.width;
        auto h = state.screen.height;

        if (!create_color_ids(data.color_ids, w, h))
        {
            return false;
        }

        data.screen_dims = { w, h };
        
        reset_state_data(data);

        proc_mbt(data.color_ids, data.proc_dst[0], data.mbt_pos, data.mbt_delta, data.iter_limit);

        data.frame_sw.start();

        return true;
    }


    void update(AppState& state, input::Input const& input)
    {       
        auto& data = get_data(state);

        data.dt_frame = data.frame_sw.get_time_sec();
        data.frame_sw.start();        

        auto cmd = map_input(input);

        data.render_new = cmd.any;
        if (data.render_new)
        {
            data.color_ids.swap();
        }

        update_state(cmd, data);
        update_color_ids(data);

        proc_render(data.color_ids, state.screen, data.format);
        
        data.render_new = false;
    }


    void reset(AppState& state)
    {
        auto& data = get_data(state);
        reset_state_data(data);
    }


    void close(AppState& state)
    {
        destroy_state_data(state);
    }


    cstr decode_error(AppResult const& result)
    {
        switch (result.error_code)
        {
        case 1: return "create_state_data";            

        default: return "OK";
        }
    }
}


#include "app_libs.cpp"