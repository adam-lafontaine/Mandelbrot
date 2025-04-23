#include "app.hpp"
#include "../../../libs/alloc_type/alloc_type.hpp"
#include "../../../libs/util/numeric.hpp"
#include "../../../libs/util/stopwatch.hpp"


namespace game_mbt
{
    namespace img = image;
    namespace num = numeric;
    namespace mb = memory_buffer;

    using fmbt = double;

    using p32 = img::Pixel;
    using ImageView = img::ImageView;
    using Input = input::Input;


    #ifdef SDL2_WASM

	constexpr u32 BUFFER_WIDTH = 640;
	constexpr u32 BUFFER_HEIGHT = BUFFER_WIDTH * 8 / 9;

	#else

	// allocate memory
	constexpr u32 BUFFER_HEIGHT = 800;
	constexpr u32 BUFFER_WIDTH = BUFFER_HEIGHT * 9 / 8;

	#endif

	constexpr u32 PIXELS_PER_SECOND = (u32)(0.2 * BUFFER_HEIGHT);

    constexpr u32 MAX_ITERTAIONS_LOWER_LIMIT = 50;
    constexpr u32 MAX_ITERATIONS_UPPER_LIMIT = 1000;
    constexpr u32 MAX_ITERATIONS_START = MAX_ITERTAIONS_LOWER_LIMIT;
    constexpr f32 ZOOM_RATE_LOWER_LIMIT = 1.0f;


    constexpr fmbt MBT_MIN_X = -2.0;
    constexpr fmbt MBT_MAX_X = 0.7;
    constexpr fmbt MBT_MIN_Y = -1.2;
    constexpr fmbt MBT_MAX_Y = 1.2;
    constexpr fmbt MBT_WIDTH = MBT_MAX_X - MBT_MIN_X;
    constexpr fmbt MBT_HEIGHT = MBT_MAX_Y - MBT_MIN_Y;
}


#include "map_input.cpp"
#include "colors.cpp"

/* color ids */

namespace game_mbt
{
    using ColorTable = colors::Palette<colors::N_COLORS>;
    using ColorFormat = colors::ColorFormat;
    using ColorId = colors::ColorId<colors::N_COLORS>;


    class ColorIdMatrix
    {
    private:
        u8 p = 1;
        u8 c = 0;

    public:

        MatrixView2D<ColorId> data_[2];
        
        //MatrixView2D<ColorId> prev() { return data_[p]; }
        //MatrixView2D<ColorId> curr() { return data_[c]; }

        MatrixView2D<ColorId> prev() const { return data_[p]; }
        MatrixView2D<ColorId> curr() const { return data_[c]; }

        void swap() { p = c; c = !p; }

        MemoryBuffer<ColorId> buffer;
    };


    static void destroy_color_ids(ColorIdMatrix& mat)
    {
        mb::destroy_buffer(mat.buffer);
    }


    static bool create_color_ids(ColorIdMatrix& mat, u32 width, u32 height)
    {
        auto n = width * height;

        if (!mb::create_buffer(mat.buffer, n * 2, "color_ids"))
        {
            return false;
        }

        for (u32 i = 0; i < 2; i++)
        {
            auto span = span::push_span(mat.buffer, n);
            mat.data_[i].matrix_data_ = span.data;
            mat.data_[i].width = width;
            mat.data_[i].height = height;
        }        

        return true;
    }


    static inline auto to_span(MatrixView2D<ColorId> const& mat)
    {
        return img::to_span(mat);
    }


    static inline auto sub_view(MatrixView2D<ColorId> const& mat, Rect2Du32 const& range)
    {
        return img::sub_view(mat, range);
    }


    static void copy(ColorIdMatrix const& mat, Rect2Du32 r_src, Rect2Du32 r_dst)
    {
        auto src = sub_view(mat.prev(), r_src);
        auto dst = sub_view(mat.curr(), r_dst);

        assert(src.width == dst.width);
        assert(src.height == dst.height);

        auto w = src.width;
        auto h = src.height;

        auto s = src.matrix_data_ + r_src.x_begin;
        auto d = dst.matrix_data_ + dst.x_begin;

        auto stride = mat.curr().width;

        for (u32 y = 0; y < h; y++)
        {
            span::copy(span::make_view(s, w), span::make_view(d, w));

            s += stride;
            d += stride;
        }
    }




}


/* render */

namespace game_mbt
{
    static p32 color_at(ColorId id, ColorFormat format)
    {
        static constexpr ColorTable Color_Table = colors::make_palette<colors::N_COLORS>();

        auto r = Color_Table.channels[format.R][id.value];
        auto g = Color_Table.channels[format.G][id.value];
        auto b = Color_Table.channels[format.B][id.value];

        return img::to_pixel(r, g, b);
    }


    static void render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format)
    {
        auto s = to_span(src.curr());
        auto d = img::to_span(dst);

        assert(s.length == d.length);

        for (u32 i = 0; i < s.length; i++)
        {
            d.data[i] = color_at(s.data[i], format);
        }
    }
}


/* mandelbrot */

namespace game_mbt
{
    static ColorId to_color_id(u32 iter, u32 iter_limit)
    {
        auto r = num::min((f32)iter / iter_limit, 1.0f);

        auto id = num::round_to_unsigned<u32>(r * ColorId::max);

        return ColorId::make(id);
    }


    static u32 mandelbrot_iter(fmbt cx, fmbt cy, u32 iter_limit)
    {
        u32 iter = 0;
    
        fmbt mx = 0.0;
        fmbt my = 0.0;
        fmbt mx2 = 0.0;
        fmbt my2 = 0.0;
    
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


    static void mbt_proc(ColorIdMatrix const& mat, Rect2Du32 r_dst, Vec2D<fmbt> const& begin, Vec2D<fmbt> const& delta, u32 limit)
    {
        auto view = mat.curr();
        auto dst = sub_view(view, r_dst);

        auto w = dst.width;
        auto h = dst.height;

        auto stride = view.width;

        auto d = dst.matrix_data_ + dst.x_begin;

        auto cy = begin.y;
        auto cx = begin.x;

        for (u32 y = 0; y < h; y++)
        {
            for (u32 x = 0; x < w; x++)
            {
                auto iter = mandelbrot_iter(cx, cy, limit);                
                d[x] = to_color_id(iter, limit);

                cx += delta.x;
            }

            d += stride;
            cy += delta.y;
            cx = begin.x;
        }
    }


    static inline Vec2D<fmbt> mbt_screen_dims(f32 zoom)
    {
        auto scale = 1.0f / zoom;

        return {
            MBT_WIDTH * scale,
            MBT_HEIGHT * scale
        };
    }
}


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
        u8 format_option = 0;

        f32 zoom_rate = 1.0f;
        f32 zoom = 1.0f;

        u32 iter_limit = MAX_ITERATIONS_START;

        Vec2D<fmbt> mbt_scale;
        Vec2D<fmbt> mbt_pos;
        Vec2D<fmbt> mbt_delta;

        Rect2Du32 copy_src;
        Rect2Du32 copy_dst;

        Rect2Du32 proc_dst[2];
        u8 n_proc = 1;

        bool render_new = false;
        bool enabled = true;
    };


    static void reset_state_data(StateData& data)
    {
        auto w = data.screen_dims.x;
        auto h = data.screen_dims.y;

        data.dt_frame = 1.0 / 60;

        data.format_option = 0;

        data.zoom_rate = ZOOM_RATE_LOWER_LIMIT;
        data.zoom = 1.0f;

        data.iter_limit = MAX_ITERATIONS_START;

        data.mbt_scale = mbt_screen_dims(data.zoom);

        data.mbt_pos = { MBT_MIN_X, MBT_MIN_Y };

        data.mbt_delta = {
            data.mbt_scale.x / w,
            data.mbt_scale.y / h
        };

        auto full_rect = img::make_rect(w, h);

        data.copy_src = full_rect;
        data.copy_dst = full_rect;

        data.proc_dst[0] = full_rect;
        data.proc_dst[1] = full_rect;
        data.n_proc = 1;

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
    static void update_color_format(i8 idelta, StateData& data)
    {
        i32 min = 0;
        i32 max = 6;

        i32 option = (i32)data.format_option + (i32)idelta;

        option = (option > max) ? min : ((option < min) ? max : option);

        data.format_option = (u8)option;
        data.format = colors::make_color_format(data.format_option);        
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

        auto w = data.screen_dims.x;
        auto h = data.screen_dims.y;

        data.mbt_delta = {
            data.mbt_scale.x / w,
            data.mbt_scale.y / h
        };
    }


    static void update_mbt_pos(Vec2D<i8> ishift, StateData& data)
    {
        auto dx = ishift.x * data.mbt_delta.x;
        auto dy = ishift.y * data.mbt_delta.y;

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
}


    static void update_state(InputCommand const& cmd, StateData& data)
    {
        namespace ns = ns_update_state;

        data.in_cmd = cmd;

        ns::update_color_format(cmd.cycle_color, data);
        ns::update_zoom_rate(cmd.zoom_rate, data);

        if (!cmd.direction)
        {
            ns::update_zoom(cmd.zoom, data);
        }

        ns::update_mbt_pos(cmd.shift, data);
        ns::update_iter_limit(cmd.resolution, data);

        static_assert(sizeof(cmd) == sizeof(cmd.any));

        data.render_new = cmd.any;
    }


    static void update_color_ids(StateData& data)
    {
        if (!data.enabled && !data.render_new)
        {
            return;
        }

        // temp
        auto rect = img::make_rect(data.screen_dims.x, data.screen_dims.y);
        mbt_proc(data.color_ids, rect, data.mbt_pos, data.mbt_delta, data.iter_limit);
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

        data.frame_sw.start();

        return true;
    }


    void update(AppState& state, input::Input const& input)
    {       
        auto& data = get_data(state);

        data.dt_frame = data.frame_sw.get_time_sec();
        data.frame_sw.start();

        auto cmd = map_input(input);
        update_state(cmd, data);
        update_color_ids(data);

        render(data.color_ids, state.screen, data.format);

        data.color_ids.swap();
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