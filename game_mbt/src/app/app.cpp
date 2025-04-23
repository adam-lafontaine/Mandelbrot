#include "app.hpp"
#include "../../../libs/alloc_type/alloc_type.hpp"
#include "../../../libs/util/numeric.hpp"

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
    constexpr f32 ZOOM_SPEED_LOWER_LIMIT = 1.0f;
}

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
        constexpr auto C = colors::N_COLORS;
        constexpr u32 N = 8;

        u32 iter_levels[N] = { 50, 100, 200, 300, 400, 500, 600, 800 };

        auto id = ColorId::make_default();

        if (iter >= iter_limit)
        {
            return id;
        }

        u32 min = 0;
	    u32 max = 0;
        u32 n_colors = 8;

        for (u32 i = 0; i < N; i++)
        {
            n_colors *= 2;
            min = max;
            max = i;

            if (iter < max)
            {
                id.value = (iter - min) % n_colors * (C / n_colors);
                return id;
            }
        }

        min = max;
        id.value = (iter - min) % C;

        return id;
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
}


/* map input */

namespace game_mbt
{
    class InputCommand
    {
    public:
        Vec2D<i8> pan;
        i8 zoom = 0; 
        i8 zoom_speed = 0;
        i8 resolution = 0;
        i8 cycle_color = 0;
    };


    static inline b8 is_up(Vec2Df32 vec)
    {
        return vec.y < 0.5f;
    }


    static inline b8 is_down(Vec2Df32 vec)
    {
        return vec.y > 0.5f;
    }


    static inline b8 is_left(Vec2Df32 vec)
    {
        return vec.x < 0.5f;
    }


    static inline b8 is_right(Vec2Df32 vec)
    {
        return vec.x > 0.5f;
    }


    static Vec2D<i8> map_pan(Input const& input)
    {
        auto right = 
            input.keyboard.kbd_D.is_down ||
            input.keyboard.npd_6.is_down ||
            is_right(input.controller.stick_left.vec);
        
        auto left = 
            input.keyboard.kbd_A.is_down ||
            input.keyboard.npd_4.is_down ||
            is_left(input.controller.stick_left.vec);

        auto up = 
            input.keyboard.kbd_W.is_down ||
            input.keyboard.npd_8.is_down ||
            is_up(input.controller.stick_left.vec);

        auto down = 
            input.keyboard.kbd_S.is_down ||
            input.keyboard.npd_2.is_down ||
            is_down(input.controller.stick_left.vec);

        return {
            (i8)((int)right - (int)left),
            (i8)((int)down - (int)up)
        };
    }


    static i8 map_zoom(Input const& input)
    {
        auto in = 
            input.keyboard.npd_plus.is_down ||
            input.controller.stick_right.vec.y < 0.5f;

        auto out = 
            input.keyboard.npd_minus.is_down ||
            input.controller.stick_right.vec.y > 0.5f;

        return (i8)((int)in - (int)out);
    }


    static i8 map_zoom_speed(Input const& input)
    {
        auto fast = 
            input.keyboard.npd_mult.is_down ||
            input.controller.trigger_right > 0.5f;

        auto slow = 
            input.keyboard.npd_div.is_down ||
            input.controller.trigger_left > 0.5f;

        return (i8)((int)fast - (int)slow);
    }


    static i8 map_resolution(Input const& input)
    {
        auto more = 
            input.keyboard.kbd_up.is_down ||
            input.controller.btn_dpad_up.is_down;

        auto less = 
            input.keyboard.kbd_down.is_down ||
            input.controller.btn_dpad_down.is_down;

        return (i8)((int)more - (int)less);
    }


    static i8 map_cycle_color(Input const& input)
    {
        auto right =
            input.keyboard.kbd_right.is_down ||
            input.controller.btn_dpad_right.is_down;
        
        auto left =
            input.keyboard.kbd_left.is_down ||
            input.controller.btn_dpad_left.is_down;

        return (i8)((int)right - (int)left);
    }


    static InputCommand map_input(Input const& input)
    {
        InputCommand cmd{};

        cmd.pan = map_pan(input);
        cmd.zoom = map_zoom(input);
        cmd.zoom_speed = map_zoom_speed(input);
        cmd.cycle_color = map_cycle_color(input);

        return cmd;
    }
}


/* state data */

namespace game_mbt
{
    class StateData
    {
    public:

        ColorIdMatrix color_ids;

        ColorFormat format;

        Rect2Du32 copy_src;
        Rect2Du32 copy_dst;

        Rect2Du32 proc_dst[2];
        u8 n_proc;

        Vec2D<fmbt> mbt_begin;
        Vec2D<fmbt> mbt_delta;

        
    };


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

        data = {};        

        return true;
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

        return true;
    }


    void update(AppState& state, input::Input const& input)
    {
        auto& controller = input.controllers[0];

        auto color = img::to_pixel(0);

        if (controller.btn_a.is_down)
        {
            color = img::to_pixel(0, 200, 0);
        }
        else if (controller.btn_b.is_down)
        {
            color = img::to_pixel(200, 0, 0);
        }
        else if (controller.btn_x.is_down)
        {
            color = img::to_pixel(0, 0, 200);
        }
        else if (controller.btn_y.is_down)
        {
            color = img::to_pixel(200, 200, 0);
        }

        img::fill(state.screen, color);

        auto cmd = map_input(input);

    }


    void reset(AppState& state)
    {

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