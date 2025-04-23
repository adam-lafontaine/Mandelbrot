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


    #ifdef SDL2_WASM

	constexpr u32 BUFFER_WIDTH = 640;
	constexpr u32 BUFFER_HEIGHT = BUFFER_WIDTH * 8 / 9;

	#else

	// allocate memory
	constexpr u32 BUFFER_HEIGHT = 800;
	constexpr u32 BUFFER_WIDTH = BUFFER_HEIGHT * 9 / 8;

	#endif

	constexpr u32 PIXELS_PER_SECOND = (u32)(0.2 * BUFFER_HEIGHT);
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
    public:

        MatrixView2D<ColorId> data;

        MemoryBuffer<ColorId> buffer;
    };


    static void destroy_color_ids(ColorIdMatrix& mat)
    {
        mb::destroy_buffer(mat.buffer);
    }


    static bool create_color_ids(ColorIdMatrix& mat, u32 width, u32 height)
    {
        auto n = width * height;

        if (!mb::create_buffer(mat.buffer, n, "color_ids"))
        {
            return false;
        }

        auto span = span::push_span(mat.buffer, n);
        mat.data.matrix_data_ = span.data;
        mat.data.width = width;
        mat.data.height = height;

        return true;
    }




}


/* render */

namespace game_mbt
{
    constexpr ColorTable Color_Table = colors::make_palette<colors::N_COLORS>();


    static p32 color_at(ColorId id, ColorFormat format)
    {
        auto r = Color_Table.channels[format.R][id.value];
        auto g = Color_Table.channels[format.G][id.value];
        auto b = Color_Table.channels[format.B][id.value];

        return img::to_pixel(r, g, b);
    }


    static void render(ColorIdMatrix const& src, ImageView const& dst, ColorFormat format)
    {
        auto s = img::to_span(src.data);
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

        fmbt min_mx;
        fmbt min_my;
        fmbt mx_step;
        fmbt my_step;

        
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