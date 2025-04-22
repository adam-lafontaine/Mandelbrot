#include "app.hpp"
#include "../../../libs/alloc_type/alloc_type.hpp"

namespace game_mbt
{
    namespace img = image;

    using p32 = img::Pixel;


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


/* state data */

namespace game_mbt
{
    class StateData
    {
    public:

        
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