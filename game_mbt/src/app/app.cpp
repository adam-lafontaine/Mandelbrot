#include "app.hpp"


namespace game_mbt
{
    namespace img = image;

    using p32 = img::Pixel;
}


namespace game_mbt
{
    AppResult init(AppState& state)
    {
        AppResult result;

        result.app_dimensions = { 401, 400 };
        result.success = true;

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

    }
}


#include "app_libs.cpp"