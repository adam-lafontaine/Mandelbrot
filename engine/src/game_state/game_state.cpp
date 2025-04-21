#include "game_state.hpp"


namespace game_state
{
    namespace img = image;


    img::ImageView temp_screen;


    bool init(Vec2Du32& screen_dimensions)
    {
        screen_dimensions.x = 401;
        screen_dimensions.y = 400;

        return true;
    }


    bool set_screen_memory(image::ImageView screen)
    {
        temp_screen = screen;

        return true;
    }


    void update(input::Input const& input)
    {
        auto& controller = input.controllers[0];

        img::Pixel color = img::to_pixel(0);

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

        img::fill(temp_screen, color);

    }


    void reset()
    {

    }


    void close()
    {

    }
}


namespace game_state
{
  
    void show_game_state()
    {
        
    }
}