#pragma once

#include "../../../libs/input/input.hpp"
#include "../../../libs/image/image.hpp"


namespace game_state
{
    bool init(Vec2Du32& screen_dimensions);

    bool set_screen_memory(image::ImageView screen);

    void update(input::Input const& input);

    void reset();

    void close();
}


namespace game_state
{
  
    void show_game_state();
}