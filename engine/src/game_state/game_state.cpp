#include "game_state.hpp"

#include "../../../game_mbt/src/app/app.cpp"

namespace game_state
{    
    namespace game = game_mbt;


    game::AppState mbt_state;


    bool init(Vec2Du32& screen_dimensions)
    {
        auto result = game::init(mbt_state);
        if (!result.success)
        {
            return false;
        }

        screen_dimensions = result.app_dimensions;

        return true;
    }


    bool set_screen_memory(image::ImageView screen)
    {
        return game::set_screen_memory(mbt_state, screen);
    }


    void update(input::Input const& input)
    {
        game::update(mbt_state, input);
    }


    void reset()
    {
        game::reset(mbt_state);
    }


    void close()
    {
        game::close(mbt_state);
    }
}


namespace game_state
{
  
    void show_game_state()
    {
        
    }
}