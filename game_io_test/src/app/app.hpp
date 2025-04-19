#pragma once

#include "../../../libs/input/input.hpp"
#include "../../../libs/image/image.hpp"


namespace game_io_test
{
    namespace img = image;
    namespace mb = memory_buffer;

    constexpr auto APP_TITLE = "IO Test";
    constexpr auto VERSION = "1.0.0";


    class StateData;


    class AppState
    {
    public:
        img::ImageView screen;

        StateData* data = nullptr;        
    };


    class AppResult
    {
    public:
        bool success = false;

        Vec2Du32 screen_dimensions;
    };


    AppResult init(AppState& state);

    bool set_screen_memory(AppState& state, img::ImageView screen);

    void update(AppState& state, input::Input const& input);

    void reset(AppState& state);

    void close(AppState& state);
}