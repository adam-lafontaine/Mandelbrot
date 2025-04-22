#pragma once

#include "../../../libs/input/input.hpp"
#include "../../../libs/image/image.hpp"


namespace game_mbt
{
    constexpr auto APP_TITLE = "Mandelbrot";
    constexpr auto VERSION = "1.2.0";


    class StateData;


    class AppState
    {
    public:
        image::ImageView screen;

        StateData* data_ = 0;
    };


    class AppResult
    {
    public:
        bool success = false;

        int error_code = 0;

        Vec2Du32 app_dimensions;
    };


    AppResult init(AppState& state);

    AppResult init(AppState& state, Vec2Du32 available_dims);

    bool set_screen_memory(AppState& state, image::ImageView screen);

    void update(AppState& state, input::Input const& input);

    void reset(AppState& state);

    void close(AppState& state);

    cstr decode_error(AppResult const& result);
}