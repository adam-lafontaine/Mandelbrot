#include "game_state.hpp"
#include "../../../libs/imgui/imgui.h"
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
    static void show_vec(cstr label, Vec2Du32 vec)
    {
        ImGui::Text("%s: {%u, %u}", label, vec.x, vec.y);
    }


    static void show_vec(cstr label, Vec2D<game::fmbt> vec)
    {
        ImGui::Text("%s: {%f, %f}", label, vec.x, vec.y);
    }


    static void show_vec(cstr label, Vec2D<i8> vec)
    {
        ImGui::Text("%s: {%d, %d}", label, (int)vec.x, (int)vec.y);
    }


    static void show_time_sec(cstr label, f32 sec)
    {
        constexpr f32 NANO =  1.0f / 1'000'000'000;
        constexpr f32 MICRO = 1.0f / 1'000'000;
        constexpr f32 MILLI = 1.0f / 1'000;

        constexpr auto nano = "ns";
        constexpr auto micro = "us";
        constexpr auto milli = "ms";
        constexpr auto s = "sec";

        cstr unit = s;

        if (sec < NANO)
        {
            unit = nano;
            sec *= 1'000'000'000;
        }
        else if (sec < MICRO)
        {
            unit = micro;
            sec *= 1'000'000;
        }
        else if (sec < MILLI)
        {
            unit = milli;
            sec *= 1'000;
        }

        ImGui::Text("%s: %f %s", label, sec, unit);
    }


    void show_input()
    {
        auto& data = game::get_data(mbt_state);
        auto cmd = data.in_cmd;

        ImGui::Text("INPUT");
        show_vec("shift", cmd.shift);
        ImGui::Text("zoom: %d", (int)cmd.zoom);
        ImGui::Text("zoom_rate: %d", (int)cmd.zoom_rate);
        ImGui::Text("resolution: %d", (int)cmd.resolution);
        ImGui::Text("cycle_color: %d", (int)cmd.cycle_color);
        ImGui::Text("any: %u", cmd.any);
    }

  
    void show_game_state()
    {
        if (!ImGui::CollapsingHeader("Game State"))
        {
            return; 
        }

        show_input();

        ImGui::Separator();

        auto& data = game::get_data(mbt_state);

        show_vec(     "screen  ", data.screen_dims);
        show_time_sec("dt_frame", data.dt_frame);
        ImGui::Text("");
        ImGui::Text("zoom rate: %f", data.zoom_rate);
        ImGui::Text("zoom     : %f", data.zoom);
        ImGui::Text("");
        ImGui::Text("format_option: %u", data.format_option);
        ImGui::Text("");
        show_vec("scale   ", data.mbt_scale);
        show_vec("position", data.mbt_pos);
        show_vec("delta   ", data.mbt_delta);

        ImGui::Separator();

        


    }
}