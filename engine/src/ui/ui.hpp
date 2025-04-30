#pragma once

#include "../../../libs/imgui/imgui.h"
#include "../../../libs/util/stopwatch.hpp"
#include "../../../libs/stb_libs/qsprintf.hpp"
#include "../../../libs/alloc_type/alloc_type.hpp"

constexpr auto ENGINE_WINDOW_TITLE = "Game Engine";
constexpr int ENGINE_WINDOW_WIDTH = 1400;
constexpr int ENGIN_WINDOW_HEIGHT = 800;

constexpr f64 NANO = 1'000'000'000;
constexpr f64 MICRO = 1'000'000;

constexpr f64 TARGET_FRAMERATE_HZ = 60.0;
constexpr f64 TARGET_NS_PER_FRAME = NANO / TARGET_FRAMERATE_HZ;
constexpr f64 TARGET_MS_PER_FRAME = TARGET_NS_PER_FRAME / MICRO;


class InputFrames
{
public:
    static constexpr u16 max_frames = 256;

    bool is_on = false;

    u64 frames[max_frames] = { 0 };
    u8 index = 0;

    void add_frame(u64 frame)
    {
        if (is_on) { frames[index++] = frame; }
    }
};


class FrameTimes
{
private:
    f32 total = 0.0f;
    f32 avg = 0.0f;

public:
    static constexpr u16 max_times = 256;

    f32 times[2 * max_times] { 0 };

    u8 begin = 0;
    u16 end = 255;

    void add_time(f32 time)
    {
        times[begin] = times[end] = time;
        ++begin;
        end = begin + 256;

        if (begin % 32)
        {
            total += time;
        }
        else
        {
            avg = total / 32;
            total = 0.0f;
        }
    }


    f32 current_avg()
    {
        return avg;
    }
};


class EngineState
{
public:
    Stopwatch thread_sw;
    f64 thread_nano = TARGET_NS_PER_FRAME;

    Stopwatch game_sw;
    f64 game_nano = TARGET_NS_PER_FRAME;

    ImGuiIO* io = nullptr;
    f32 io_display_scale = 1.5f;

#ifdef SHOW_IMGUI_DEMO

    bool show_demo_window = true;
    bool show_another_window = false;
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

#endif

    f32 game_window_scale = 1.0f;

    InputFrames keyboard_frames{};
    InputFrames controller_frames{};
    InputFrames mouse_frames{};

    FrameTimes frame_times{};

    b8 hard_pause = 0;

    b8 cmd_toggle_pause = 0;
    b8 cmd_reset_game = 0;

    b8 cmd_save_screenshot = 0;
};


namespace ig
{
    constexpr auto WHITE = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);


    constexpr auto gray(f32 value)
    {
        return ImVec4(value, value, value, 1.0f);
    }
}


/* game and input windows */

namespace ui
{
    static void game_window(cstr title, ImTextureID image_texture, u32 width, u32 height, EngineState const& state)
    {        
        ImGui::Begin(title, 0, ImGuiWindowFlags_HorizontalScrollbar);

        auto min = [](f32 a, f32 b) { return a < b ? a : b; };

        if (state.hard_pause)
        {
            ImGui::Text("size = %u x %u | PAUSED", width, height);
        }
        else
        {
            auto game_ms = min((f32)(state.game_nano / MICRO), 999.999f);
            auto thread_ms = min((f32)(state.thread_nano / MICRO), 999.999f);
            ImGui::Text("size = %u x %u | scale = %.1f | frame = %7.3f/%7.3f ms", width, height, state.game_window_scale, game_ms, thread_ms);
        }

        auto w = width * state.game_window_scale;
        auto h = height * state.game_window_scale;
        
        ImGui::Image(image_texture, ImVec2(w, h));
        ImGui::End();
    }


    static void io_test_window(ImTextureID image_texture, u32 width, u32 height, EngineState const& state)
    {
        ImGui::Begin("Input");

        auto w = width * state.io_display_scale;
        auto h = height * state.io_display_scale;

        ImGui::Image(image_texture, ImVec2(w, h));

        ImGui::End();
    }


    static void game_control_window(EngineState& state)
    {
        ImGui::Begin("Game");

        if (ImGui::Button("Pause##PauseButton"))
        {
            state.cmd_toggle_pause = true;
        }

        ImGui::SameLine();

        if (ImGui::Button("Reset##ResetButton"))
        {
            state.cmd_reset_game = true;
        }

        ImGui::SameLine();

        constexpr auto slider_scale_min = 0.5f;
        static f32 slider_scale_max = 10.0f;

        ImGui::SliderFloat("##ScaleSlider", 
            &state.game_window_scale, 
            slider_scale_min, 
            slider_scale_max, 
            "scale = %3.1f");
        
        ImGui::SameLine();

        ImGui::InputFloat("Max##MaxSliderScale", &slider_scale_max, 0.1f, 0.1f, "%.1f");

        auto const plot_data = state.frame_times.times;
        auto const data_count = (int)state.frame_times.max_times;
        auto const data_offset = (int)state.frame_times.begin;
        constexpr auto plot_min = 0.0f;
        constexpr auto plot_max = 1.0f;
        constexpr auto plot_size = ImVec2(0, 120.0f);
        constexpr auto data_stride = sizeof(f32);

        char overlay[32] = { 0 };
        stb::qsnprintf(overlay, 32, "frame ratio: %.4f", state.frame_times.current_avg());

        // https://github.com/epezent/implot/tree/master
        
        ImGui::PlotLines("##FrameTimesPlot", 
            plot_data, 
            data_count, 
            data_offset, 
            overlay,
            plot_min, plot_max, 
            plot_size, 
            data_stride);

        ImGui::End();
    }


    
}


#include "diagnostics_window.hpp"

#include "input_frames_window.hpp"