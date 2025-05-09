#include "game_state.hpp"
#include "../../../libs/imgui/imgui.h"
#include "../../../libs/stb_libs/qsprintf.hpp"
#include "../../../game_mbt/src/app/app.cpp"

#include <thread>

namespace game_state
{    
    namespace game = game_mbt;


    game::AppState mbt_state;
    bool game_running = false;


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
        return game_running = game::set_screen_memory(mbt_state, screen);
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


    static void show_vec(cstr label, Vec2D<i16> vec)
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


    static void show_rect(cstr label, Rect2Du32 r)
    {
        ImGui::Text("%s: {%u, %u, %u, %u}", label, r.x_begin, r.y_begin, r.x_end, r.y_end);
    }


    static void thread_sleep_ms(int ms)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }


    static ImVec4 to_im_color(img::Pixel p)
    {
        return ImVec4(p.red / 255.0f, p.green / 255.0f, p.blue / 255.0f, p.alpha / 255.0f );
    }


    static void show_input()
    {
        auto& data = game::get_data(mbt_state);
        auto cmd = data.in_cmd;

        //ImGui::SeparatorText("INPUT");
        show_vec(  "shift        ", cmd.shift);
        ImGui::Text("zoom        : %d", (int)cmd.zoom);
        ImGui::Text("zoom_rate   : %d", (int)cmd.zoom_rate);
        ImGui::Text("resolution  : %d", (int)cmd.resolution);
        ImGui::Text("change_color: %d", (int)cmd.change_color);
        ImGui::Text("any         : %u", cmd.any);
    }


    static void show_properties()
    {
        auto& data = game::get_data(mbt_state);

        show_vec(     "screen     ", data.screen_dims);
        show_time_sec("dt_frame   ", data.dt_frame);
        show_vec(     "pixel_shift", data.pixel_shift);
        ImGui::Text("");
        ImGui::Text("zoom rate : %f", data.zoom_rate);
        ImGui::Text("zoom      : %f", data.zoom);
        ImGui::Text("iter_limit: %u", data.iter_limit);
        ImGui::Text("");
        show_vec("scale   ", data.mbt_scale);
        show_vec("position", data.mbt_pos);
        show_vec("delta   ", data.mbt_delta);

        ImGui::Text("");
        ImGui::Text("n_copy    : %u", data.n_copy);
        show_rect(   "copy_src", data.copy_src);
        show_rect(   "copy_dst", data.copy_dst);
        ImGui::Text("");
        ImGui::Text("n_proc     : %u", data.n_proc);
        show_rect(  "proc_dst[0]", data.proc_dst[0]);
        show_rect(  "proc_dst[1]", data.proc_dst[1]);
    }


    static void show_colors()
    {
        ImGui::SeparatorText("Color Table");

        auto& data = game::get_data(mbt_state);
        
        int id_min = 0;
        int id_max = game::ColorId::max;
        static int id_val = game::ColorId::max;
        static ImVec4 color;

        auto id = game::ColorId::make_default();

        ImGui::SliderInt("ColorId", &id_val, id_min, id_max);

        id = game::ColorId::make((u32)id_val);
        
        color = to_im_color(game::color_at(id, data.format));

        ImGui::Text(" Id: %u", id.value);
        ImGui::ColorEdit3("Color##1", (float*)&color);

    }
    
}


namespace game_state
{
    class PlotProps
    {
    private:
        static constexpr int N = 256;

    public:
        cstr label = 0;

        cstr units = 0;

        f32 plot_data[N] = { 0 };
        int data_count = N;
        int data_offset = 0;
        int data_stride = sizeof(f32);
        f32 plot_min = 0.0f;
        f32 plot_max = 0.0f;
        ImVec2 plot_size = ImVec2(0, 100.0f);

        bool enabled = false;
        bool started = false;
        bool minmax = false;

        int count = 0;
        f32 total = 0.0f;
        f32 avg = 0.0f;

        void add_data(f32 val)
        {            
            if (count == N)
            {
                total -= plot_data[data_offset];
            }

            total += val;

            plot_data[data_offset] = val;
            data_offset = (data_offset + 1) & (N - 1);

            count = num::min(count + 1, N);

            avg = total / count;            

            if (!minmax)
            {                
                plot_min = plot_max = val;

                minmax = true;
            }

            plot_min = num::min(val, plot_min);
            plot_max = num::max(val, plot_max);
        }


        void reset()
        {
            enabled = false;
            started = false;
            minmax = false;
            count = 0;
            total = 0.0f;
            avg = 0.0f;
        }

    };


    static void show_plot(PlotProps& props, cstr label)
    {
        char label_text[32] = { 0 };
        stb::qsnprintf(label_text, 32, "avg: %f##%s", props.avg, label);

        char overlay[32] = { 0 };

        ImGui::Text("min: %f %s, max: %f %s", props.plot_min, props.units, props.plot_max, props.units);
        ImGui::PlotLines(label_text, 
            props.plot_data, 
            props.data_count, 
            props.data_offset, 
            overlay,
            props.plot_min, props.plot_max, 
            props.plot_size, 
            props.data_stride);
    }


    static void start_proc_copy(PlotProps& props)
    {
        props.units = "ms";

        auto copy_loop = [&]()
        {
            auto& data = game::get_data(mbt_state);
            auto r = img::make_rect(data.screen_dims.x, data.screen_dims.y);

            Stopwatch sw;
            while (game_running && props.enabled)
            {
                sw.start();
                game::proc_copy(data.mbt_mat, r, r);
                props.add_data((f32)sw.get_time_milli());

                thread_sleep_ms(10);
            }
        };

        std::thread th(copy_loop);
        th.detach();        
    }


    static void start_proc_mbt(PlotProps& props)
    {
        props.units = "ms";

        auto copy_loop = [&]()
        {
            auto& data = game::get_data(mbt_state);

            Stopwatch sw;
            while (game_running && props.enabled)
            {
                sw.start();
                game::proc_mbt(data.mbt_mat, data.mbt_pos, data.mbt_delta, data.iter_limit);
                props.add_data((f32)sw.get_time_milli());

                thread_sleep_ms(10);
            }
        };

        std::thread th(copy_loop);
        th.detach();   
    }


    static void start_proc_render(PlotProps& props)
    {
        props.units = "ms";

        auto copy_loop = [&]()
        {
            auto& data = game::get_data(mbt_state);
            auto r = img::make_rect(data.screen_dims.x, data.screen_dims.y);

            Stopwatch sw;
            while (game_running && props.enabled)
            {
                sw.start();
                game::proc_render(data.mbt_mat, mbt_state.screen, data.format);
                props.add_data((f32)sw.get_time_milli());

                thread_sleep_ms(10);
            }
        };

        std::thread th(copy_loop);
        th.detach();
    }
}


namespace game_state
{
    void show_game_state()
    {
        if (!ImGui::CollapsingHeader("Game State"))
        {
            return; 
        }

        if (!mbt_state.data_)
        {
            return;
        }

        if (ImGui::TreeNode("Input"))
        {
            show_input();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Properties"))
        {
            show_properties();
            ImGui::TreePop();
        }

        if (ImGui::TreeNode("Colors"))
        {
            show_colors();
            ImGui::TreePop();
        }
    }


    void show_profiling()
    {   
        static bool enable_profile = false;
        static PlotProps copy_props = {};
        static PlotProps mbt_props = {};
        static PlotProps render_props = {};

        if (!ImGui::CollapsingHeader("Profiling"))
        {
            return; 
        }

        if (!mbt_state.data_)
        {
            return;
        }

        ImGui::Checkbox("Enable", &enable_profile);        

        state.hard_pause = enable_profile;

        ImGui::Separator();

        if (!enable_profile)
        {
            copy_props.enabled = false;
            mbt_props.enabled = false;
            render_props.enabled = false;

            ImGui::BeginDisabled();
        }
        
        ImGui::Checkbox("proc_copy", &copy_props.enabled);
        if (copy_props.enabled)
        {
            if (!copy_props.started)
            {
                start_proc_copy(copy_props);
                copy_props.started = true;
            }
            show_plot(copy_props, "CopyPlot");
        }
        else
        {
            copy_props.reset();
        }

        ImGui::Checkbox("proc_mbt", &mbt_props.enabled);
        
        if (mbt_props.enabled)
        {
            if (!mbt_props.started)
            {
                start_proc_mbt(mbt_props);
                mbt_props.started = true;
            }
            show_plot(mbt_props, "MbtPlot");
        }
        else
        {
            mbt_props.reset();
        }

        
        ImGui::Checkbox("proc_render", &render_props.enabled);
        if (render_props.enabled)
        {
            if (!render_props.started)
            {
                start_proc_render(render_props);
                render_props.started = true;
            }
            show_plot(render_props, "RenderPlot");
        }
        else
        {
            render_props.reset();
        }


        

        if (!enable_profile)
        {
            ImGui::EndDisabled();
        }
    }
}