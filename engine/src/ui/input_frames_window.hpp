#pragma once

/* input frames window */

namespace ui
{
    namespace input_fr
    {
        static void frame_list(cstr title, InputFrames& frames)
        {
            ImGui::BeginGroup();

            ImGui::Checkbox(title, &frames.is_on);

            for (u32 i = 0; i < frames.max_frames; i++)
            {
                u8 id = frames.index - (u8)i - 1;
                ImGui::Text("0x%016lX", frames.frames[id]);
            }

            ImGui::EndGroup();
        }
    }


    void input_frames_window(EngineState& state)
    {
        ImGui::Begin("Frames");

        input_fr::frame_list("Keyboard", state.keyboard_frames);
        ImGui::SameLine();
        input_fr::frame_list("Controller", state.controller_frames);
        ImGui::SameLine();
        input_fr::frame_list("Mouse", state.mouse_frames);

        ImGui::End();
    }
}


