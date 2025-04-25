#pragma once

#include "../game_state/game_state.hpp"

namespace gs = game_state;


/* diagnostics window */

namespace ui
{
    /* memory */

    namespace diagnostics
    {
        static void bytes_text(auto bytes)
        {
            auto suffix = "B";
            auto num = (f32)bytes;

            if (num >= 1'000'000'000.0f)
            {
                num /= 1'000'000'000.0f;
                suffix = "G";
            }
            else if (num >= 1'000'000.0f)
            {
                num /= 1'000'000.0f;
                suffix = "M";            
            }
            else if (num >= 1'000.0f)
            {
                num /= 1'000.0f;
                suffix = "K";
            }

            char text[32] = { 0 };
            stb::qsnprintf(text, 32, "%5.1f %s", num, suffix);

            ImGui::Text("%s", text);
        }


        static void current_alloc_table()
        {
            constexpr int col_type = 0;
            constexpr int col_bytes = 1;
            constexpr int col_alloc = 2;
            constexpr int n_columns = 3;

            int table_flags = ImGuiTableFlags_BordersInnerV;
            auto table_dims = ImVec2(0.0f, 0.0f);

            ImU32 cell_bg_color = ImGui::GetColorU32(ig::gray(0.1f));            

            int expand_action = -1;

            u32 total_alloc = 0;
            u32 total_max_alloc = 0;
            u64 total_bytes = 0;

            auto const setup_columns = []()
            {
                ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 200.0f);
                ImGui::TableSetupColumn("Bytes", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Allocations", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableHeadersRow();
            };

            auto const table_row = [&](u32 size)
            {
                auto status = mem::query_status(size);

                total_alloc += status.n_allocations;
                total_max_alloc += status.max_allocations;
                total_bytes += status.bytes_allocated;

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(col_alloc);
                ImGui::Text("%u/%u", status.n_allocations, status.max_allocations);

                ImGui::TableSetColumnIndex(col_bytes);
                bytes_text(status.bytes_allocated);

                ImGui::TableSetColumnIndex(col_type);
                if (!status.n_allocations)
                {
                    ImGui::Text("   %s", status.type_name);
                }
                else 
                {
                    if (expand_action != -1)
                    {
                        ImGui::SetNextItemOpen(expand_action != 0);
                    }
                    if (ImGui::TreeNode(status.type_name))
                    {
                        for (u32 i = 0; i < status.n_allocations; i++)
                        {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(col_type);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            ImGui::Text(" %s", status.slot_tags[i]);

                            ImGui::TableSetColumnIndex(col_bytes);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            bytes_text(status.slot_sizes[i]);

                            ImGui::TableSetColumnIndex(col_alloc);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                        }

                        ImGui::TreePop();
                    }
                }
            };

            auto const totals_row = [&]()
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(col_type);
                ImGui::Text("   Total");
                
                ImGui::TableSetColumnIndex(col_alloc);
                ImGui::Text("%u/%u", total_alloc, total_max_alloc);

                ImGui::TableSetColumnIndex(col_bytes);
                bytes_text(total_bytes);
            };

            ImGui::Separator();
            ImGui::Text("Current Allocations");
            ImGui::SameLine();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetStyle().IndentSpacing);
            if (ImGui::Button("Expand all##CAT"))
            {
                expand_action = 1;
            }
            ImGui::SameLine();
            if (ImGui::Button("Close all##CAT"))
            {
                expand_action = 0;
            }

            if (!ImGui::BeginTable("MemoryTable", n_columns, table_flags, table_dims)) 
            { 
                return; 
            }

            setup_columns(); 
            
            table_row(1);
            table_row(2);
            table_row(4);
            table_row(8);
            totals_row();

            ImGui::EndTable();
        }


        static void alloc_history_table()
        {
            constexpr int col_type = 0;
            constexpr int col_action = 1;
            constexpr int col_size = 2;
            constexpr int col_bytes = 3;
            constexpr int col_allocs = 4;
            constexpr int n_columns = 5;

            auto const setup_columns = []()
            {
                ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthStretch, 200.0f);
                ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("size", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Bytes", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableSetupColumn("Allocations", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                ImGui::TableHeadersRow();
            };

            ImU32 cell_bg_color = ImGui::GetColorU32(ig::gray(0.1f));

            int table_flags = ImGuiTableFlags_BordersInnerV;
            auto table_dims = ImVec2(0.0f, 0.0f);

            int expand_action = -1;

            auto const table_row = [&](u32 size)
            {
                auto hist = mem::query_history(size);

                ImGui::TableNextRow();

                ImGui::TableSetColumnIndex(col_type);
                if (!hist.n_items)
                {
                    ImGui::Text("   %s", hist.type_name);
                }
                else
                {
                    if (expand_action != -1)
                    {                
                        ImGui::SetNextItemOpen(expand_action != 0);
                    }
                    if (ImGui::TreeNode(hist.type_name))            
                    {
                        for (u32 i = 0; i < hist.n_items; i++)
                        {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(col_type);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            ImGui::Text(" %s", hist.tags[i]);

                            ImGui::TableSetColumnIndex(col_action);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            ImGui::Text("%s", hist.actions[i]);

                            ImGui::TableSetColumnIndex(col_size);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            bytes_text(hist.sizes[i]);

                            ImGui::TableSetColumnIndex(col_bytes);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            bytes_text(hist.n_bytes[i]);

                            ImGui::TableSetColumnIndex(col_allocs);
                            ImGui::TableSetBgColor(ImGuiTableBgTarget_CellBg, cell_bg_color);
                            ImGui::Text("%u/%u", hist.n_allocs[i], hist.max_allocations);
                        }

                        ImGui::TreePop();
                    }            
                }
            };

            ImGui::Separator();
            ImGui::Separator();
            ImGui::Text("Allocation History");
            ImGui::SameLine();
            ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetStyle().IndentSpacing);
            if (ImGui::Button("Expand all##AHT"))
            {
                expand_action = 1;
            }
            ImGui::SameLine();
            if (ImGui::Button("Close all##AHT"))
            {
                expand_action = 0;
            }

            if (!ImGui::BeginTable("AllocationHistoryTable", n_columns, table_flags, table_dims)) 
            { 
                return; 
            }

            setup_columns();
            
            table_row(1);
            table_row(2);
            table_row(4);
            table_row(8);

            ImGui::EndTable();
        }


        static void show_memory()
        {
            if (!ImGui::CollapsingHeader("Memory"))
            {
                return; 
            }
            
            current_alloc_table();
            alloc_history_table();
        }

    }


    void diagnostics_window()
    {
        ImGui::Begin("Diagnostics");

        diagnostics::show_memory();
        gs::show_game_state();
        gs::show_profiling();

        ImGui::End();
    }
}
