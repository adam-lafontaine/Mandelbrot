namespace game_mbt
{
    class InputCommand
    {
    public:
        union
        {
            struct
            {
                union
                {
                    Vec2D<i8> shift;

                    u16 direction;
                };

                i8 zoom; 
                i8 zoom_rate;
                i8 resolution;
                i8 cycle_color;

                u8 available[2];
            };

            u64 any = 0;
        };
        
    };


namespace ns_map_input
{
    static inline b8 is_up(Vec2Df32 vec)
    {
        return vec.y < 0.5f;
    }


    static inline b8 is_down(Vec2Df32 vec)
    {
        return vec.y > 0.5f;
    }


    static inline b8 is_left(Vec2Df32 vec)
    {
        return vec.x < 0.5f;
    }


    static inline b8 is_right(Vec2Df32 vec)
    {
        return vec.x > 0.5f;
    }


    static Vec2D<i8> map_shift(Input const& input)
    {
        auto right = 
            input.keyboard.kbd_D.is_down ||
            input.keyboard.npd_6.is_down ||
            is_right(input.controller.stick_left.vec);
        
        auto left = 
            input.keyboard.kbd_A.is_down ||
            input.keyboard.npd_4.is_down ||
            is_left(input.controller.stick_left.vec);

        auto up = 
            input.keyboard.kbd_W.is_down ||
            input.keyboard.npd_8.is_down ||
            is_up(input.controller.stick_left.vec);

        auto down = 
            input.keyboard.kbd_S.is_down ||
            input.keyboard.npd_2.is_down ||
            is_down(input.controller.stick_left.vec);

        return {
            (i8)((int)right - (int)left),
            (i8)((int)down - (int)up)
        };
    }


    static i8 map_zoom(Input const& input)
    {
        auto in = 
            input.keyboard.npd_plus.is_down ||
            input.controller.stick_right.vec.y < 0.5f;

        auto out = 
            input.keyboard.npd_minus.is_down ||
            input.controller.stick_right.vec.y > 0.5f;

        return (i8)((int)in - (int)out);
    }


    static i8 map_zoom_rate(Input const& input)
    {
        auto fast = 
            input.keyboard.npd_mult.is_down ||
            input.controller.trigger_right > 0.5f;

        auto slow = 
            input.keyboard.npd_div.is_down ||
            input.controller.trigger_left > 0.5f;

        return (i8)((int)fast - (int)slow);
    }


    static i8 map_resolution(Input const& input)
    {
        auto more = 
            input.keyboard.kbd_up.is_down ||
            input.controller.btn_dpad_up.is_down;

        auto less = 
            input.keyboard.kbd_down.is_down ||
            input.controller.btn_dpad_down.is_down;

        return (i8)((int)more - (int)less);
    }


    static i8 map_cycle_color(Input const& input)
    {
        auto right =
            input.keyboard.kbd_right.is_down ||
            input.controller.btn_dpad_right.is_down;
        
        auto left =
            input.keyboard.kbd_left.is_down ||
            input.controller.btn_dpad_left.is_down;

        return (i8)((int)right - (int)left);
    }
}


    static InputCommand map_input(Input const& input)
    {
        namespace ns = ns_map_input;

        InputCommand cmd{};

        cmd.shift = ns::map_shift(input);
        cmd.zoom = ns::map_zoom(input);
        cmd.zoom_rate = ns::map_zoom_rate(input);
        cmd.cycle_color = ns::map_cycle_color(input);

        return cmd;
    }
}