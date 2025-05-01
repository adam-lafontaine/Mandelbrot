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
                b8 change_color;

                u8 available[2];
            };

            u64 any = 0;
        };
        
    };


namespace ns_map_input
{
    static inline b8 is_up(Vec2Df32 vec)
    {
        return vec.y < -0.5f;
    }


    static inline b8 is_down(Vec2Df32 vec)
    {
        return vec.y > 0.5f;
    }


    static inline b8 is_left(Vec2Df32 vec)
    {
        return vec.x < -0.5f;
    }


    static inline b8 is_right(Vec2Df32 vec)
    {
        return vec.x > 0.5f;
    }


    static Vec2D<i8> map_shift(Input const& input)
    {
        auto right = 
            input.keyboard.kbd_D.is_down ||
            input.keyboard.kbd_right.is_down ||
            input.keyboard.npd_6.is_down ||
            input.controller.btn_dpad_right.is_down ||
            is_right(input.controller.stick_left.vec);
        
        auto left = 
            input.keyboard.kbd_A.is_down ||
            input.keyboard.kbd_left.is_down ||
            input.keyboard.npd_4.is_down ||
            input.controller.btn_dpad_left.is_down ||
            is_left(input.controller.stick_left.vec);

        auto up = 
            input.keyboard.kbd_W.is_down ||
            input.keyboard.kbd_up.is_down ||
            input.keyboard.npd_8.is_down ||
            input.controller.btn_dpad_up.is_down ||
            is_up(input.controller.stick_left.vec);

        auto down = 
            input.keyboard.kbd_S.is_down ||
            input.keyboard.kbd_down.is_down ||
            input.keyboard.npd_2.is_down ||
            input.controller.btn_dpad_down.is_down ||
            is_down(input.controller.stick_left.vec);

        return {
            (i8)((int)right - (int)left),
            (i8)((int)down - (int)up)
        };
    }


    static i8 map_zoom(Input const& input)
    {
        auto in = 
            input.keyboard.kbd_P.is_down ||
            input.keyboard.npd_plus.is_down ||
            input.controller.btn_a.is_down ||
            is_up(input.controller.stick_right.vec);

        auto out = 
            input.keyboard.kbd_O.is_down ||
            input.keyboard.npd_minus.is_down ||
            input.controller.btn_b.is_down ||
            is_down(input.controller.stick_right.vec);

        return (i8)((int)in - (int)out);
    }


    static i8 map_zoom_rate(Input const& input)
    {
        auto fast = 0;

        auto slow = 0;

        // disabled TODO

        return (i8)((int)fast - (int)slow);
    }


    static i8 map_resolution(Input const& input)
    {
        auto more =             
            input.keyboard.kbd_L.is_down ||
            input.keyboard.npd_mult.is_down ||
            input.controller.btn_x.is_down;

        auto less = 
            input.keyboard.kbd_K.is_down ||
            input.keyboard.npd_div.is_down ||
            input.controller.btn_y.is_down;

        return (i8)((int)more - (int)less);
    }


    static b8 map_change_color(Input const& input)
    {
        auto change =
            input.keyboard.kbd_space.pressed ||
            input.controller.btn_shoulder_left.pressed ||
            input.controller.btn_shoulder_right.pressed ||
            input.keyboard.npd_0.pressed;

        return (b8)(change);
    }
}


    static InputCommand map_input(Input const& input)
    {
        namespace ns = ns_map_input;

        InputCommand cmd{};
        cmd.any = 0;

        cmd.shift = ns::map_shift(input);
        cmd.zoom = ns::map_zoom(input);
        cmd.zoom_rate = ns::map_zoom_rate(input);
        cmd.resolution = ns::map_resolution(input);
        cmd.change_color = ns::map_change_color(input);

        return cmd;
    }
}