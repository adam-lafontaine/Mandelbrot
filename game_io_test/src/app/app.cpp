#include "app.hpp"
#include "../../../libs/util/numeric.hpp"
#include "../../../libs/ascii_image/ascii_image.hpp"
#include "../../../libs/stb_libs/qsprintf.hpp"

#include "assets.cpp"


/* definitions */

namespace game_io_test
{
    namespace num = numeric;

    using Input = input::Input;

    template <typename T>
    using ControllerDef = assets::controller::ControllerDef<T>;

    template <typename T>
    using ControllerStickDef = assets::controller::ControllerStickDef<T>;

    template <typename T>
    using KeyboardDef = assets::keyboard::KeyboardDef<T>;

    template <typename T>
    using MouseDef = assets::mouse::MouseDef<T>;
}


/* mask view */

namespace game_io_test
{
    class MaskViewMap
    {
    public:
        img::GraySubView mask;
        img::SubView out;
    };

    using ControllerMaskViewMap = ControllerDef<MaskViewMap>;
    using KeyboardMaskViewMap = KeyboardDef<MaskViewMap>;
    using MouseMaskViewMap = MouseDef<MaskViewMap>;
    using ControllerStickMaskViewMap = ControllerStickDef<MaskViewMap>;


    class MaskViewMapList
    {
    public:
        
        MaskViewMap controller1;
        MaskViewMap controller2;
        MaskViewMap keyboard;
        MaskViewMap mouse;

        ControllerMaskViewMap controller1_inputs;
        ControllerMaskViewMap controller2_inputs;
        KeyboardMaskViewMap keyboard_inputs;
        MouseMaskViewMap mouse_inputs;

        ControllerStickMaskViewMap controller1_thumbsticks;
        ControllerStickMaskViewMap controller2_thumbsticks;
    };    


    static Vec2Du32 app_screen_dimensions(assets::DrawMaskData const& masks)
    {
        /*
        | ctlr ctlr |
        | kbd   mse |
        */

        auto& c = masks.controller_view;
        auto& k = masks.keyboard_view;
        auto& m = masks.mouse_view;

        auto w = num::max(c.width * 2, k.width + m.width);
        auto h = num::max(c.height + k.height, c.height + m.height);

        return { w, h };
    }


    template <class M, class MV>
    static void set_map_masks(M const& m, MV& mv)
    {
        static_assert(M::count == MV::count);
        for (u32 i = 0; i < mv.count; i++)
        {
            mv.list[i].mask = m.list[i];
        }
    }


    template <class V, class R, class MV>
    static void set_map_out(V const& view, R const& reg, MV& mv)
    {
        static_assert(R::count == MV::count);
        for (u32 i = 0; i < mv.count; i++)
        {
            mv.list[i].out = img::sub_view(view, reg.list[i]);
        }
    }


    static void set_map_masks(img::GrayView const& view, ControllerStickMaskViewMap& mv)
    {
        auto sub_full = img::sub_view(view, img::make_rect(view.width, view.height));
        mv.stick_left.mask = sub_full;
        mv.stick_right.mask = sub_full;
    }
    

    static void set_map_out(img::SubView const& view, assets::controller::RectList reg, ControllerStickMaskViewMap& mv)
    {
        mv.stick_left.out = img::sub_view(view, reg.stick_left);
        mv.stick_right.out = img::sub_view(view, reg.stick_right);
    }


    static void set_mask_views(assets::DrawMaskData const& masks, img::ImageView const& out, MaskViewMapList& mv)
    {
        auto& c_mask = masks.controller_view;
        auto& k_mask = masks.keyboard_view;
        auto& m_mask = masks.mouse_view;

        auto sw = out.width;
        auto sh = out.height;
        auto cw = c_mask.width;
        auto ch = c_mask.height;
        auto kw = k_mask.width;
        auto kh = k_mask.height;
        auto mw = m_mask.width;
        auto mh = m_mask.height;

        auto c_out1 = img::sub_view(out, img::make_rect(0, 0, cw, ch));
        auto c_out2 = img::sub_view(out, img::make_rect(sw - cw, 0, cw, ch));
        auto k_out = img::sub_view(out, img::make_rect(0, sh - kh, kw, kh));
        auto m_out = img::sub_view(out, img::make_rect(sw - mw, sh - mh, mw, mh));

        auto const sub_full = [](auto const& v) { return img::sub_view(v, img::make_rect(v.width, v.height)); };

        mv.controller1.mask = sub_full(c_mask);
        mv.controller1.out = c_out1;

        mv.controller2.mask = sub_full(c_mask);
        mv.controller2.out = c_out2;

        mv.keyboard.mask = sub_full(k_mask);
        mv.keyboard.out = k_out;

        mv.mouse.mask = sub_full(m_mask);
        mv.mouse.out = m_out;

        auto creg = assets::controller::get_region_rects();
        auto kreg = assets::keyboard::get_region_rects();
        auto mreg = assets::mouse::get_region_rects();

        set_map_masks(masks.controller, mv.controller1_inputs);
        set_map_masks(masks.controller, mv.controller2_inputs);
        set_map_masks(masks.keyboard, mv.keyboard_inputs);
        set_map_masks(masks.mouse, mv.mouse_inputs);

        set_map_out(c_out1, creg, mv.controller1_inputs);
        set_map_out(c_out2, creg, mv.controller2_inputs);
        set_map_out(k_out, kreg, mv.keyboard_inputs);
        set_map_out(m_out, mreg, mv.mouse_inputs);

        set_map_masks(masks.arrow_view, mv.controller1_thumbsticks);
        set_map_masks(masks.arrow_view, mv.controller2_thumbsticks);

        set_map_out(c_out1, creg, mv.controller1_thumbsticks);
        set_map_out(c_out2, creg, mv.controller2_thumbsticks);
    }
}


/* update */

namespace game_io_test
{
    using ControllerBtnOnOff = ControllerDef<b8>;
    using KeyboardOnOff = KeyboardDef<b8>;
    using MouseBtnOnOff = MouseDef<b8>;
    using ControllerStickRotation = ControllerStickDef<Vec2Df32>;
    

    class InputList
    {
    public:

        ControllerBtnOnOff controller1;
        ControllerBtnOnOff controller2;
        KeyboardOnOff keyboard;
        MouseBtnOnOff mouse;

        Vec2Di32 mouse_pos;

        ControllerStickRotation sticks1;
        ControllerStickRotation sticks2;
    };


    static void clear_input_list(InputList& inputs)
    {
        auto const& clear = [](auto& in)
        {
            for (u32 i = 0; i < in.count; i++)
            {
                in.list[i] = 0;
            }
        };

        clear(inputs.controller1);
        clear(inputs.controller2);
        clear(inputs.keyboard);
        clear(inputs.mouse);

        inputs.mouse_pos = { 0 };
    }


    static inline void map_button(input::ButtonState const& btn, b8& dst)
    {
        dst |= btn.is_down;
    }


    static void map_button_sound(input::ButtonState const& btn, audio::Sound& sound)
    {
        if (btn.pressed)
        {
            audio::play_sound(sound);
        }
    }


    static void map_button_music(input::ButtonState const& btn, audio::Music& music)
    {
        if (btn.pressed)
        {
            audio::stop_music();
            audio::play_music(music);
        }
        else if (btn.raised)
        {
            audio::stop_music();
        }
    }


    static void map_controller_input(input::ControllerInput const& src, ControllerBtnOnOff& dst)
    {
        map_button(src.btn_dpad_up, dst.dpad_up);
        map_button(src.btn_dpad_down, dst.dpad_down);
        map_button(src.btn_dpad_left, dst.dpad_left);
        map_button(src.btn_dpad_right, dst.dpad_right);

        map_button(src.btn_a, dst.a);
        map_button(src.btn_b, dst.b);
        map_button(src.btn_x, dst.x);
        map_button(src.btn_y, dst.y);

        map_button(src.btn_start, dst.start);
        map_button(src.btn_back, dst.back);
        
        map_button(src.btn_shoulder_left, dst.shoulder_left);
        map_button(src.btn_shoulder_right, dst.shoulder_right);

        map_button(src.btn_stick_left, dst.stick_left);
        map_button(src.btn_stick_right, dst.stick_right);

        dst.stick_left |= src.stick_left.magnitude > 0.0f;
        dst.stick_right |= src.stick_right.magnitude > 0.0f;

        dst.trigger_left = src.trigger_left > 0.0f;
        dst.trigger_right = src.trigger_right > 0.0f;
    }


    static void map_joystick_input(input::JoystickInput const& src, ControllerBtnOnOff& dst)
    {
        map_button(src.btn_0, dst.x);
        map_button(src.btn_1, dst.a);
        map_button(src.btn_2, dst.b);
        map_button(src.btn_3, dst.y);
        map_button(src.btn_4, dst.shoulder_left);
        map_button(src.btn_5, dst.shoulder_right);
        map_button(src.btn_8, dst.back);
        map_button(src.btn_9, dst.start);

        map_button(src.btn_axis_up, dst.dpad_up);
        map_button(src.btn_axis_down, dst.dpad_down);
        map_button(src.btn_axis_left, dst.dpad_left);
        map_button(src.btn_axis_right, dst.dpad_right);

        /*
        Iconic Arcade

        0:X
        1:A
        2:B
        3:Y
        4:L
        5:R
        6:
        7:
        8:Select
        9:Start
                
        */
    }


    static void map_keyboard_inputsput(input::KeyboardInput const& src, KeyboardOnOff& dst)
    {
        map_button(src.kbd_1, dst.n_1);
        map_button(src.kbd_2, dst.n_2);
        map_button(src.kbd_3, dst.n_3);
        map_button(src.kbd_4, dst.n_4);

        map_button(src.kbd_W, dst.w);
        map_button(src.kbd_A, dst.a);
        map_button(src.kbd_S, dst.s);
        map_button(src.kbd_D, dst.d);

        map_button(src.kbd_space, dst.space);
    }


    static void map_mouse_inputsput(input::MouseInput const& src, MouseBtnOnOff& dst)
    {
        map_button(src.btn_left, dst.left);
        map_button(src.btn_right, dst.right);
        map_button(src.btn_middle, dst.middle);
    }


    static void map_thumbstick_input(input::ControllerInput const& src, ControllerStickRotation& dst)
    {
        dst.stick_left = src.stick_left.unit_direction;
        dst.stick_right = src.stick_right.unit_direction;
    }


    static void update_visual(Input const& src, InputList& dst)
    {
        clear_input_list(dst);

        map_controller_input(src.controllers[0], dst.controller1);
        map_controller_input(src.controllers[1], dst.controller2);

        map_joystick_input(src.joysticks[0], dst.controller1);
        map_joystick_input(src.joysticks[1], dst.controller2);

        map_keyboard_inputsput(src.keyboard, dst.keyboard);        
        map_mouse_inputsput(src.mouse, dst.mouse);

        dst.mouse_pos = src.mouse.window_pos;

        map_thumbstick_input(src.controllers[0], dst.sticks1);
        map_thumbstick_input(src.controllers[1], dst.sticks2);
    }


    static void update_sound(Input const& src, assets::SoundList& sounds)
    {
        map_button_sound(src.keyboard.kbd_W, sounds.explosion);
        map_button_sound(src.keyboard.kbd_A, sounds.laser);
        map_button_sound(src.keyboard.kbd_S, sounds.ui_confirm);
        map_button_sound(src.keyboard.kbd_D, sounds.ui_select);
    }


    static void update_music(Input const& src, assets::MusicList& music)
    {
        map_button_music(src.keyboard.kbd_1, music.game_00);
        map_button_music(src.keyboard.kbd_2, music.game_01);
        map_button_music(src.keyboard.kbd_3, music.game_02);
        map_button_music(src.keyboard.kbd_4, music.game_03);
    }
}


/* draw */

namespace game_io_test
{
    using p32 = img::Pixel;

    constexpr auto COLOR_BLACK = img::to_pixel(0);
    constexpr auto COLOR_TRANSPARENT = img::to_pixel(0, 0, 0, 0);
    constexpr auto COLOR_BACKGROUND = img::to_pixel(200);
    constexpr auto COLOR_ON = img::to_pixel(50, 255, 50);
    constexpr auto COLOR_OFF = img::to_pixel(127);


    static p32 mask_set_on(u8 m, p32 d)
    {
        switch (m)
        {
        case 1: return COLOR_BLACK;
        case 2: return COLOR_ON;
        default: return d;
        }
    }


    static p32 mask_set_off(u8 m, p32 d)
    {
        switch (m)
        {
        case 1: return COLOR_BLACK;
        case 2: return COLOR_OFF;
        default: return d;
        }
    }


    static void draw(MaskViewMap const& view, b8 is_on)
    {
        auto f = is_on ? mask_set_on : mask_set_off;

        img::transform(view.mask, view.out, f);
    }


    static void draw_thumstick_directions(ControllerStickMaskViewMap const& m, ControllerStickRotation const& rot)
    {
        auto const is_on = [](Vec2Df32 v) { return num::abs((v.x * v.x + v.y * v.y) - 1.0f) < 0.001f; };

        auto const f = [](u8 s) { return s ? COLOR_BLACK : COLOR_TRANSPARENT; };

        if (is_on(rot.stick_left))
        {
            img::rotate_blend_transform(m.stick_left.mask, m.stick_left.out, rot.stick_left, f);
        }

        if (is_on(rot.stick_right))
        {
            img::rotate_blend_transform(m.stick_right.mask, m.stick_right.out, rot.stick_right, f);
        }        

    }


    static void draw_mouse_coords(MouseMaskViewMap const& mv, Vec2Di32 pos)
    {
        auto font = ascii::Font::Joystick8;
        auto color = COLOR_BLACK;

        char buffer[16];

        auto out = mv.pos_x.out;
        img::fill(out, COLOR_BACKGROUND);
        stb::qsnprintf(buffer, 16, "X: %d", pos.x);
        ascii::render_text(buffer, out, font, color);

        out = mv.pos_y.out;
        img::fill(out, COLOR_BACKGROUND);
        stb::qsnprintf(buffer, 16, "Y: %d", pos.y);
        ascii::render_text(buffer, out, font, color);
    }


    template <class M, class I>
    static void draw_masks(M const& m, I const& in)
    {
        static_assert(M::count == I::count);
        for (u32 i = 0; i < m.count; i++)
        {
            draw(m.list[i], in.list[i]);
        }
    }


    static void draw(MaskViewMapList const& mv, InputList const& input)
    {
        draw(mv.controller1, 0);
        draw(mv.controller2, 0);
        draw(mv.keyboard, 0);
        draw(mv.mouse, 0);

        draw_masks(mv.controller1_inputs, input.controller1);
        draw_masks(mv.controller2_inputs, input.controller2);
        draw_masks(mv.keyboard_inputs, input.keyboard);
        draw_masks(mv.mouse_inputs, input.mouse);

        draw_thumstick_directions(mv.controller1_thumbsticks, input.sticks1);
        draw_thumstick_directions(mv.controller2_thumbsticks, input.sticks2);

        draw_mouse_coords(mv.mouse_inputs, input.mouse_pos);
    }
}


/* state */

namespace game_io_test
{
    class StateData
    {
    public:

        assets::SoundList sound_list;
        assets::MusicList music_list;
        
        assets::DrawMaskData masks;

        MaskViewMapList mask_views;
        InputList inputs;

        img::ImageView out_src;
        img::SubView out_dst;

        u32 out_scale = 1;

        img::Buffer32 buffer32;
        img::Buffer8 buffer8;
    };


    static StateData& get_data(AppState& state)
    {
        return *state.data;
    }


    static void destroy_state_data(AppState& state)
    {
        auto& data = get_data(state);

        mb::destroy_buffer(data.buffer32);
        mb::destroy_buffer(data.buffer8);
        mem::free(state.data);
    }


    static bool create_state_data(AppState& state)
    {
        auto state_data = mem::alloc<StateData>("StateData");
        if (!state_data)
        {
            return false;
        }

        state.data = state_data;

        auto& data = get_data(state);
        
        assets::AssetMemory am;
        if (!assets::load_asset_memory(am))
        {
            assert(" *** ASSET MEMORY ERROR *** " && false);
        }

        data.buffer8 = img::create_buffer8(assets::draw_mask_size(am), "buffer8");
        if (!data.buffer8.ok)
        {
            return false;
        }

        data.masks = assets::create_draw_mask_data(am, data.buffer8);

        auto dim = app_screen_dimensions(data.masks);
        data.buffer32 = img::create_buffer32(dim.x * dim.y, "buffer32");
        if (!data.buffer32.ok)
        {
            return false;
        }

        data.out_src = img::make_view(dim.x, dim.y, data.buffer32);
        set_mask_views(data.masks, data.out_src, data.mask_views);

        data.sound_list = assets::create_sound_list(am);
        if (!data.sound_list.ok)
        {
            return false;
        }

        data.music_list = assets::create_music_list(am);
        if (!data.music_list.ok)
        {
            return false;
        }

        clear_input_list(data.inputs);

        assets::destroy_asset_memory(am);

        audio::set_sound_volume(0.5f);
        audio::set_music_volume(1.0f);

        return true;
    }


    static void reset_data(StateData& data)
    {
        clear_input_list(data.inputs);
    }
}


/* api */

namespace game_io_test
{
    AppResult init(AppState& state)
    {
        AppResult res{};
        res.success = false;

        if (!audio::init_audio())
        {
            return res;
        }
        
        if (!create_state_data(state))
        {
            return res;
        }

        auto& data = get_data(state);

        res.screen_dimensions = app_screen_dimensions(data.masks);

        res.success = true;

        return res;
    }


    bool set_screen_memory(AppState& state, image::ImageView screen)
    {
        state.screen = screen;

        auto& data = get_data(state);

        auto dim = app_screen_dimensions(data.masks);        

        auto scale_w = screen.width / dim.x;
        auto scale_h = screen.height / dim.y;

        auto scale = num::min(scale_w, scale_h);

        if (!scale)
        {
            return false; // down scaling not supported
        }

        auto w = dim.x * scale;
        auto h = dim.y * scale;

        auto x = (screen.width - w) / 2;
        auto y = (screen.height - h) / 2;

        auto r = img::make_rect(x, y, w, h);

        data.out_dst = img::sub_view(screen, r);
        data.out_scale = scale;

        return true;
    }


    void update(AppState& state, Input const& input)
    {
        auto& kbd = input.keyboard;

        auto& data = get_data(state);

        update_visual(input, data.inputs);
        update_sound(input, data.sound_list);
        update_music(input, data.music_list);

        img::fill(data.out_src, COLOR_BACKGROUND);

        draw(data.mask_views, data.inputs);
        img::scale_up(data.out_src, data.out_dst, data.out_scale);
    }


    void reset(AppState& state)
    {
        auto& data = get_data(state);
        reset_data(data);

        img::fill(state.screen, img::to_pixel(0));
    }


    void close(AppState& state)
    {
        destroy_state_data(state);
    }
}