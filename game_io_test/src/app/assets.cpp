#pragma once

#include "app.hpp"
#include "../../../libs/output/audio.hpp"

//#define __EMSCRIPTEN__ 1

#ifdef __EMSCRIPTEN__

#include <emscripten/fetch.h>

#else

#include <filesystem>
#include <fstream>

#endif




namespace game_io_test
{

    namespace fs = std::filesystem;

/* bin data */

namespace assets
{
    constexpr auto BIN_DATA_FALLBACK = "/home/adam/Repos/SDL3Game/game_io_test/src/res/io_test_data.bin";
    constexpr auto BIN_DATA_PATH = "./io_test_data.bin";

    static cstr get_file_name(cstr full_path)
    {
        auto str = span::to_string_view(full_path);
        auto c = str.data + str.length;

        for (; c >= str.data && *c != '/'; c--)
        { }

        //return (cstr)(c + 1); // no leading '/'
        return (cstr)c; // keep leading '/'
    }


    static MemoryBuffer<u8> read_bytes(cstr path)
    {
        MemoryBuffer<u8> buffer;

        auto size = fs::file_size(path);
        if (size == 0 || !mb::create_buffer(buffer, size, get_file_name(path)))
        {
            assert(false && " *** no file memory *** ");
            return buffer;
        }

        std::ifstream file(path, std::ios::binary);
        if (!file.is_open())
        {
            assert(false && " *** file error *** ");
            mb::destroy_buffer(buffer);
            return buffer;
        }

        file.read((char*)buffer.data_, size);

        file.close();
        return buffer;
    }


    class AssetMemory
    {
    public:

        struct
        {
            img::Image controller;
            img::Image keyboard;
            img::Image mouse;
            img::Image arrow;

        } image;

        struct
        {
            ByteView A;
            ByteView B;
            ByteView C;
            ByteView D;

        } music;

        struct
        {
            ByteView laser;
            ByteView explosion;
            ByteView confirm;
            ByteView select;

        } sound;

        MemoryBuffer<u8> buffer;
    };


    static void destroy_asset_memory(AssetMemory& memory)
    {
        img::destroy_image(memory.image.controller);
        img::destroy_image(memory.image.keyboard);
        img::destroy_image(memory.image.mouse);
        img::destroy_image(memory.image.arrow);

        mb::destroy_buffer(memory.buffer);
    }


    static bool load_asset_memory(AssetMemory& memory)
    {
    #include "../res/asset_sizes.cpp"

        img::Buffer8 buffer;
        if (fs::exists(BIN_DATA_PATH))
        {
            buffer = read_bytes(BIN_DATA_PATH);
        }
        else if (fs::exists(BIN_DATA_FALLBACK))
        {
            buffer = read_bytes(BIN_DATA_FALLBACK);
        }
        else
        {
            return false;
        }
        
        if (!buffer.ok)
        {
            return false;
        }

        auto const make_view = [&](auto const& s) { return span::make_view(buffer.data_ + s.offset, s.size); };

        auto const read_image = [&](auto const& s, auto& im)
        {
            return img::read_image_from_memory(make_view(s), im);
        };

        bool res = false;
        res = read_image(asset_sizes.masks.controller, memory.image.controller);
        if (!res)
        {
            return false;
        }

        res = read_image(asset_sizes.masks.keyboard, memory.image.keyboard);
        if (!res)
        {
            return false;
        }

        res = read_image(asset_sizes.masks.mouse, memory.image.mouse);
        if (!res)
        {
            return false;
        }

        res = read_image(asset_sizes.masks.arrow, memory.image.arrow);
        if (!res)
        {
            return false;
        }

        memory.music.A = make_view(asset_sizes.music.game_00);
        memory.music.B = make_view(asset_sizes.music.game_01);
        memory.music.C = make_view(asset_sizes.music.game_02);
        memory.music.D = make_view(asset_sizes.music.game_03);

        memory.sound.confirm = make_view(asset_sizes.sfx.confirmation_002);
        memory.sound.explosion = make_view(asset_sizes.sfx.explosionCrunch_003);
        memory.sound.laser = make_view(asset_sizes.sfx.laserRetro_000);
        memory.sound.select = make_view(asset_sizes.sfx.open_001);

        return true;
    }
}


/* images */

namespace assets
{

    using p32 = img::Pixel;


    static u8 to_u8_mask(p32 p)
    {
        if (!p.alpha) // transparent
        {
            return 0;
        }

        auto sum = p.red + p.green + p.blue;
        if (!sum) // black
        {
            return 1;
        }

        return 2;
    }


    static img::GrayView make_mask(img::Image const& img32, img::Buffer8& buffer)
    {
        auto w = img32.width;
        auto h = img32.height;

        auto mask = img::make_view(w, h, buffer);
        auto view = img::make_view(img32);

        img::transform(view, mask, to_u8_mask);

        return mask;
    }

}


/* controller */

namespace assets
{

namespace controller
{
    template <class T>
    class ControllerDef
    {
    public:
        static constexpr u32 count = 16;

        union
        {
            T list[count];

            struct
            {
                //T all;

                T dpad_up;
                T dpad_down;
                T dpad_left;
                T dpad_right;

                T a;
                T b;
                T x;
                T y;

                T start;
                T back;

                T shoulder_left;
                T shoulder_right;

                T stick_left;
                T stick_right;

                T trigger_left;
                T trigger_right;
            };
        };
    };


    template <class T>
    class ControllerStickDef
    {
    public:
        T stick_left;
        T stick_right;
    };


    using MaskList = ControllerDef<img::GraySubView>;
    using RectList = ControllerDef<Rect2Du32>;


    static RectList get_region_rects()
    {
        RectList r{};

        r.trigger_left  = img::make_rect(17, 4, 19, 15);
        r.trigger_right = img::make_rect(156, 4, 19, 15);

        r.shoulder_left  = img::make_rect(17, 21, 19, 8);
        r.shoulder_right = img::make_rect(156, 21, 19, 8);

        r.back  = img::make_rect(74, 23, 16, 9);
        r.start = img::make_rect(102, 23, 16, 9);

        r.dpad_up    = img::make_rect(21, 32, 10, 17);
        r.dpad_down  = img::make_rect(21, 60, 10, 17);
        r.dpad_left  = img::make_rect(5, 50, 17, 10);
        r.dpad_right = img::make_rect(32, 50, 17, 10);

        r.stick_left  = img::make_rect(59, 44, 25, 25);
        r.stick_right = img::make_rect(108, 44, 25, 25);

        r.a = img::make_rect(158, 62, 15, 15);
        r.b = img::make_rect(173, 47, 15, 15);
        r.x = img::make_rect(143, 47, 15, 15);
        r.y = img::make_rect(158, 32, 15, 15);

        return r;
    }


}

}


/* keyboard */

namespace assets
{

namespace keyboard
{
    template <class T>
    class KeyboardDef
    {
    public:
        static constexpr u32 count = 9;

        union
        {
            T list[count];

            struct
            {
                //T all;

                T n_1;
                T n_2;
                T n_3;
                T n_4;

                T w;
                T a;
                T s;
                T d;

                T space;
            };
        };
    };


    using MaskList = KeyboardDef<img::GraySubView>;
    using RectList = KeyboardDef<Rect2Du32>;


    static RectList get_region_rects()
    {
        RectList r{};

        r.n_1 = img::make_rect(20, 2, 16, 16);
        r.n_2 = img::make_rect(38, 2, 16, 16);
        r.n_3 = img::make_rect(56, 2, 16, 16);
        r.n_4 = img::make_rect(74, 2, 16, 16);

        r.w = img::make_rect(47, 20, 16, 16);
        r.a = img::make_rect(34, 38, 16, 16);
        r.s = img::make_rect(52, 38, 16, 16);
        r.d = img::make_rect(70, 38, 16, 16);

        r.space = img::make_rect(83, 74, 106, 16);

        return r;
    }
    
}
}


/* mouse */

namespace assets
{
namespace mouse
{
    template <class T>
    class MouseDef
    {
    public:
        static constexpr u32 count = 5;

        union
        {
            T list[count];

            struct
            {
                //T all;

                T left;
                T right;
                T middle;

                T pos_x;
                T pos_y;
            };
        };
    };


    using MaskList = MouseDef<img::GraySubView>;
    using RectList = MouseDef<Rect2Du32>;


    static RectList get_region_rects()
    {
        RectList r{};

        r.left   = img::make_rect(1, 1, 30, 30);
        r.right  = img::make_rect(49, 1, 30, 30);
        r.middle = img::make_rect(32, 1, 14, 30);
        r.pos_x  = img::make_rect(6, 44, 68, 8);
        r.pos_y  = img::make_rect(6, 56, 68, 8);

        return r;
    }

    
}

}


namespace assets
{
    class DrawMaskData
    {
    public:

        controller::MaskList controller;
        keyboard::MaskList keyboard;
        mouse::MaskList mouse;

        img::GrayView controller_view;
        img::GrayView keyboard_view;
        img::GrayView mouse_view;

        img::GrayView arrow_view;
    };


    static u32 draw_mask_size(AssetMemory const& am)
    {
        auto c = am.image.controller;
        auto k = am.image.keyboard;
        auto m = am.image.mouse;
        auto a = am.image.arrow;

        auto cn = c.width * c.height;
        auto kn = k.width * k.height;
        auto mn = m.width * m.height;
        auto an = a.width * a.height;

        return cn + kn + mn + an;
    }


    template <class V, class R, class M>
    static void set_mask_regions(V const& view, R const& reg, M& mask)
    {
        static_assert(R::count == M::count);
        for (u32 i = 0; i < reg.count; i++)
        {
            mask.list[i] = img::sub_view(view, reg.list[i]);
        }
    }


    static DrawMaskData create_draw_mask_data(AssetMemory const& am, img::Buffer8& buffer)
    {
        auto cr = controller::get_region_rects();
        auto kr = keyboard::get_region_rects();
        auto mr = mouse::get_region_rects();

        DrawMaskData data;

        auto cmv = make_mask(am.image.controller, buffer);
        auto kmv = make_mask(am.image.keyboard, buffer);
        auto mmv = make_mask(am.image.mouse, buffer);
        auto amv = make_mask(am.image.arrow, buffer);

        set_mask_regions(cmv, cr, data.controller);
        set_mask_regions(kmv, kr, data.keyboard);
        set_mask_regions(mmv, mr, data.mouse);

        data.controller_view = cmv;
        data.keyboard_view = kmv;
        data.mouse_view = mmv;
        data.arrow_view = amv;

        return data;
    }

}


/* sounds */

namespace assets
{
    using Sound = audio::Sound;    


    class SoundList
    {
    public:
        static constexpr u32 count = 4;

        bool ok = false;

        union
        {
            Sound list[count];

            struct
            {
                Sound laser;
                Sound explosion;
                Sound ui_confirm;
                Sound ui_select;
            };
        };
    };


    static void destroy_sound_list(SoundList& s)
    {
        for (u32 i = 0; i < s.count; i++)
        {
            audio::destroy_sound(s.list[i]);
        }
    }


    static SoundList create_sound_list(AssetMemory const& am)
    {
        SoundList sounds;
        sounds.ok = false;

        bool res = true;
        res &= audio::load_sound_from_bytes(am.sound.laser, sounds.laser);
        assert(res && " *** laser sound *** ");

        res &= audio::load_sound_from_bytes(am.sound.explosion, sounds.explosion);
        assert(res && " *** explosion sound *** ");

        res &= audio::load_sound_from_bytes(am.sound.confirm, sounds.ui_confirm);
        assert(res && " *** confirm sound *** ");

        res &= audio::load_sound_from_bytes(am.sound.select, sounds.ui_select);
        assert(res && " *** select sound *** ");

        sounds.ok = res;       

        return sounds;
    }

}


/* music */

namespace assets
{
    using Music = audio::Music;


    class MusicList
    {
    public:
        static constexpr u32 count = 4;

        bool ok = false;

        union
        {
            Music list[count];

            struct
            {
                Music game_00;
                Music game_01;
                Music game_02;
                Music game_03;
            };
        };
    };


    static void destroy_music_list(MusicList& m)
    {
        for (u32 i = 0; i < m.count; i++)
        {
            audio::destroy_music(m.list[i]);
        }
    }


    static MusicList create_music_list(AssetMemory const& am)
    {
        MusicList music;
        music.ok = false;

        bool res = true;

        res &= audio::load_music_from_bytes(am.music.A, music.game_00);
        assert(res && " *** music A *** ");

        res &= audio::load_music_from_bytes(am.music.B, music.game_01);
        assert(res && " *** music B *** ");

        res &= audio::load_music_from_bytes(am.music.C, music.game_02);
        assert(res && " *** music C *** ");

        res &= audio::load_music_from_bytes(am.music.D, music.game_03);
        assert(res && " *** music D *** ");

        music.ok = true;
        return music;
    }
}


} // game_io_test
