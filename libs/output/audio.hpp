#pragma once

#include "../span/span.hpp"


/* audio basic */

namespace audio
{
    class Music
    {
    public:
        i32 id;

        void* data_;

        bool is_on;
        bool is_paused;
    };


    class Sound
    {
    public:
        i32 id;

        void* data_;

        bool is_on;
    };


    void destroy_music(Music& music);

    void destroy_sound(Sound& sound);

}


namespace audio
{
    bool init_audio();

    void stop_audio();

    void close_audio();


    bool load_music_from_file(cstr music_file_path, Music& music);

    bool load_sound_from_file(cstr sound_file_path, Sound& sound);

    bool load_music_from_bytes(ByteView const& bytes, Music& music, cstr tag = "music bytes");

    bool load_sound_from_bytes(ByteView const& bytes, Sound& sound, cstr tag = "sound bytes");


    f32 set_music_volume(f32 volume);

    f32 set_sound_volume(f32 volume);

    f32 set_sound_volume(Sound& sound, f32 volume);


    void play_music(Music& music);

    void toggle_pause_music();

    void stop_music();

    void fade_in_music(Music& music, u32 fade_ms);

    void fade_out_music(u32 fade_ms);


    void play_sound(Sound& sound);

    void play_sound_loop(Sound& sound);

    void stop_sound(Sound& sound);

    void stop_sound();


    inline f32 set_master_volume(f32 volume)
    {
        return set_music_volume(set_sound_volume(volume));
    }
}