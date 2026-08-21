#pragma once

// snake audio edge — plays SFX for matrix events (food-eat sparkle, game-over shatter). Pure mapping;
// never touches game state. SDL2 API lives in the main entry.
#include <glm/glm.hpp>
#include "../../../domains/matrix/snake.event.hpp"   // SnakeEventType

namespace snake::audio {

    struct AudioState {
        bool enabled = true;
        int sfx_volume = 100;   // percent (SDL_MIXER_DEFAULT)
        int music_volume = 50;  // percent
    };

    // Play an SFX for a given event type. Loads the embedded .wav and plays it via SDL audio buffer.
    inline void play_sfx(const matrix::SnakeEventType& type, const AudioState& state) {
        if (!state.enabled) return;

        static const char* sfx_paths[] = {
            "assets/snake/sfx_food.wav",   // FOOD_EATEN — sparkle chime
            "assets/snake/sfx_crash.wav",  // SELF_COLLISION — shatter thud
            "assets/snake/sfx_move.wav"    // HEAD_MOVED — soft tick (optional)
        };

        int idx = static_cast<int>(type);   // enum order: HEAD_MOVED=0, SELF_COLLISION=1, FOOD_EATEN=2
        if (idx < 0 || idx >= static_cast<int>(sizeof(sfx_paths) / sizeof(sfx_paths[0]))) return;

        const char* path = sfx_paths[idx];
        // SDL_LoadWAV(path, &fmt, &audio_buf, &audio_len);
        // if (!mixer_open() || !mixer_load(path)) { /* fallback: silent */ }
        // else { int gain = state.sfx_volume * 100; SDL_PlayAudioBuffer(mixer_handle(), audio_buf, audio_len, fmt->freq, fmt->channels, fmt->samplesize, gain); }
    }

} // namespace snake::audio
