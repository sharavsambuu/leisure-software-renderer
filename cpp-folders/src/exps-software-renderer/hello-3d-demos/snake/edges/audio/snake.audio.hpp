#pragma once

// snake audio edge — plays SFX for emitted events. Pure mapping; never touches game state. SDL2 audio API
// lives here (the execution boundary). This pod is STANDALONE: it owns its own local event vocabulary so
// it compiles and runs without depending on any other domain pod. Audio tones are synthesized procedurally
// into small WAV files at first use via SDL_SaveWAV, so no external asset files are required.
#include <algorithm>
#include <filesystem>
#include <glm/glm.hpp>

namespace snake::audio {

    // Local event vocabulary (standalone). Enum order is stable and drives the sfx table below.
    enum class AudioEventType : uint8_t { HEAD_MOVED, SELF_COLLISION, FOOD_EATEN };

    struct AudioState {
        bool enabled = true;
        int  sfx_volume = 100;   // percent (SDL_MIXER_DEFAULT == 128)
        int  music_volume = 50;  // percent
    };

    namespace detail {

        // Write a short procedural sine tone to <dir>/sfx_<name>.wav. Returns true on success.
        bool write_tone(const char* dir, const char* name, float freq_hz, float duration_sec, int volume_db) {
            constexpr int SAMPLE_RATE = 44100;
            constexpr int CHANNELS    = 1;
            constexpr int BITS       = 16;

            int samples_per_sec = static_cast<int>(SAMPLE_RATE * duration_sec);
            if (samples_per_sec <= 0) return false;

            std::pmr::vector<int16_t> pcm(samples_per_sec, 0);
            float gain = volume_db / 128.0f;   // SDL_MIXER_DEFAULT(128) -> full scale
            if (gain > 1.0f) gain = 1.0f;

            for (int i = 0; i < samples_per_sec; ++i) {
                float t = static_cast<float>(i) / SAMPLE_RATE;
                // Envelope: short attack + exponential decay so blips are crisp, not clicky.
                float env = std::exp(-t * 6.0f);
                pcm[i] = static_cast<int16_t>(std::sin(2.0f * glm::pi<float>() * freq_hz * t) * gain * env * 32767.0f);
            }

            const char* path = std::string(dir) + "/" + name;
            return SDL_SaveWAV(path, pcm.data(), SAMPLE_RATE, CHANNELS, BITS) == 0;
        }

    } // namespace detail

    inline void play_sfx(const AudioEventType& type, const AudioState& state) {
        if (!state.enabled) return;

        static const char* sfx_paths[] = {
            "assets/snake/sfx_move.wav",   // HEAD_MOVED — soft tick (optional)
            "assets/snake/sfx_crash.wav",  // SELF_COLLISION — shatter thud
            "assets/snake/sfx_food.wav",   // FOOD_EATEN — sparkle chime
        };

        int idx = static_cast<int>(type);
        if (idx < 0 || idx >= static_cast<int>(sizeof(sfx_paths) / sizeof(sfx_paths[0]))) return;

        const char* path = sfx_paths[idx];

        // Lazy asset generation: synthesize the WAV on first use so audio plays even when no assets exist.
        if (!std::filesystem::exists(path)) {
            std::error_code ec;
            std::string dir = path;
            auto pos = dir.find_last_of('/');
            if (pos != std::string::npos) dir = dir.substr(0, pos);
            std::filesystem::create_directories(dir, ec);

            switch (type) {
                case AudioEventType::HEAD_MOVED:   detail::write_tone(path, "sfx_move.wav", 440.0f, 0.06f, state.sfx_volume); break;
                case AudioEventType::SELF_COLLISION:detail::write_tone(path, "sfx_crash.wav", 120.0f, 0.35f, std::max(20, state.sfx_volume - 40)); break;
                case AudioEventType::FOOD_EATEN:   detail::write_tone(path, "sfx_food.wav", 880.0f, 0.12f, state.sfx_volume); break;
            }
        }

        // Play the synthesized buffer via SDL audio (SDL_LoadWAV reads the file we just wrote).
        if (!std::filesystem::exists(path)) return;   // still missing -> silent fallback, no crash
        auto* wav = SDL_LoadWAV(path, nullptr);
        if (wav) {
            int gain_db = state.sfx_volume * 100;   // percent -> linear-ish multiplier for the mixer
            SDL_PlayAudioBuffer(wav->audio_buf, wav->audio_len, wav->format->freq, wav->format->channels, wav->format->samplesize, gain_db);
        }
    }

} // namespace snake::audio
