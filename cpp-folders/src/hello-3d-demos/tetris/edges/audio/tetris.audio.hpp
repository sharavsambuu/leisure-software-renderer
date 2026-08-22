#pragma once
// tetris/edges/audio/tetris.audio.hpp — SDL AUDIO BOUNDARY (tetris::audio)
// Verbatim port of the original demo synth (12 voices, lock-free SPSC ring).
#include <SDL2/SDL.h>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace tetris::audio {
enum SoundType : uint8_t {
    SND_NONE        = 0,
    SND_MOVE        = 1,
    SND_ROTATE      = 2,
    SND_DROP_SLAM   = 3,
    SND_LINE_CLEAR  = 4,
    SND_TETRIS_FOUR = 5,
    SND_HOLD        = 6,
    SND_GAME_OVER   = 7,
    SND_TICK        = 8   // blitz clock threshold tick (30s boundaries)
};

struct AudioEventRing {
    static const uint32_t CAP = 64;
    SoundType buffer[CAP]{};
    alignas(64) std::atomic<uint32_t> write_idx{ 0 };
    alignas(64) uint32_t              read_idx { 0 };

    inline void push(SoundType type) {
        uint32_t wi = write_idx.load(std::memory_order_relaxed);
        buffer[wi % CAP] = type;
        write_idx.store(wi + 1, std::memory_order_release);
    }
    inline bool pop(SoundType& out) {
        uint32_t wi = write_idx.load(std::memory_order_acquire);
        if (read_idx == wi) return false;
        out = buffer[read_idx % CAP];
        read_idx++;
        return true;
    }
};

struct SoundVoice {
    SoundType type     = SND_NONE;
    float     time     = 0.0f;
    float     duration = 0.1f;
    float     phase    = 0.0f;
    bool      active   = false;
};

struct TetrisAudioSynth {
    static const int MAX_VOICES = 12;
    SoundVoice       voices[MAX_VOICES];
    AudioEventRing   event_queue;

    inline void play(SoundType type) { event_queue.push(type); }

    void mix(float* stream, int frames, int channels, int sample_rate) {
        SoundType new_type;
        while (event_queue.pop(new_type)) {
            if (new_type == SND_NONE) continue;
            for (int i = 0; i < MAX_VOICES; ++i) {
                if (!voices[i].active) {
                    voices[i].type     = new_type;
                    voices[i].time     = 0.0f;
                    voices[i].phase    = 0.0f;
                    voices[i].active   = true;
                    voices[i].duration = (new_type == SND_TETRIS_FOUR) ? 0.45f
                                       : (new_type == SND_TICK)        ? 0.09f
                                                                       : 0.12f;
                    break;
                }
            }
        }

        float dt = 1.0f / (float)sample_rate;
        for (int f = 0; f < frames; ++f) {
            float sample = 0.0f;
            for (int v = 0; v < MAX_VOICES; ++v) {
                if (!voices[v].active) continue;
                SoundVoice& vox = voices[v];
                vox.time += dt;
                float p = vox.time / vox.duration;
                if (p >= 1.0f) { vox.active = false; continue; }

                float env = (1.0f - p);
                switch (vox.type) {
                    case SND_MOVE:
                        vox.phase += 400.0f * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.12f;
                        break;
                    case SND_ROTATE:
                        vox.phase += (600.0f + p * 200.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.14f;
                        break;
                    case SND_DROP_SLAM:
                        vox.phase += (140.0f - p * 80.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.28f;
                        break;
                    case SND_LINE_CLEAR:
                        vox.phase += (523.25f + p * 400.0f) * dt; // C5 to G5
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.22f;
                        break;
                    case SND_TETRIS_FOUR:
                        vox.phase += (659.25f + std::sin(p * 20.0f) * 100.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.30f;
                        break;
                    case SND_HOLD:
                        vox.phase += (320.0f + p * 150.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.15f;
                        break;
                    case SND_GAME_OVER:
                        vox.phase += (220.0f - p * 140.0f) * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.25f;
                        break;
                    case SND_TICK:
                        vox.phase += 1250.0f * dt;
                        sample += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.18f;
                        break;
                    default: break;
                }
            }
            sample = std::tanh(sample);
            for (int c = 0; c < channels; ++c) stream[f * channels + c] = sample;
        }
    }
};

static TetrisAudioSynth g_audio;
static void audio_callback(void* userdata, Uint8* stream, int len) {
    TetrisAudioSynth* synth = reinterpret_cast<TetrisAudioSynth*>(userdata);
    float* out = reinterpret_cast<float*>(stream);
    synth->mix(out, len / (int)(sizeof(float) * 2), 2, 44100);
}
} // namespace tetris::audio
