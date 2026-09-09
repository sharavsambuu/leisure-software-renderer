#pragma once

// ============================================================================
// fps/edges/audio/fps.audio.hpp — SDL AUDIO BOUNDARY (fps::audio)
// Procedural sound synthesis (no binary assets). Faithful port of the
// original demo's synth: phase-accumulator oscillators with exponential
// pitch sweeps, per-sound noise bursts, oldest-same-type voice stealing,
// sqrt(count) voice normalization, and soft clipping.
//
// Threading: main thread pushes SoundTypes into a lock-free SPSC ring;
// the SDL audio callback drains it and mixes voices. Nothing else in this
// pod is touched cross-thread.
// ============================================================================

#include <SDL2/SDL.h>

#include <atomic>
#include <cmath>
#include <cstdint>

#include <glm/gtc/constants.hpp>

namespace fps::audio {

    enum class SoundType : uint8_t {
        PLAYER_SHOOT,
        ENEMY_SHOOT,
        HITMARKER,
        ENEMY_EXPLODE,
        PLAYER_HURT,
        PLAYER_JUMP
    };

    // --- Lock-free SPSC ring for cross-thread sound requests -----------------
    // Monotonic indices; buffer slot = index % CAP (original design).
    struct AudioEventRing {
        static constexpr uint32_t         CAP = 64;
        SoundType                         buffer[CAP]{};
        alignas(64) std::atomic<uint32_t> write_idx{ 0 };
        alignas(64) uint32_t              read_idx{ 0 };

        inline void push(SoundType type) {
            const uint32_t wi = write_idx.load(std::memory_order_relaxed);
            buffer[wi % CAP]  = type;
            write_idx.store(wi + 1, std::memory_order_release);
        }

        inline bool pop(SoundType& out) {
            const uint32_t wi = write_idx.load(std::memory_order_acquire);
            if (read_idx == wi) return false;
            out = buffer[read_idx % CAP];
            ++read_idx;
            return true;
        }
    };

    // --- One procedural voice --------------------------------------------------
    struct SoundVoice {
        SoundType type     = SoundType::PLAYER_SHOOT;
        float     time     = 0.0f;
        float     duration = 0.1f;
        float     phase    = 0.0f; // cycles; sample = sin(phase * 2*pi)
        bool      active   = false;
    };

    // --- Synth state + mixing ----------------------------------------------------
    struct FpsSoundSynth {
        static constexpr int MAX_VOICES = 16;
        SoundVoice           voices[MAX_VOICES]{};
        AudioEventRing       event_queue{};
        uint32_t             rng_state = 0x853c49e6u;

        // xorshift32 -> [-1, 1)
        inline float noise() {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 17;
            rng_state ^= rng_state << 5;
            return static_cast<float>(rng_state & 0xFFFFu) / 32768.0f - 1.0f;
        }

        void play(SoundType type) { event_queue.push(type); }

        void mix(float* stream, int frames, int channels, int sample_rate) {
            // Drain queued events into free voices (steal oldest same-type
            // voice when full).
            SoundType new_type;
            while (event_queue.pop(new_type)) {
                int   slot        = -1;
                float oldest_time = -1.0f;
                int   oldest_slot = 0;

                for (int i = 0; i < MAX_VOICES; ++i) {
                    if (!voices[i].active) {
                        slot = i;
                        break;
                    }
                    if (voices[i].type == new_type && voices[i].time > oldest_time) {
                        oldest_time = voices[i].time;
                        oldest_slot = i;
                    }
                }
                if (slot < 0) slot = oldest_slot;

                SoundVoice& v = voices[slot];
                v.type     = new_type;
                v.time     = 0.0f;
                v.phase    = 0.0f;
                v.active   = true;

                switch (new_type) {
                case SoundType::PLAYER_SHOOT:  v.duration = 0.10f; break;
                case SoundType::HITMARKER:     v.duration = 0.07f; break;
                case SoundType::ENEMY_SHOOT:   v.duration = 0.14f; break;
                case SoundType::ENEMY_EXPLODE: v.duration = 0.35f; break;
                case SoundType::PLAYER_HURT:   v.duration = 0.18f; break;
                case SoundType::PLAYER_JUMP:   v.duration = 0.11f; break;
                }
            }

            const float dt = 1.0f / static_cast<float>(sample_rate);

            for (int f = 0; f < frames; ++f) {
                float mono_sample  = 0.0f;
                int   active_count = 0;

                for (int vi = 0; vi < MAX_VOICES; ++vi) {
                    if (!voices[vi].active) continue;

                    SoundVoice& vox      = voices[vi];
                    vox.time            += dt;
                    const float progress = vox.time / vox.duration;

                    if (progress >= 1.0f) {
                        vox.active = false;
                        continue;
                    }

                    ++active_count;
                    const float attack  = std::min(1.0f, vox.time / 0.002f);
                    const float release = 1.0f - progress;

                    switch (vox.type) {
                    case SoundType::PLAYER_SHOOT: {
                        const float freq = 120.0f + 850.0f * std::exp(-35.0f * vox.time);
                        vox.phase       += freq * dt;
                        const float env  = attack * release * release;
                        float s          = std::sin(vox.phase * glm::two_pi<float>());
                        if (vox.time < 0.010f) s += noise() * 0.25f;
                        mono_sample     += s * env * 0.18f;
                        break;
                    }
                    case SoundType::HITMARKER: {
                        vox.phase      += 2500.0f * dt;
                        const float env = attack * std::exp(-55.0f * vox.time);
                        const float s   = std::sin(vox.phase * glm::two_pi<float>()) * 0.7f
                                        + std::sin(vox.phase * 1.5f * glm::two_pi<float>()) * 0.3f;
                        mono_sample    += s * env * 0.16f;
                        break;
                    }
                    case SoundType::ENEMY_SHOOT: {
                        const float freq = 80.0f + 320.0f * std::exp(-18.0f * vox.time);
                        vox.phase       += freq * dt;
                        const float env  = attack * release;
                        mono_sample     += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.14f;
                        break;
                    }
                    case SoundType::ENEMY_EXPLODE: {
                        const float freq = 30.0f + 110.0f * std::exp(-9.0f * vox.time);
                        vox.phase       += freq * dt;
                        const float env  = attack * std::exp(-8.0f * vox.time);
                        const float s    = std::sin(vox.phase * glm::two_pi<float>()) * 0.6f + noise() * 0.4f;
                        mono_sample     += s * env * 0.25f;
                        break;
                    }
                    case SoundType::PLAYER_HURT: {
                        vox.phase      += 75.0f * dt;
                        const float env = attack * std::exp(-18.0f * vox.time);
                        mono_sample    += (std::sin(vox.phase * glm::two_pi<float>()) + noise() * 0.25f) * env * 0.22f;
                        break;
                    }
                    case SoundType::PLAYER_JUMP: {
                        const float freq = 150.0f + 280.0f * progress;
                        vox.phase       += freq * dt;
                        const float env  = attack * release;
                        mono_sample     += std::sin(vox.phase * glm::two_pi<float>()) * env * 0.12f;
                        break;
                    }
                    }
                }

                if (active_count > 1) {
                    mono_sample /= std::sqrt(static_cast<float>(active_count));
                }
                mono_sample = mono_sample / (1.0f + 0.8f * std::abs(mono_sample));

                for (int c = 0; c < channels; ++c) {
                    stream[f * channels + c] = mono_sample;
                }
            }
        }
    };

    inline void SDLCALL fps_audio_callback(void* userdata, Uint8* stream_bytes, int len) {
        auto*  synth    = static_cast<FpsSoundSynth*>(userdata);
        auto*  out      = reinterpret_cast<float*>(stream_bytes);
        constexpr int channels = 2;
        const int frames      = len / static_cast<int>(sizeof(float) * channels);
        synth->mix(out, frames, channels, 44100);
    }

} // namespace fps::audio