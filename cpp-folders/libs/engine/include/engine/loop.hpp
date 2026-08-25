#pragma once
// engine/loop.hpp - FIXED-STEP ACCUMULATOR (tetris-independent)
//
// Pure scheduling logic: no SDL, no rendering, no game types. The caller
// supplies a clock callback and a step callback; this module decides WHEN
// steps run. Determinism: with a synthetic clock the sequence of dt values
// is fully reproducible (headless runs use this).
//
// House pattern for future games on this skeleton.
#include <cstdint>
#include <functional>

namespace shs::engine {

    struct LoopConfig {
        float fixed_dt      = 1.0f / 60.0f; // simulation tick length
        float max_frame_dt  = 0.05f;        // clamp huge frames (alt-tab)
        int   max_catchup   = 5;            // spiral-of-death guard
    };

    // ClockFn returns seconds since arbitrary origin. StepFn receives dt
    // (always == fixed_dt) and the frame index within this loop call.
    using ClockFn = std::function<float()>;
    using StepFn  = std::function<void(float dt, uint64_t tick)>;

    // Runs ONE frame's worth of catch-up steps. Returns ticks executed.
    inline uint64_t pump_frame(const LoopConfig& cfg,
                               float raw_dt,
                               float& accumulator,
                               uint64_t& tick_counter,
                               const StepFn& step) {
        if (raw_dt > cfg.max_frame_dt) raw_dt = cfg.max_frame_dt;
        accumulator += raw_dt;

        uint64_t ran = 0;
        while (accumulator >= cfg.fixed_dt && ran < (uint64_t)cfg.max_catchup) {
            step(cfg.fixed_dt, tick_counter++);
            accumulator -= cfg.fixed_dt;
            ++ran;
        }
        return ran;
    }

    // Convenience driver: pulls time from `clock` until `should_stop`.
    inline void run_loop(const LoopConfig& cfg,
                         const ClockFn& clock,
                         const std::function<bool()>& should_stop,
                         const StepFn& step) {
        float accumulator = 0.0f;
        uint64_t tick = 0;
        float last = clock();
        while (!should_stop()) {
            const float now = clock();
            float raw = now - last;
            last = now;
            pump_frame(cfg, raw, accumulator, tick, step);
        }
    }

} // namespace shs::engine
