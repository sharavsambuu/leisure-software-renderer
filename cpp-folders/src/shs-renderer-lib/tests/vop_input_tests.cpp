#include <cmath>
#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/domains/input/input.contract.hpp"
#include "shs/domains/pod_test_kit.hpp"

// Headless tests for the input pod (R3: P3.1 + P4.1 + P4.2).
// Links only shs::renderer-values + glm: no SDL, no Vulkan, no Context.
namespace
{
    std::vector<shs::RuntimeAction> make_mixed_log()
    {
        std::vector<shs::RuntimeAction> actions{};
        actions.push_back(shs::make_move_local_action(glm::vec3(0.0f, 0.0f, 1.0f), 4.0f));
        actions.push_back(shs::make_look_action(10.0f, -3.0f, 0.01f));
        actions.push_back(shs::make_toggle_light_shafts_action());
        actions.push_back(shs::make_toggle_bot_action());
        actions.push_back(shs::make_quit_action());
        return actions;
    }

    auto reduce_via_pod = [](shs::RuntimeState& s,
                             std::span<const shs::RuntimeAction> a,
                             const shs::input::InputReduceInputs& in,
                             std::pmr::vector<shs::input::InputEvent>& e)
    {
        shs::input::reduce_input(s, a, in, e);
    };

    // P4.2 kit: same log twice -> identical states + identical event logs.
    bool test_replay_deterministic()
    {
        const shs::RuntimeState s0{};
        const std::vector<shs::RuntimeAction> actions = make_mixed_log();
        const shs::input::InputReduceInputs in{0.5f};
        return shs::pod_test::replay_is_deterministic<shs::RuntimeState, shs::RuntimeAction,
            shs::input::InputReduceInputs, shs::input::InputEvent>(
            reduce_via_pod, s0,
            std::span<const shs::RuntimeAction>{actions.data(), actions.size()}, in);
    }

    // P4.2 kit: empty log -> bit-identical state, zero events.
    bool test_empty_log_stable()
    {
        const shs::RuntimeState s0{};
        const shs::input::InputReduceInputs in{0.5f};
        return shs::pod_test::empty_log_is_stable<shs::RuntimeState, shs::RuntimeAction,
            shs::input::InputReduceInputs, shs::input::InputEvent>(
            reduce_via_pod, s0, in);
    }

    // Event pins: one fact per applied command, in order, with post-values.
    bool test_event_contents()
    {
        shs::RuntimeState s{};
        const glm::vec3 pos0 = s.camera.pos;
        const std::vector<shs::RuntimeAction> actions = make_mixed_log();

        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::input::reduce_input(s, std::span<const shs::RuntimeAction>{actions.data(), actions.size()},
            shs::input::InputReduceInputs{0.5f}, events);

        if (events.size() != 5) return false;

        const auto* moved = std::get_if<shs::input::CameraTranslatedEvent>(&events[0]);
        if (!moved) return false;
        if (s.camera.pos != pos0 + moved->world_delta) return false; // state matches its own fact

        const auto* rotated = std::get_if<shs::input::CameraRotatedEvent>(&events[1]);
        if (!rotated) return false;
        if (s.camera.yaw != glm::half_pi<float>() + 10.0f * 0.01f) return false;

        const auto* shafts = std::get_if<shs::input::RuntimeFlagToggledEvent>(&events[2]);
        if (!shafts) return false;
        if (shafts->id != shs::input::RuntimeFlagId::LightShafts) return false;
        if (shafts->value != false) return false; // default true -> toggled false
        if (s.enable_light_shafts != false) return false;

        const auto* bot = std::get_if<shs::input::RuntimeFlagToggledEvent>(&events[3]);
        if (!bot) return false;
        if (bot->id != shs::input::RuntimeFlagId::Bot) return false;
        if (bot->value != true) return false; // default false -> toggled true

        if (!std::holds_alternative<shs::input::QuitRequestedEvent>(events[4])) return false;
        if (!s.quit_requested) return false;
        return true;
    }

    // Look clamp pin: violent pitch input saturates at +-85 degrees.
    bool test_look_clamp()
    {
        shs::RuntimeState s{};
        const std::vector<shs::RuntimeAction> actions{
            shs::make_look_action(0.0f, -100000.0f, 1.0f) // pitch -= negative -> drives up
        };
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::input::reduce_input(s, std::span<const shs::RuntimeAction>{actions.data(), actions.size()},
            shs::input::InputReduceInputs{0.016f}, events);
        if (s.camera.pitch != glm::radians(85.0f)) return false;

        shs::RuntimeState s2{};
        const std::vector<shs::RuntimeAction> down{
            shs::make_look_action(0.0f, 100000.0f, 1.0f)
        };
        std::pmr::monotonic_buffer_resource arena2{4096};
        std::pmr::vector<shs::input::InputEvent> events2{&arena2};
        shs::input::reduce_input(s2, std::span<const shs::RuntimeAction>{down.data(), down.size()},
            shs::input::InputReduceInputs{0.016f}, events2);
        return s2.camera.pitch == glm::radians(-85.0f);
    }

    // Legacy/new conformance: the old signature is a pure delegate now.
    bool test_legacy_conformance()
    {
        const std::vector<shs::RuntimeAction> actions = make_mixed_log();
        const shs::RuntimeState legacy = shs::reduce_runtime_state(
            shs::RuntimeState{}, std::span<const shs::RuntimeAction>{actions.data(), actions.size()}, 0.5f);

        shs::RuntimeState fresh{};
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::input::InputEvent> events{&arena};
        shs::input::reduce_input(fresh, std::span<const shs::RuntimeAction>{actions.data(), actions.size()},
            shs::input::InputReduceInputs{0.5f}, events);

        return legacy == fresh && events.size() == 5;
    }

    // Latch reducer stays deterministic (double-run equality, kit-adjacent).
    bool test_latch_deterministic()
    {
        const shs::RuntimeInputLatch s0{};
        const std::vector<shs::RuntimeInputEvent> ev{
            shs::make_bool_input_event(shs::RuntimeInputEventType::SetForward, true),
            shs::make_mouse_delta_input_event(4.0f, -2.0f),
            shs::make_quit_input_event()
        };
        const shs::RuntimeInputLatch a = shs::reduce_runtime_input_latch(s0,
            std::span<const shs::RuntimeInputEvent>{ev.data(), ev.size()});
        const shs::RuntimeInputLatch b = shs::reduce_runtime_input_latch(s0,
            std::span<const shs::RuntimeInputEvent>{ev.data(), ev.size()});
        return a == b && a.forward && a.mouse_dx_accum == 4.0f && a.quit_requested;
    }
} // namespace

int main()
{
    bool ok = true;

    ok = test_replay_deterministic() && ok;
    ok = test_empty_log_stable() && ok;
    ok = test_event_contents() && ok;
    ok = test_look_clamp() && ok;
    ok = test_legacy_conformance() && ok;
    ok = test_latch_deterministic() && ok;

    if (!ok)
    {
        std::fprintf(stderr, "[input-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[input-tests] all tests passed\n");
    return 0;
}
