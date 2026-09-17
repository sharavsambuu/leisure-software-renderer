#include <atomic>
#include <cstdio>
#include <memory_resource>
#include <stdexcept>
#include <thread>
#include <variant>
#include <vector>

#include "shs/renderpath/renderpath.gateway.hpp"
#include "shs/task/thread_pool_job_system.hpp"

// Step 4.4 regression suite (engine_domain_separation_migration.md):
// thread access, arena lifetimes, shutdown ordering, rejection preservation
// and partial-batch/event-allocation failure semantics. Pins the LIFECYCLE
// POLICY documented in shs/renderpath/renderpath.gateway.hpp and the
// IJobSystem contract in shs/task/job_system.hpp:
//   - rejection: an invalid compile keeps plan + recipe + generation and the
//     batch continues with the remaining commands (per-command absorption),
//   - event-allocation failure: bad_alloc propagates (never swallowed), the
//     applied prefix persists in state, the caller keeps the event prefix,
//     the Step is lost; a healthy-arena rerun reproduces the full log,
//   - arena lifetimes: log valid as long as the arena, Step is a plain
//     value, per-frame arena reset is legal, replay parity across arenas,
//   - thread access: concurrent gateway batches on disjoint states/arenas
//     are deterministic and equal the single-thread reference,
//   - shutdown ordering: wait_idle() guarantees completion; destruction
//     drains accepted jobs; concurrent enqueue never loses work.
// Links only shs::renderer-values: no SDL, no Vulkan, no Context.
namespace
{
    // Software-backend capability snapshot (same as renderpath_tests).
    shs::renderpath::RenderPathCapabilitySet make_sw_caps()
    {
        shs::rhi::BackendCapabilities backend_caps{};
        return shs::renderpath::make_render_path_capability_set(
            shs::RenderBackendType::Software, backend_caps);
    }

    // Valid forward-lit software recipe (same as renderpath_tests).
    shs::renderpath::RenderPathRecipe make_forward_recipe(const char* name)
    {
        shs::renderpath::RenderPathRecipe recipe{};
        recipe.name = name;
        recipe.backend = shs::RenderBackendType::Software;
        recipe.render_technique = shs::RenderPathRenderingTechnique::ForwardLit;
        recipe.technique_mode = shs::TechniqueMode::Forward;
        recipe.view_culling = shs::RenderPathCullingMode::Frustum;
        recipe.pass_chain = {
            shs::renderpath::make_render_path_pass_entry(shs::PassId::ShadowMap, true),
            shs::renderpath::make_render_path_pass_entry(shs::PassId::PBRForward, true),
            shs::renderpath::make_render_path_pass_entry(shs::PassId::Tonemap, true)
        };
        return recipe;
    }

    // Install the forward recipe through the gateway (gen 0 -> 1).
    void install_forward_plan(
        shs::renderpath::RenderPathPodState& state,
        const shs::renderpath::RenderPathCompiler& compiler,
        const shs::renderpath::RenderPathCapabilitySet& caps)
    {
        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};
        const std::vector<shs::renderpath::RenderPathCommand> commands = {
            shs::renderpath::SelectPathPresetIntent{make_forward_recipe("forward_sw")}
        };
        shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);
    }

    // PMR resource with an allocation budget: the (budget_+1)-th allocation
    // request throws std::bad_alloc. Delegates to the default upstream.
    class BudgetedResource final : public std::pmr::memory_resource
    {
    public:
        explicit BudgetedResource(size_t alloc_budget) : budget_(alloc_budget) {}

    protected:
        void* do_allocate(size_t bytes, size_t alignment) override
        {
            if (count_ >= budget_) throw std::bad_alloc();
            count_ += 1;
            return upstream_->allocate(bytes, alignment);
        }

        void do_deallocate(void* p, size_t bytes, size_t alignment) override
        {
            upstream_->deallocate(p, bytes, alignment);
        }

        bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
        {
            return this == &other;
        }

    private:
        size_t budget_;
        size_t count_ = 0;
        std::pmr::memory_resource* upstream_ = std::pmr::get_default_resource();
    };

    // Toggle batch with an observable flip per command (defaults: shadow
    // occlusion false, debug aabb false, lit mode true, shadows true).
    std::vector<shs::renderpath::RenderPathCommand> make_toggle_batch()
    {
        return {
            shs::renderpath::SetRuntimeToggleIntent{
                shs::renderpath::RuntimeToggle::ShadowOcclusion, true},
            shs::renderpath::SetRuntimeToggleIntent{
                shs::renderpath::RuntimeToggle::DebugAabb, true},
            shs::renderpath::SetRuntimeToggleIntent{
                shs::renderpath::RuntimeToggle::LitMode, false},
            shs::renderpath::SetRuntimeToggleIntent{
                shs::renderpath::RuntimeToggle::Shadows, false},
        };
    }

    // Rejection preservation: an invalid compile keeps plan + recipe +
    // generation, emits PATH_SWAP_REJECTED, and the batch CONTINUES with
    // the remaining commands (per-command absorption).
    bool test_rejection_preservation_continues_batch()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps(); // no occlusion

        shs::renderpath::RenderPathPodState state{};
        install_forward_plan(state, compiler, caps);
        const shs::renderpath::RenderPathExecutionPlan plan_before = state.plan;
        const uint32_t generation_before = state.plan_generation;
        if (generation_before == 0) return false;

        std::pmr::monotonic_buffer_resource arena{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};

        // (1) view culling requiring occlusion -> rejected under these caps;
        // (2) a runtime toggle -> must still be applied after the rejection.
        const std::vector<shs::renderpath::RenderPathCommand> commands = {
            shs::renderpath::SetViewCullingModeIntent{
                shs::RenderPathCullingMode::FrustumAndOcclusion},
            shs::renderpath::SetRuntimeToggleIntent{
                shs::renderpath::RuntimeToggle::ShadowOcclusion, true},
        };
        const shs::renderpath::RenderPathStep step =
            shs::renderpath::renderpath_gateway(state, commands, compiler, caps, events);

        if (step.swaps_rejected != 1) return false;
        if (step.commands_applied != 1) return false;
        if (step.plan_generation != generation_before) return false;
        // Rejection preserves the previous plan wrt the rejected swap. The
        // FOLLOWING toggle legitimately mirrors into plan.runtime_state
        // (gen != 0), so the expected plan is plan_before + that mirror.
        shs::renderpath::RenderPathExecutionPlan plan_expected = plan_before;
        shs::renderpath::apply_runtime_toggle(
            plan_expected.runtime_state,
            shs::renderpath::RuntimeToggle::ShadowOcclusion, true);
        if (!(state.plan == plan_expected)) return false;
        if (state.recipe.view_culling != shs::RenderPathCullingMode::Frustum)
        {
            return false;
        }
        // The batch continued past the rejection: the toggle landed.
        if (!state.recipe.runtime_defaults.shadow_occlusion_enabled) return false;

        // Event order: rejection fact first, then the toggle fact.
        if (events.size() != 2) return false;
        const auto* rejected =
            std::get_if<shs::renderpath::PathSwapRejectedEvent>(&events[0]);
        const auto* toggled =
            std::get_if<shs::renderpath::RuntimeToggledEvent>(&events[1]);
        if (!rejected || !toggled) return false;
        if (rejected->reason !=
            shs::renderpath::PathSwapRejectionReason::MissingRequiredPass)
        {
            return false;
        }
        return true;
    }


    // Event-allocation failure: bad_alloc propagates out of the gateway;
    // the applied command prefix persists in state; the caller keeps the
    // event prefix; the Step is lost; a healthy-arena rerun of the same
    // batch reproduces the full log.
    bool test_partial_batch_event_alloc_failure()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();

        shs::renderpath::RenderPathPodState reference_state{};
        const std::vector<shs::renderpath::RenderPathCommand> commands =
            make_toggle_batch();
        {
            std::pmr::monotonic_buffer_resource arena{4096};
            std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};
            const shs::renderpath::RenderPathStep reference_step =
                shs::renderpath::renderpath_gateway(
                    reference_state, commands, compiler, caps, events);
            if (reference_step.commands_applied != 4) return false;
            if (events.size() != 4) return false;
        }

        // Budgeted run: allow exactly two event-vector allocations, so the
        // third event push throws out of the gateway.
        shs::renderpath::RenderPathPodState state{};
        BudgetedResource budgeted{2};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events{&budgeted};
        bool threw = false;
        try
        {
            shs::renderpath::renderpath_gateway(
                state, commands, compiler, caps, events);
        }
        catch (const std::bad_alloc&)
        {
            threw = true;
        }
        if (!threw) return false; // allocation failure must propagate

        // Caller keeps the prefix of events that landed before the failure.
        if (events.size() != 2) return false;
        if (!std::holds_alternative<shs::renderpath::RuntimeToggledEvent>(events[0]))
        {
            return false;
        }

        // State mutations persist (no rollback): a command mutates state
        // BEFORE emitting, so the failing third command's mutation landed
        // even though its event did not — state and log diverge here.
        shs::renderpath::RenderPathRuntimeState prefix_expected{};
        prefix_expected.shadow_occlusion_enabled = true;
        prefix_expected.debug_aabb = true;
        prefix_expected.lit_mode = false;
        if (!(state.recipe.runtime_defaults == prefix_expected)) return false;

        // Rerunning the same batch from the same initial state on a healthy
        // arena reproduces the full log and the full state (determinism).
        shs::renderpath::RenderPathPodState rerun_state{};
        {
            std::pmr::monotonic_buffer_resource arena{4096};
            std::pmr::vector<shs::renderpath::RenderPathEvent> events_rerun{&arena};
            shs::renderpath::renderpath_gateway(
                rerun_state, commands, compiler, caps, events_rerun);
            if (events_rerun.size() != 4) return false;
            if (!(rerun_state.recipe.runtime_defaults ==
                  reference_state.recipe.runtime_defaults))
            {
                return false;
            }
            // Log/state divergence pin: the failed run applied only the
            // first three commands (the failing third mutated state, the
            // fourth never ran), so its state is a strict prefix of the
            // rerun's final state — shadows still at its default true.
            shs::renderpath::RenderPathRuntimeState failed_prefix_expected =
                rerun_state.recipe.runtime_defaults;
            failed_prefix_expected.enable_shadows = true;
            if (!(state.recipe.runtime_defaults == failed_prefix_expected))
            {
                return false;
            }
        }
        return true;
    }



    // Arena lifetimes: the log is valid as long as the arena (Step is a
    // plain value), copying the log before reset keeps the facts, the
    // per-frame arena reset/reuse pattern is legal, and replay parity holds
    // across disjoint arenas.
    bool test_arena_lifetimes_replay_parity()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();
        const std::vector<shs::renderpath::RenderPathCommand> batch_a = {
            shs::renderpath::SelectPathPresetIntent{make_forward_recipe("forward_sw")}
        };
        const std::vector<shs::renderpath::RenderPathCommand> batch_b = {
            shs::renderpath::SetRuntimeToggleIntent{
                shs::renderpath::RuntimeToggle::DebugAabb, true},
        };

        std::pmr::monotonic_buffer_resource arena_a{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events_a{&arena_a};
        shs::renderpath::RenderPathPodState state_a{};
        const shs::renderpath::RenderPathStep step_a = shs::renderpath::renderpath_gateway(
            state_a, batch_a, compiler, caps, events_a);
        if (events_a.size() != 1) return false;
        if (step_a.plan_generation != 1) return false;

        // Copy the log before the arena is reset (long-lived vector).
        const std::vector<shs::renderpath::RenderPathEvent> kept_log{
            events_a.begin(), events_a.end()};
        // Snapshot the state after batch_a for the replay-parity comparison.
        const shs::renderpath::RenderPathPodState state_after_a = state_a;

        // Per-frame pattern: the frame arena is reset and REUSED — legal.
        arena_a.release();
        std::pmr::vector<shs::renderpath::RenderPathEvent> events_a2{&arena_a};
        const shs::renderpath::RenderPathStep step_a2 = shs::renderpath::renderpath_gateway(
            state_a, batch_b, compiler, caps, events_a2);
        if (step_a2.plan_generation != 1) return false; // toggles never recompile
        if (events_a2.size() != 1) return false;
        // The copied log survived the reset.
        if (kept_log.size() != 1) return false;
        if (!std::holds_alternative<shs::renderpath::PathCompiledEvent>(kept_log[0]))
        {
            return false;
        }

        // Replay parity: the identical batch on a disjoint arena yields the
        // same Step, same state, same event log.
        std::pmr::monotonic_buffer_resource arena_b{4096};
        std::pmr::vector<shs::renderpath::RenderPathEvent> events_b{&arena_b};
        shs::renderpath::RenderPathPodState state_b{};
        const shs::renderpath::RenderPathStep step_b = shs::renderpath::renderpath_gateway(
            state_b, batch_a, compiler, caps, events_b);
        if (!(step_a == step_b)) return false;
        if (!(state_b == state_after_a)) return false;
        // Compare against the long-lived copy — events_a points into the
        // reset arena and must not be read after release().
        if (events_b.size() != kept_log.size()) return false;
        for (size_t i = 0; i < kept_log.size(); ++i)
        {
            if (!(events_b[i] == kept_log[i])) return false;
        }
        return true;
    }


    // Thread access: concurrent gateway batches over DISJOINT states and
    // arenas are reentrant and reproduce the single-thread reference.
    bool test_concurrent_gateway_batches()
    {
        shs::renderpath::RenderPathCompiler compiler{};
        const shs::renderpath::RenderPathCapabilitySet caps = make_sw_caps();
        const std::vector<shs::renderpath::RenderPathCommand> commands =
            make_toggle_batch();

        // Single-thread reference.
        shs::renderpath::RenderPathPodState reference_state{};
        shs::renderpath::RenderPathStep reference_step{};
        {
            std::pmr::monotonic_buffer_resource ref_arena{4096};
            std::pmr::vector<shs::renderpath::RenderPathEvent> ref_events{&ref_arena};
            reference_step = shs::renderpath::renderpath_gateway(
                reference_state, commands, compiler, caps, ref_events);
        }

        auto run_batch =
            [&compiler, &caps, &commands](
                shs::renderpath::RenderPathPodState& state,
                shs::renderpath::RenderPathStep& step)
        {
            std::pmr::monotonic_buffer_resource arena{4096};
            std::pmr::vector<shs::renderpath::RenderPathEvent> events{&arena};
            step = shs::renderpath::renderpath_gateway(
                state, commands, compiler, caps, events);
        };

        shs::renderpath::RenderPathPodState state_left{};
        shs::renderpath::RenderPathPodState state_right{};
        shs::renderpath::RenderPathStep step_left{};
        shs::renderpath::RenderPathStep step_right{};

        std::thread left{run_batch, std::ref(state_left), std::ref(step_left)};
        std::thread right{run_batch, std::ref(state_right), std::ref(step_right)};
        left.join();
        right.join();

        // Deterministic and identical to the single-thread reference.
        if (!(step_left == reference_step)) return false;
        if (!(step_right == reference_step)) return false;
        if (!(state_left == reference_state)) return false;
        if (!(state_right == reference_state)) return false;
        return true;
    }

    // Shutdown ordering: wait_idle() guarantees completion; concurrent
    // enqueues never lose work; destruction drains accepted jobs.
    bool test_thread_pool_shutdown_order()
    {
        // (a) wait_idle() observes the whole queue.
        std::atomic<int> counter{0};
        {
            shs::task::ThreadPoolJobSystem system{4};
            for (int i = 0; i < 64; ++i)
            {
                system.enqueue([&counter]() { counter.fetch_add(1); });
            }
            system.wait_idle();
            if (counter.load() != 64) return false;
        }

        // (b) enqueue() is safe from multiple threads concurrently.
        {
            std::atomic<int> concurrent_counter{0};
            {
                shs::task::ThreadPoolJobSystem system{4};
                auto submit = [&system, &concurrent_counter]()
                {
                    for (int i = 0; i < 32; ++i)
                    {
                        system.enqueue(
                            [&concurrent_counter]()
                            { concurrent_counter.fetch_add(1); });
                    }
                };
                std::thread a{submit};
                std::thread b{submit};
                a.join();
                b.join();
                system.wait_idle();
            }
            if (concurrent_counter.load() != 64) return false;
        }

        // (c) Destruction drains: jobs accepted before the destructor
        // started all run by the time the scope ends.
        int drained = 0;
        {
            shs::task::ThreadPoolJobSystem system{2};
            for (int i = 0; i < 50; ++i)
            {
                system.enqueue([&drained]() { drained += 1; });
            }
            // No wait_idle() on purpose — destructor must drain.
        }
        if (drained != 50) return false;
        return true;
    }
} // namespace

int main()
{
    struct NamedTest
    {
        const char* name;
        bool (*fn)();
    };

    const NamedTest tests[] = {
        {"rejection_preservation_continues_batch",
         test_rejection_preservation_continues_batch},
        {"partial_batch_event_alloc_failure",
         test_partial_batch_event_alloc_failure},
        {"arena_lifetimes_replay_parity", test_arena_lifetimes_replay_parity},
        {"concurrent_gateway_batches", test_concurrent_gateway_batches},
        {"thread_pool_shutdown_order", test_thread_pool_shutdown_order},
    };

    int failures = 0;
    for (const NamedTest& test : tests)
    {
        const bool ok = test.fn();
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", test.name);
        if (!ok) failures += 1;
    }
    if (failures == 0)
    {
        std::printf("All %zu lifecycle-semantics tests passed.\n",
                    sizeof(tests) / sizeof(tests[0]));
    }
    return failures == 0 ? 0 : 1;
}

