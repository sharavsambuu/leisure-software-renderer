#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: render_pass.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <memory>
#include <string>
#include <string_view>
#include <cstdint>
#include <vector>
#include <functional>
#include <utility>
#include <algorithm>

#include "shs/app/context.hpp"
#include "shs/render/frame/frame_params.hpp"
#include "shs/render/targets/rt_handle.hpp"
#include "shs/render/targets/storage/rt_registry.hpp"
#include "shs/renderpath/planning/pass_contract.hpp"
#include "shs/rhi/command/command_desc.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    struct LightCullingRuntimePayload
    {
        uint32_t tile_size = 16u;
        uint32_t tile_count_x = 0u;
        uint32_t tile_count_y = 0u;
        uint32_t max_lights_per_tile = 128u;
        uint32_t visible_light_count = 0u;
        std::vector<uint32_t> tile_light_counts{};

        void reset()
        {
            tile_size = 16u;
            tile_count_x = 0u;
            tile_count_y = 0u;
            max_lights_per_tile = 128u;
            visible_light_count = 0u;
            tile_light_counts.clear();
        }
    };

    struct PassRuntimeInputs
    {
        const Scene* scene = nullptr;
        const FrameParams* frame = nullptr;
        RTRegistry* registry = nullptr;
        LightCullingRuntimePayload* light_culling = nullptr;
    };

    struct PassExecutionRequest
    {
        PassRuntimeInputs inputs{};
        std::vector<std::pair<std::string, RTHandle>> named_rt_handles{};
        bool depth_prepass_ready = false;
        bool light_culling_ready = false;
        bool valid = false;

        void set_named_rt(std::string key, RTHandle handle)
        {
            const auto it = std::ranges::find_if(named_rt_handles, [&](const auto& kv) {
                return kv.first == key;
            });
            if (it != named_rt_handles.end())
            {
                it->second = handle;
                return;
            }
            named_rt_handles.emplace_back(std::move(key), handle);
        }

        RTHandle find_named_rt(std::string_view key) const
        {
            const auto it = std::ranges::find_if(named_rt_handles, [&](const auto& kv) {
                return kv.first == key;
            });
            if (it != named_rt_handles.end()) return it->second;
            return RTHandle{};
        }
    };

    // ROP-2 (owner ruling 2026-09-18). The pass rim used to answer with a
    // `bool executed` beside output flags — the `(payload, bool valid)` shape
    // banned by Constitution II §8 (`value_oriented_programming.md` §8). It
    // also collapsed three distinct facts into one `false`: a malformed
    // request, unmet (retryable) prerequisites, and a deliberate decline. The
    // state is now a closed, string-free vocabulary, so no pass can report
    // "did not run" without saying *why*, and a decline is a recorded fact
    // rather than an inferred absence (ERROR_FLOW non-vacuity law).
    enum class PassOutcome : uint8_t
    {
        InvalidRequest = 0,      // the request itself is unusable
        PrerequisitesUnmet = 1,  // well-formed, but a dependency is absent/not ready
        Declined = 2,            // dependencies satisfied; the pass chose no output
        Executed = 3
    };

    struct PassExecutionResult
    {
        PassOutcome outcome = PassOutcome::InvalidRequest;
        bool produced_depth = false;
        bool produced_light_grid = false;
        bool produced_light_index_list = false;

        // The only non-executing factories. Each names its own fact and none of
        // them can produce `Executed`, so the old reason-free
        // `not_executed()` — which made all three collapses legal — is gone.
        static constexpr PassExecutionResult invalid_request()
        {
            return PassExecutionResult{PassOutcome::InvalidRequest, false, false, false};
        }

        static constexpr PassExecutionResult prerequisites_unmet()
        {
            return PassExecutionResult{PassOutcome::PrerequisitesUnmet, false, false, false};
        }

        static constexpr PassExecutionResult declined()
        {
            return PassExecutionResult{PassOutcome::Declined, false, false, false};
        }

        static constexpr PassExecutionResult executed_no_outputs()
        {
            return PassExecutionResult{PassOutcome::Executed, false, false, false};
        }

        // Derived query over the one stored source of truth — deliberately not
        // a second bit that could disagree with `outcome`.
        constexpr bool executed() const { return outcome == PassOutcome::Executed; }

        bool operator==(const PassExecutionResult&) const = default;
    };

    enum class PassResourceType : uint32_t
    {
        Unknown = 0,
        Shadow = 1,
        ColorHDR = 2,
        ColorLDR = 3,
        Motion = 4,
        Temp = 5
    };

    enum class PassResourceAccess : uint32_t
    {
        Read = 1,
        Write = 2,
        ReadWrite = 3
    };

    // `PassResourceDomain` is RETIRED (RP-2; owner ruling 2026-09-18). A
    // resource now declares a `RenderDomain` - execution unit plus an optional
    // substrate pin - defined in planning/pass_contract.hpp alongside the two
    // axes. The duplicated six-value enum and its "CPU == Software" special
    // case are gone with it.

    // RenderBackendType and Substrate are 1:1 and share ordinals, so the
    // "backend realises exactly one substrate" mapping is an identity rather
    // than a hand-maintained table.
    inline constexpr Substrate substrate_of_backend(RenderBackendType backend)
    {
        return static_cast<Substrate>(static_cast<uint8_t>(backend));
    }

    inline constexpr RenderBackendType backend_of_substrate(Substrate s)
    {
        return static_cast<RenderBackendType>(static_cast<uint8_t>(s));
    }

    // Does a declared resource domain accept this concrete backend?
    inline bool render_domain_matches_backend(const RenderDomain& d, RenderBackendType backend)
    {
        if (d.kind == RenderDomainKind::Unspecified) return true;
        if (d.kind == RenderDomainKind::Any) return true;
        const Substrate target = substrate_of_backend(backend);
        if (d.unit != execution_unit_of(target)) return false;
        if (!d.substrate_pinned) return true;
        return d.substrate == target;
    }

    struct PassResourceRef
    {
        uint64_t key = 0;
        PassResourceType type = PassResourceType::Unknown;
        PassResourceAccess access = PassResourceAccess::Read;
        RenderDomain domain = render_domain_unspecified();
        std::string name{};
    };

    struct PassIODesc
    {
        std::vector<PassResourceRef> resources{};

        void read(const PassResourceRef& r) { resources.push_back(PassResourceRef{r.key, r.type, PassResourceAccess::Read, r.domain, r.name}); }
        void write(const PassResourceRef& r) { resources.push_back(PassResourceRef{r.key, r.type, PassResourceAccess::Write, r.domain, r.name}); }
        void read_write(const PassResourceRef& r) { resources.push_back(PassResourceRef{r.key, r.type, PassResourceAccess::ReadWrite, r.domain, r.name}); }
    };

    inline bool pass_access_has_read(PassResourceAccess a)
    {
        return a == PassResourceAccess::Read || a == PassResourceAccess::ReadWrite;
    }

    inline bool pass_access_has_write(PassResourceAccess a)
    {
        return a == PassResourceAccess::Write || a == PassResourceAccess::ReadWrite;
    }

    inline PassResourceRef make_rt_resource_ref(
        const RTHandle& rt,
        PassResourceType type,
        const char* name = nullptr,
        RenderDomain domain = render_domain_unspecified()
    )
    {
        PassResourceRef out{};
        if (!rt.valid()) return out;
        out.key = ((uint64_t)type << 32ull) | (uint64_t)rt.id;
        out.type = type;
        out.access = PassResourceAccess::Read;
        out.domain = domain;
        if (name) out.name = name;
        return out;
    }

    constexpr uint64_t pass_named_resource_flag()
    {
        return (1ull << 63ull);
    }

    inline uint64_t pass_rt_resource_key(PassResourceType type, uint32_t rt_id)
    {
        return ((uint64_t)type << 32ull) | (uint64_t)rt_id;
    }

    inline uint32_t pass_rt_id_from_key(uint64_t key)
    {
        if ((key & pass_named_resource_flag()) != 0ull) return 0u;
        return (uint32_t)(key & 0xffffffffull);
    }

    inline bool pass_resource_key_is_named(uint64_t key)
    {
        return (key & pass_named_resource_flag()) != 0ull;
    }

    inline PassResourceRef make_named_resource_ref(
        const std::string& name,
        PassResourceType type,
        RenderDomain domain = render_domain_unspecified()
    )
    {
        PassResourceRef out{};
        out.type = type;
        out.access = PassResourceAccess::Read;
        out.domain = domain;
        out.name = name;
        const uint64_t h = std::hash<std::string>{}(name);
        out.key = pass_named_resource_flag() | ((uint64_t)type << 32ull) | (h & 0x7fffffffull);
        return out;
    }

    class IRenderPass
    {
    public:
        virtual ~IRenderPass() = default;
        virtual const char* id() const = 0;

        virtual bool enabled() const { return enabled_; }
        virtual void set_enabled(bool v) { enabled_ = v; }
        virtual RenderBackendType preferred_backend() const { return RenderBackendType::Software; }
        virtual RHIQueueClass preferred_queue() const { return RHIQueueClass::Graphics; }
        virtual bool supports_backend(RenderBackendType backend) const { (void)backend; return true; }
        virtual TechniquePassContract describe_contract() const { return TechniquePassContract{}; }
        virtual bool supports_technique_mode(TechniqueMode mode) const
        {
            return technique_mode_in_mask(describe_contract().supported_modes_mask, mode);
        }
        // Planning-side hook: resolve runtime inputs as an explicit value object.
        virtual PassExecutionRequest build_execution_request(
            const Context& ctx,
            const Scene& scene,
            const FrameParams& fp,
            RTRegistry& rtr) const
        {
            (void)ctx;
            PassExecutionRequest out{};
            out.inputs.scene = &scene;
            out.inputs.frame = &fp;
            out.inputs.registry = &rtr;
            out.valid = true;
            return out;
        }
        virtual bool is_interop_pass() const { return false; }
        virtual PassIODesc describe_io() const { return PassIODesc{}; }
        virtual void on_resize(Context& ctx, RTRegistry& rtr, int w, int h) { (void)ctx; (void)rtr; (void)w; (void)h; }
        virtual void on_scene_reset(Context& ctx, RTRegistry& rtr) { (void)ctx; (void)rtr; }
        virtual void reset_history(Context& ctx, RTRegistry& rtr) { (void)ctx; (void)rtr; }
        // Runtime-side hook: all pass execution must flow through resolved request values.
        virtual PassExecutionResult execute_resolved(Context& ctx, const PassExecutionRequest& request) = 0;

    protected:
        bool enabled_ = true;
    };

    } // inline namespace renderpath
}
