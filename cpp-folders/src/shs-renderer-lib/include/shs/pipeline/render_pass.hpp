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
#include <cstdint>
#include <vector>
#include <functional>

#include "shs/core/context.hpp"
#include "shs/frame/frame_params.hpp"
#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_registry.hpp"
#include "shs/pipeline/pass_contract.hpp"
#include "shs/rhi/command/command_desc.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
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

    enum class PassResourceDomain : uint8_t
    {
        Any = 0,
        CPU = 1,
        GPU = 2,
        Software = 3,
        OpenGL = 4,
        Vulkan = 5
    };

    inline const char* pass_resource_domain_name(PassResourceDomain d)
    {
        switch (d)
        {
            case PassResourceDomain::Any: return "any";
            case PassResourceDomain::CPU: return "cpu";
            case PassResourceDomain::GPU: return "gpu";
            case PassResourceDomain::Software: return "software";
            case PassResourceDomain::OpenGL: return "opengl";
            case PassResourceDomain::Vulkan: return "vulkan";
        }
        return "unknown";
    }

    inline bool pass_resource_domain_matches_backend(PassResourceDomain d, RenderBackendType backend)
    {
        switch (d)
        {
            case PassResourceDomain::Any: return true;
            case PassResourceDomain::CPU: return backend == RenderBackendType::Software;
            case PassResourceDomain::GPU: return backend == RenderBackendType::OpenGL || backend == RenderBackendType::Vulkan;
            case PassResourceDomain::Software: return backend == RenderBackendType::Software;
            case PassResourceDomain::OpenGL: return backend == RenderBackendType::OpenGL;
            case PassResourceDomain::Vulkan: return backend == RenderBackendType::Vulkan;
        }
        return false;
    }

    inline bool pass_resource_domains_compatible(PassResourceDomain a, PassResourceDomain b)
    {
        if (a == PassResourceDomain::Any || b == PassResourceDomain::Any) return true;
        if (a == b) return true;
        if ((a == PassResourceDomain::GPU && (b == PassResourceDomain::OpenGL || b == PassResourceDomain::Vulkan))
            || (b == PassResourceDomain::GPU && (a == PassResourceDomain::OpenGL || a == PassResourceDomain::Vulkan)))
        {
            return true;
        }
        if ((a == PassResourceDomain::CPU && b == PassResourceDomain::Software)
            || (b == PassResourceDomain::CPU && a == PassResourceDomain::Software))
        {
            return true;
        }
        return false;
    }

    struct PassResourceRef
    {
        uint64_t key = 0;
        PassResourceType type = PassResourceType::Unknown;
        PassResourceAccess access = PassResourceAccess::Read;
        PassResourceDomain domain = PassResourceDomain::Any;
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
        PassResourceDomain domain = PassResourceDomain::Any
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
        PassResourceDomain domain = PassResourceDomain::Any
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
        virtual bool is_interop_pass() const { return false; }
        virtual PassIODesc describe_io() const { return PassIODesc{}; }
        virtual void on_resize(Context& ctx, RTRegistry& rtr, int w, int h) { (void)ctx; (void)rtr; (void)w; (void)h; }
        virtual void on_scene_reset(Context& ctx, RTRegistry& rtr) { (void)ctx; (void)rtr; }
        virtual void reset_history(Context& ctx, RTRegistry& rtr) { (void)ctx; (void)rtr; }

        virtual void execute(Context& ctx, const Scene& scene, const FrameParams& fp, RTRegistry& rtr) = 0;

    protected:
        bool enabled_ = true;
    };
}
