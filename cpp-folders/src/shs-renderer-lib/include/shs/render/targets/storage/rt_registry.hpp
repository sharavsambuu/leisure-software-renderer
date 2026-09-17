#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: rt_registry.hpp
    МОДУЛЬ: gfx
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн gfx модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <cstdint>
#include <cmath>
#include <memory>
#include <memory_resource>
#include <string>
#include <type_traits>

#include "shs/containers/flat_map.hpp"
#include "shs/render/targets/rt_handle.hpp"
#include "shs/render/targets/rt_shadow.hpp"
#include "shs/render/targets/rt_types.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
    {
    enum class RTKind : uint8_t
    {
        Unknown = 0,
        Shadow = 1,
        ColorHDR = 2,
        ColorLDR = 3,
        Motion = 4
    };

    namespace detail
    {
        template <typename T> struct rt_kind_of { static constexpr RTKind value = RTKind::Unknown; };
        template <> struct rt_kind_of<RT_ShadowDepth> { static constexpr RTKind value = RTKind::Shadow; };
        template <> struct rt_kind_of<RT_ColorHDR> { static constexpr RTKind value = RTKind::ColorHDR; };
        template <> struct rt_kind_of<RT_ColorLDR> { static constexpr RTKind value = RTKind::ColorLDR; };
        template <> struct rt_kind_of<RT_ColorDepthMotion> { static constexpr RTKind value = RTKind::Motion; };
    }

    class RTRegistry
    {
    public:
        // Cold-registry container migration (W-E, 2026-09-17): keyed state is
        // node-free (shs::containers::FlatMap, §7.2 rule 5/6 — shared lib
        // utility, no private copies). Defaults to the default pmr resource;
        // pass an arena for arena-scoped lifetimes.
        explicit RTRegistry(
            std::pmr::memory_resource* resource = std::pmr::get_default_resource())
            : map_{resource},
              transient_ldr_{resource}, transient_hdr_{resource},
              transient_motion_{resource}, transient_shadow_{resource} {}

        struct Extent
        {
            int w = 0;
            int h = 0;
            bool valid() const { return w > 0 && h > 0; }
        };

        void reset()
        {
            next_id_ = 1;
            map_.clear();
            transient_ldr_.clear();
            transient_hdr_.clear();
            transient_motion_.clear();
            transient_shadow_.clear();
        }

        // Register an existing RT pointer from demo code.
        template<typename THandle>
        THandle reg(void* ptr)
        {
            return reg_impl<THandle>(ptr, RTKind::Unknown);
        }

        template<typename THandle, typename TRT>
        THandle reg(TRT* ptr)
        {
            using T = typename std::remove_cv<TRT>::type;
            return reg_impl<THandle>((void*)ptr, detail::rt_kind_of<T>::value);
        }

        template<typename THandle>
        bool has(THandle h) const
        {
            return map_.contains(h.id);
        }

        template<typename THandle>
        void* get(THandle h) const
        {
            const Entry* e = map_.find(h.id);
            return e ? e->ptr : nullptr;
        }

        template<typename THandle>
        RTKind kind(THandle h) const
        {
            const Entry* e = map_.find(h.id);
            return e ? e->kind : RTKind::Unknown;
        }

        RTHandle ensure_transient_color_ldr(const std::string& name, int w, int h, Color clear = {0, 0, 0, 255})
        {
            TransientLdr* slot = transient_ldr_.find(name);
            if (!slot)
            {
                auto rt = std::make_unique<RT_ColorLDR>(w, h, clear);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::ColorLDR);
                slot = transient_ldr_.insert_or_assign(name, TransientLdr{hdl, std::move(rt)});
                return slot->handle;
            }

            RT_ColorLDR* rt = slot->rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h)
            {
                rt->w = w;
                rt->h = h;
                rt->color.resize(w, h, clear);
            }
            return slot->handle;
        }

        RTHandle ensure_transient_color_hdr(const std::string& name, int w, int h, ColorF clear = {0.0f, 0.0f, 0.0f, 1.0f})
        {
            TransientHdr* slot = transient_hdr_.find(name);
            if (!slot)
            {
                auto rt = std::make_unique<RT_ColorHDR>(w, h, clear);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::ColorHDR);
                slot = transient_hdr_.insert_or_assign(name, TransientHdr{hdl, std::move(rt)});
                return slot->handle;
            }

            RT_ColorHDR* rt = slot->rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h)
            {
                rt->w = w;
                rt->h = h;
                rt->color.resize(w, h, clear);
            }
            return slot->handle;
        }

        RTHandle ensure_transient_motion(const std::string& name, int w, int h, float zn, float zf, Color clear = {0, 0, 0, 255})
        {
            TransientMotion* slot = transient_motion_.find(name);
            if (!slot)
            {
                auto rt = std::make_unique<RT_ColorDepthMotion>(w, h, zn, zf, clear);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::Motion);
                slot = transient_motion_.insert_or_assign(name, TransientMotion{hdl, std::move(rt)});
                return slot->handle;
            }

            RT_ColorDepthMotion* rt = slot->rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h || std::abs(rt->zn - zn) > 1e-6f || std::abs(rt->zf - zf) > 1e-6f)
            {
                *rt = RT_ColorDepthMotion(w, h, zn, zf, clear);
            }
            return slot->handle;
        }

        RTHandle ensure_transient_shadow(const std::string& name, int w, int h)
        {
            TransientShadow* slot = transient_shadow_.find(name);
            if (!slot)
            {
                auto rt = std::make_unique<RT_ShadowDepth>(w, h);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::Shadow);
                slot = transient_shadow_.insert_or_assign(name, TransientShadow{hdl, std::move(rt)});
                return slot->handle;
            }

            RT_ShadowDepth* rt = slot->rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h)
            {
                rt->resize(w, h);
            }
            return slot->handle;
        }

        template<typename THandle>
        Extent extent(THandle h) const
        {
            Extent e{};
            const Entry* entry = map_.find(h.id);
            if (!entry || !entry->ptr) return e;
            switch (entry->kind)
            {
                case RTKind::ColorLDR:
                {
                    auto* p = static_cast<const RT_ColorLDR*>(entry->ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::ColorHDR:
                {
                    auto* p = static_cast<const RT_ColorHDR*>(entry->ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::Motion:
                {
                    auto* p = static_cast<const RT_ColorDepthMotion*>(entry->ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::Shadow:
                {
                    auto* p = static_cast<const RT_ShadowDepth*>(entry->ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::Unknown:
                default:
                    break;
            }
            return e;
        }

    private:
        struct Entry
        {
            void* ptr = nullptr;
            RTKind kind = RTKind::Unknown;
        };

        template<typename THandle>
        THandle reg_impl(void* ptr, RTKind kind)
        {
            THandle h{};
            h.id = next_id_++;
            map_.insert_or_assign(h.id, Entry{ptr, kind});
            return h;
        }

        struct TransientLdr
        {
            RTHandle handle{};
            std::unique_ptr<RT_ColorLDR> rt{};
        };
        struct TransientHdr
        {
            RTHandle handle{};
            std::unique_ptr<RT_ColorHDR> rt{};
        };
        struct TransientMotion
        {
            RTHandle handle{};
            std::unique_ptr<RT_ColorDepthMotion> rt{};
        };
        struct TransientShadow
        {
            RTHandle handle{};
            std::unique_ptr<RT_ShadowDepth> rt{};
        };

        uint32_t next_id_ = 1;
        containers::FlatMap<uint32_t, Entry> map_;
        containers::FlatMap<std::string, TransientLdr> transient_ldr_;
        containers::FlatMap<std::string, TransientHdr> transient_hdr_;
        containers::FlatMap<std::string, TransientMotion> transient_motion_;
        containers::FlatMap<std::string, TransientShadow> transient_shadow_;
    };

    } // inline namespace render
}
