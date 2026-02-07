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
#include <string>
#include <type_traits>
#include <unordered_map>

#include "shs/gfx/rt_handle.hpp"
#include "shs/gfx/rt_shadow.hpp"
#include "shs/gfx/rt_types.hpp"

namespace shs
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
            return map_.find(h.id) != map_.end();
        }

        template<typename THandle>
        void* get(THandle h) const
        {
            auto it = map_.find(h.id);
            return (it == map_.end()) ? nullptr : it->second.ptr;
        }

        template<typename THandle>
        RTKind kind(THandle h) const
        {
            auto it = map_.find(h.id);
            return (it == map_.end()) ? RTKind::Unknown : it->second.kind;
        }

        RTHandle ensure_transient_color_ldr(const std::string& name, int w, int h, Color clear = {0, 0, 0, 255})
        {
            auto it = transient_ldr_.find(name);
            if (it == transient_ldr_.end())
            {
                auto rt = std::make_unique<RT_ColorLDR>(w, h, clear);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::ColorLDR);
                auto [ins_it, _] = transient_ldr_.emplace(name, TransientLdr{hdl, std::move(rt)});
                return ins_it->second.handle;
            }

            RT_ColorLDR* rt = it->second.rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h)
            {
                rt->w = w;
                rt->h = h;
                rt->color.resize(w, h, clear);
            }
            return it->second.handle;
        }

        RTHandle ensure_transient_color_hdr(const std::string& name, int w, int h, ColorF clear = {0.0f, 0.0f, 0.0f, 1.0f})
        {
            auto it = transient_hdr_.find(name);
            if (it == transient_hdr_.end())
            {
                auto rt = std::make_unique<RT_ColorHDR>(w, h, clear);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::ColorHDR);
                auto [ins_it, _] = transient_hdr_.emplace(name, TransientHdr{hdl, std::move(rt)});
                return ins_it->second.handle;
            }

            RT_ColorHDR* rt = it->second.rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h)
            {
                rt->w = w;
                rt->h = h;
                rt->color.resize(w, h, clear);
            }
            return it->second.handle;
        }

        RTHandle ensure_transient_motion(const std::string& name, int w, int h, float zn, float zf, Color clear = {0, 0, 0, 255})
        {
            auto it = transient_motion_.find(name);
            if (it == transient_motion_.end())
            {
                auto rt = std::make_unique<RT_ColorDepthMotion>(w, h, zn, zf, clear);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::Motion);
                auto [ins_it, _] = transient_motion_.emplace(name, TransientMotion{hdl, std::move(rt)});
                return ins_it->second.handle;
            }

            RT_ColorDepthMotion* rt = it->second.rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h || std::abs(rt->zn - zn) > 1e-6f || std::abs(rt->zf - zf) > 1e-6f)
            {
                *rt = RT_ColorDepthMotion(w, h, zn, zf, clear);
            }
            return it->second.handle;
        }

        RTHandle ensure_transient_shadow(const std::string& name, int w, int h)
        {
            auto it = transient_shadow_.find(name);
            if (it == transient_shadow_.end())
            {
                auto rt = std::make_unique<RT_ShadowDepth>(w, h);
                RTHandle hdl = reg_impl<RTHandle>((void*)rt.get(), RTKind::Shadow);
                auto [ins_it, _] = transient_shadow_.emplace(name, TransientShadow{hdl, std::move(rt)});
                return ins_it->second.handle;
            }

            RT_ShadowDepth* rt = it->second.rt.get();
            if (!rt) return RTHandle{};
            if (rt->w != w || rt->h != h)
            {
                rt->resize(w, h);
            }
            return it->second.handle;
        }

        template<typename THandle>
        Extent extent(THandle h) const
        {
            Extent e{};
            auto it = map_.find(h.id);
            if (it == map_.end() || !it->second.ptr) return e;
            switch (it->second.kind)
            {
                case RTKind::ColorLDR:
                {
                    auto* p = static_cast<const RT_ColorLDR*>(it->second.ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::ColorHDR:
                {
                    auto* p = static_cast<const RT_ColorHDR*>(it->second.ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::Motion:
                {
                    auto* p = static_cast<const RT_ColorDepthMotion*>(it->second.ptr);
                    e.w = p ? p->w : 0;
                    e.h = p ? p->h : 0;
                    break;
                }
                case RTKind::Shadow:
                {
                    auto* p = static_cast<const RT_ShadowDepth*>(it->second.ptr);
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
            map_[h.id] = Entry{ptr, kind};
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
        std::unordered_map<uint32_t, Entry> map_{};
        std::unordered_map<std::string, TransientLdr> transient_ldr_{};
        std::unordered_map<std::string, TransientHdr> transient_hdr_{};
        std::unordered_map<std::string, TransientMotion> transient_motion_{};
        std::unordered_map<std::string, TransientShadow> transient_shadow_{};
    };
}
