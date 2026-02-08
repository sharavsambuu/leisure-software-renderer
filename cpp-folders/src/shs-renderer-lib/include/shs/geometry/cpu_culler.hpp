#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: cpu_culler.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: ShapeVolume vs ConvexCell дээр суурилсан CPU batch culling utility.
*/

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
#include <variant>

#include "shs/geometry/culling_query.hpp"
#include "shs/job/parallel_for.hpp"

#if defined(SHS_HAS_XSIMD) && ((SHS_HAS_XSIMD + 0) == 1)
    #include <xsimd/xsimd.hpp>
    #define SHS_CPU_CULLER_HAS_XSIMD 1
#elif defined(__has_include)
    #if __has_include(<xsimd/xsimd.hpp>)
        #include <xsimd/xsimd.hpp>
        #define SHS_CPU_CULLER_HAS_XSIMD 1
    #else
        #define SHS_CPU_CULLER_HAS_XSIMD 0
    #endif
#else
    #define SHS_CPU_CULLER_HAS_XSIMD 0
#endif

namespace shs
{
    struct CPUCullerConfig
    {
        // Broad-phase sphere test хийх эсэх.
        bool use_broad_phase = true;
        // Broad phase Intersecting үед exact shape test руу орох эсэх.
        bool refine_intersections = true;
        // Conservative sphere Inside болсон тохиолдолд exact refine-ийг алгасах эсэх.
        bool accept_broad_inside = true;
        // XSIMD ашиглах боломжтой үед sphere-vs-cell тестийг SIMD-р түргэсгэнэ.
        bool prefer_xsimd = true;

        // Optional parallel classification.
        IJobSystem* job_system = nullptr;
        int parallel_min_items = 1024;

        CullTolerance tolerance{};
    };

    struct CPUCullerStats
    {
        uint64_t tested = 0;
        uint64_t outside = 0;
        uint64_t intersecting = 0;
        uint64_t inside = 0;
    };

    struct CPUCullResult
    {
        // One-to-one with input list.
        std::vector<CullClass> classes{};
        // Visible = Inside + Intersecting.
        std::vector<size_t> visible_indices{};
        CPUCullerStats stats{};
    };

    inline bool cpu_culler_xsimd_available()
    {
#if SHS_CPU_CULLER_HAS_XSIMD
        return true;
#else
        return false;
#endif
    }

    namespace detail
    {
        struct ConvexCellPlaneSOA
        {
            uint32_t plane_count = 0;
            alignas(64) std::array<float, k_convex_cell_max_planes> nx{};
            alignas(64) std::array<float, k_convex_cell_max_planes> ny{};
            alignas(64) std::array<float, k_convex_cell_max_planes> nz{};
            alignas(64) std::array<float, k_convex_cell_max_planes> d{};
        };

        inline ConvexCellPlaneSOA make_cell_plane_soa(const ConvexCell& cell)
        {
            ConvexCellPlaneSOA out{};
            out.plane_count = std::min(cell.plane_count, k_convex_cell_max_planes);
            for (uint32_t i = 0; i < out.plane_count; ++i)
            {
                out.nx[i] = cell.planes[i].normal.x;
                out.ny[i] = cell.planes[i].normal.y;
                out.nz[i] = cell.planes[i].normal.z;
                out.d[i] = cell.planes[i].d;
            }
            return out;
        }

        inline CullClass classify_sphere_scalar_soa(
            const Sphere& sphere,
            const ConvexCellPlaneSOA& cell_soa,
            const CullTolerance& tol)
        {
            const float r = std::max(sphere.radius, 0.0f);
            bool fully_inside = true;
            for (uint32_t i = 0; i < cell_soa.plane_count; ++i)
            {
                const float dist =
                    cell_soa.nx[i] * sphere.center.x +
                    cell_soa.ny[i] * sphere.center.y +
                    cell_soa.nz[i] * sphere.center.z +
                    cell_soa.d[i];
                if (dist < -(r + tol.outside_epsilon)) return CullClass::Outside;
                if (dist < (r + tol.inside_epsilon)) fully_inside = false;
            }
            return fully_inside ? CullClass::Inside : CullClass::Intersecting;
        }

#if SHS_CPU_CULLER_HAS_XSIMD
        inline CullClass classify_sphere_xsimd_soa(
            const Sphere& sphere,
            const ConvexCellPlaneSOA& cell_soa,
            const CullTolerance& tol)
        {
            using batch_t = xsimd::batch<float>;
            constexpr uint32_t lane_count = static_cast<uint32_t>(batch_t::size);

            const float r = std::max(sphere.radius, 0.0f);
            const batch_t cx(sphere.center.x);
            const batch_t cy(sphere.center.y);
            const batch_t cz(sphere.center.z);
            const batch_t outside_threshold(-(r + tol.outside_epsilon));
            const batch_t inside_threshold(r + tol.inside_epsilon);

            bool fully_inside = true;
            uint32_t i = 0;
            for (; i + lane_count <= cell_soa.plane_count; i += lane_count)
            {
                const batch_t nx = xsimd::load_unaligned(&cell_soa.nx[i]);
                const batch_t ny = xsimd::load_unaligned(&cell_soa.ny[i]);
                const batch_t nz = xsimd::load_unaligned(&cell_soa.nz[i]);
                const batch_t d = xsimd::load_unaligned(&cell_soa.d[i]);
                const batch_t dist = nx * cx + ny * cy + nz * cz + d;

                if (xsimd::any(dist < outside_threshold)) return CullClass::Outside;
                if (fully_inside && xsimd::any(dist < inside_threshold)) fully_inside = false;
            }

            for (; i < cell_soa.plane_count; ++i)
            {
                const float dist =
                    cell_soa.nx[i] * sphere.center.x +
                    cell_soa.ny[i] * sphere.center.y +
                    cell_soa.nz[i] * sphere.center.z +
                    cell_soa.d[i];
                if (dist < -(r + tol.outside_epsilon)) return CullClass::Outside;
                if (dist < (r + tol.inside_epsilon)) fully_inside = false;
            }
            return fully_inside ? CullClass::Inside : CullClass::Intersecting;
        }
#endif

        inline CullClass classify_sphere_fast_soa(
            const Sphere& sphere,
            const ConvexCellPlaneSOA& cell_soa,
            const CullTolerance& tol,
            bool prefer_xsimd)
        {
#if SHS_CPU_CULLER_HAS_XSIMD
            if (prefer_xsimd)
            {
                return classify_sphere_xsimd_soa(sphere, cell_soa, tol);
            }
#else
            (void)prefer_xsimd;
#endif
            return classify_sphere_scalar_soa(sphere, cell_soa, tol);
        }
    }

    inline bool cull_class_visible(CullClass c, bool include_intersecting = true)
    {
        if (c == CullClass::Inside) return true;
        if (include_intersecting && c == CullClass::Intersecting) return true;
        return false;
    }

    inline CullClass classify_cpu(
        const ShapeVolume& shape,
        const ConvexCell& cell,
        const CPUCullerConfig& cfg,
        const detail::ConvexCellPlaneSOA* cell_soa);

    inline CullClass classify_cpu(
        const ShapeVolume& shape,
        const ConvexCell& cell,
        const CPUCullerConfig& cfg)
    {
        return classify_cpu(shape, cell, cfg, nullptr);
    }

    inline CullClass classify_cpu(
        const ShapeVolume& shape,
        const ConvexCell& cell,
        const CPUCullerConfig& cfg,
        const detail::ConvexCellPlaneSOA* cell_soa)
    {
        const bool use_fast_sphere =
            (cell_soa != nullptr) &&
            cfg.prefer_xsimd &&
            cpu_culler_xsimd_available();

        const auto classify_sphere_local = [&](const Sphere& sphere) -> CullClass {
            if (use_fast_sphere)
            {
                return detail::classify_sphere_fast_soa(sphere, *cell_soa, cfg.tolerance, true);
            }
            return classify(sphere, cell, cfg.tolerance);
        };

        const auto classify_exact = [&]() -> CullClass {
            if (use_fast_sphere)
            {
                if (const Sphere* sphere = std::get_if<Sphere>(&shape.value))
                {
                    return detail::classify_sphere_fast_soa(*sphere, *cell_soa, cfg.tolerance, true);
                }
            }
            return classify(shape, cell, cfg.tolerance);
        };

        if (!cfg.use_broad_phase)
        {
            return classify_exact();
        }

        const Sphere broad = conservative_bounds_sphere(shape);
        const CullClass broad_class = classify_sphere_local(broad);
        if (broad_class == CullClass::Outside) return CullClass::Outside;

        if (broad_class == CullClass::Inside && cfg.accept_broad_inside)
        {
            return CullClass::Inside;
        }

        if (!cfg.refine_intersections)
        {
            return broad_class;
        }
        return classify_exact();
    }

    inline CPUCullResult cull_shapes_cpu(
        const ConvexCell& cell,
        const ShapeVolume* shapes,
        size_t shape_count,
        const CPUCullerConfig& cfg = {})
    {
        CPUCullResult out{};
        if (!shapes || shape_count == 0) return out;

        out.classes.assign(shape_count, CullClass::Intersecting);
        out.visible_indices.clear();
        out.visible_indices.reserve(shape_count);

        out.stats.tested = shape_count;

        if (!convex_cell_valid(cell))
        {
            // Invalid cell => conservative fallback (do not drop anything).
            for (size_t i = 0; i < shape_count; ++i)
            {
                out.visible_indices.push_back(i);
            }
            out.stats.intersecting = shape_count;
            return out;
        }

        detail::ConvexCellPlaneSOA cell_soa{};
        const detail::ConvexCellPlaneSOA* cell_soa_ptr = nullptr;
        if (cfg.prefer_xsimd && cpu_culler_xsimd_available())
        {
            cell_soa = detail::make_cell_plane_soa(cell);
            cell_soa_ptr = &cell_soa;
        }

        if (shape_count <= (size_t)std::numeric_limits<int>::max())
        {
            const int work_count = static_cast<int>(shape_count);
            parallel_for_1d(cfg.job_system, 0, work_count, std::max(1, cfg.parallel_min_items), [&](int b, int e) {
                for (int i = b; i < e; ++i)
                {
                    out.classes[(size_t)i] = classify_cpu(shapes[(size_t)i], cell, cfg, cell_soa_ptr);
                }
            });
        }
        else
        {
            // Extremely large batches use serial fallback because parallel_for_1d is int-indexed.
            for (size_t i = 0; i < shape_count; ++i)
            {
                out.classes[i] = classify_cpu(shapes[i], cell, cfg, cell_soa_ptr);
            }
        }

        for (size_t i = 0; i < shape_count; ++i)
        {
            const CullClass c = out.classes[i];
            switch (c)
            {
                case CullClass::Outside:
                    ++out.stats.outside;
                    break;
                case CullClass::Intersecting:
                    ++out.stats.intersecting;
                    out.visible_indices.push_back(i);
                    break;
                case CullClass::Inside:
                    ++out.stats.inside;
                    out.visible_indices.push_back(i);
                    break;
            }
        }
        return out;
    }

    inline CPUCullResult cull_shapes_cpu(
        const ConvexCell& cell,
        const std::vector<ShapeVolume>& shapes,
        const CPUCullerConfig& cfg = {})
    {
        if (shapes.empty()) return CPUCullResult{};
        return cull_shapes_cpu(cell, shapes.data(), shapes.size(), cfg);
    }
}
