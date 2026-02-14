#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: light_types.hpp
    МОДУЛЬ: lighting
    ЗОРИЛГО: Орчин үеийн гэрлийн төрлүүд, culling bound семантик, GPU pack форматыг
            нэг цэгт тодорхойлж, Vulkan/Software backend-д өргөтгөх суурь болгоно.
*/

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>

#include "shs/geometry/volumes.hpp"

namespace shs
{
    enum class LightType : uint32_t
    {
        Directional = 0,
        Point = 1,
        Spot = 2,
        RectArea = 3,
        TubeArea = 4,
        EnvironmentProbe = 5
    };

    enum class LightCullingShape : uint32_t
    {
        Infinite = 0,
        Sphere = 1,
        Cone = 2,
        OrientedBox = 3,
        Capsule = 4,
        Cylinder = 5,
        TaperedCapsule = 6,
        ConvexHull = 7,
        Mesh = 8,
        Compound = 9,
        GenericJoltBounds = 10
    };

    enum class LightAttenuationModel : uint32_t
    {
        Linear = 0,
        Smooth = 1,
        InverseSquare = 2
    };

    enum : uint32_t
    {
        LightFlagEnabled = 1u << 0,
        LightFlagAffectsDiffuse = 1u << 1,
        LightFlagAffectsSpecular = 1u << 2,
        LightFlagAffectsShadows = 1u << 3,
        LightFlagsDefault = LightFlagEnabled | LightFlagAffectsDiffuse | LightFlagAffectsSpecular
    };

    inline const char* light_type_name(LightType type)
    {
        switch (type)
        {
            case LightType::Directional: return "directional";
            case LightType::Point: return "point";
            case LightType::Spot: return "spot";
            case LightType::RectArea: return "rect_area";
            case LightType::TubeArea: return "tube_area";
            case LightType::EnvironmentProbe: return "environment_probe";
        }
        return "unknown";
    }

    inline bool is_local_cullable_light_type(LightType type)
    {
        switch (type)
        {
            case LightType::Point:
            case LightType::Spot:
            case LightType::RectArea:
            case LightType::TubeArea:
                return true;
            case LightType::Directional:
            case LightType::EnvironmentProbe:
            default:
                return false;
        }
    }

    struct LocalLightCommon
    {
        glm::vec3 position_ws{0.0f};
        float range = 1.0f;
        glm::vec3 color{1.0f};
        float intensity = 1.0f;
        uint32_t flags = LightFlagsDefault;
        LightAttenuationModel attenuation_model = LightAttenuationModel::Smooth;
        float attenuation_power = 1.0f;
        float attenuation_bias = 0.05f;
        float attenuation_cutoff = 0.0f;
    };

    struct PointLight
    {
        LocalLightCommon common{};
    };

    struct SpotLight
    {
        LocalLightCommon common{};
        // Light-ийн "гэрэл цацах чиглэл" (light -> scene).
        glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
        float inner_angle_rad = glm::radians(16.0f);
        float outer_angle_rad = glm::radians(26.0f);
    };

    struct RectAreaLight
    {
        LocalLightCommon common{};
        glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
        // Local X axis on emitter plane.
        glm::vec3 right_ws{1.0f, 0.0f, 0.0f};
        glm::vec2 half_extents{1.0f, 1.0f};
    };

    struct TubeAreaLight
    {
        LocalLightCommon common{};
        glm::vec3 axis_ws{1.0f, 0.0f, 0.0f};
        float half_length = 1.0f;
        float radius = 0.25f;
    };

    // std430-compatible generic local-light payload.
    // Fragment/compute shader аль аль нь энэ бүтэц дээр ажиллана.
    struct alignas(16) CullingLightGPU
    {
        // xyz: position ws, w: range
        glm::vec4 position_range{0.0f};
        // rgb: color, a: intensity
        glm::vec4 color_intensity{1.0f};
        // xyz: direction ws, w: spot inner cosine
        glm::vec4 direction_spot{0.0f, -1.0f, 0.0f, 1.0f};
        // xyz: rect right axis or tube axis, w: spot outer cosine
        glm::vec4 axis_spot_outer{1.0f, 0.0f, 0.0f, 0.0f};
        // xyz: rect up axis, w: shape.x (rect half width / tube half length)
        glm::vec4 up_shape_x{0.0f, 1.0f, 0.0f, 0.0f};
        // x: shape.y (rect half height / tube radius)
        // y: attenuation power
        // z: attenuation bias (for inverse square denom floor)
        // w: attenuation cutoff
        glm::vec4 shape_attenuation{0.0f, 1.0f, 0.05f, 0.0f};
        // x: LightType, y: LightCullingShape, z: flags, w: LightAttenuationModel
        glm::uvec4 type_shape_flags{0u};
        // xyz: generic culling sphere center ws, w: radius
        glm::vec4 cull_sphere{0.0f};
        // xyz: world-space AABB min (generic culling proxy)
        glm::vec4 cull_aabb_min{0.0f};
        // xyz: world-space AABB max (generic culling proxy)
        glm::vec4 cull_aabb_max{0.0f};
    };
    static_assert(sizeof(CullingLightGPU) % 16 == 0, "CullingLightGPU must stay std430-aligned");

    inline AABB aabb_from_sphere(const Sphere& s)
    {
        const float r = std::max(s.radius, 0.0f);
        const glm::vec3 ext(r);
        AABB out{};
        out.minv = s.center - ext;
        out.maxv = s.center + ext;
        return out;
    }

    inline AABB aabb_from_obb(const OBB& obb)
    {
        const glm::vec3 ex = glm::max(obb.half_extents, glm::vec3(0.0f));
        const glm::vec3 abs_x = glm::abs(obb.axis_x);
        const glm::vec3 abs_y = glm::abs(obb.axis_y);
        const glm::vec3 abs_z = glm::abs(obb.axis_z);
        const glm::vec3 world_ext = abs_x * ex.x + abs_y * ex.y + abs_z * ex.z;

        AABB out{};
        out.minv = obb.center - world_ext;
        out.maxv = obb.center + world_ext;
        return out;
    }

    inline AABB aabb_from_capsule(const Capsule& capsule)
    {
        const float r = std::max(capsule.radius, 0.0f);
        const glm::vec3 ext(r);
        AABB out{};
        out.minv = glm::min(capsule.a, capsule.b) - ext;
        out.maxv = glm::max(capsule.a, capsule.b) + ext;
        return out;
    }

    inline void assign_light_cull_bounds(
        CullingLightGPU& out,
        const Sphere& broad_sphere,
        const AABB& world_aabb)
    {
        const float r = std::max(broad_sphere.radius, 0.0f);
        out.cull_sphere = glm::vec4(broad_sphere.center, r);
        out.cull_aabb_min = glm::vec4(world_aabb.minv, 1.0f);
        out.cull_aabb_max = glm::vec4(world_aabb.maxv, 1.0f);
    }

    inline void assign_light_cull_bounds(
        CullingLightGPU& out,
        const Sphere& broad_sphere)
    {
        assign_light_cull_bounds(out, broad_sphere, aabb_from_sphere(broad_sphere));
    }

    template<typename T>
    concept LightCullSphereSource = requires(const T& src) {
        { src.bounding_sphere() } -> std::convertible_to<Sphere>;
    };

    template<typename T>
    concept LightCullAABBSource = requires(const T& src) {
        { src.world_aabb() } -> std::convertible_to<AABB>;
    };

    template<LightCullSphereSource T>
    inline void apply_light_cull_bounds_from_source(
        CullingLightGPU& out,
        const T& source,
        LightCullingShape source_shape = LightCullingShape::GenericJoltBounds)
    {
        const Sphere broad = source.bounding_sphere();
        if constexpr (LightCullAABBSource<T>)
        {
            assign_light_cull_bounds(out, broad, source.world_aabb());
        }
        else
        {
            assign_light_cull_bounds(out, broad);
        }
        out.type_shape_flags.y = static_cast<uint32_t>(source_shape);
    }

    // Note: These helpers provide geometry bounds for GPU packing and broad-phase culling.
    inline Sphere point_light_culling_sphere(const PointLight& point)
    {
        return Sphere{
            point.common.position_ws,
            std::max(point.common.range, 0.0f)
        };
    }

    inline Sphere spot_light_culling_sphere(const SpotLight& spot)
    {
        return Sphere{
            spot.common.position_ws,
            std::max(spot.common.range, 0.0f)
        };
    }

    inline OBB rect_area_light_culling_obb(const RectAreaLight& rect)
    {
        const glm::vec3 dir = normalize_or(rect.direction_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        glm::vec3 right = rect.right_ws - dir * glm::dot(rect.right_ws, dir);
        right = normalize_or(right, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 up = glm::normalize(glm::cross(dir, right));

        OBB obb{};
        obb.center = rect.common.position_ws;
        obb.axis_x = right;
        obb.axis_y = up;
        obb.axis_z = -dir;
        obb.half_extents = glm::vec3(
            std::max(rect.half_extents.x, 0.001f),
            std::max(rect.half_extents.y, 0.001f),
            std::max(rect.common.range, 0.0f) * 0.5f);
        return obb;
    }

    inline Sphere rect_area_light_culling_sphere(const RectAreaLight& rect)
    {
        const OBB obb = rect_area_light_culling_obb(rect);
        return Sphere{
            obb.center,
            glm::length(obb.half_extents)
        };
    }

    inline Capsule tube_area_light_culling_capsule(const TubeAreaLight& tube)
    {
        const glm::vec3 axis = normalize_or(tube.axis_ws, glm::vec3(1.0f, 0.0f, 0.0f));
        const float half_len = std::max(tube.half_length, 0.001f);
        Capsule c{};
        c.a = tube.common.position_ws - axis * half_len;
        c.b = tube.common.position_ws + axis * half_len;
        c.radius = std::max(tube.radius, 0.001f);
        return c;
    }

    inline Sphere tube_area_light_culling_sphere(const TubeAreaLight& tube)
    {
        const Capsule c = tube_area_light_culling_capsule(tube);
        return Sphere{
            tube.common.position_ws,
            std::max(glm::length(c.b - c.a) * 0.5f + c.radius, 0.0f)
        };
    }


    inline CullingLightGPU make_point_culling_light(const PointLight& point)
    {
        CullingLightGPU out{};
        const Sphere bounds = point_light_culling_sphere(point);
        out.position_range = glm::vec4(bounds.center, bounds.radius);
        out.color_intensity = glm::vec4(glm::max(point.common.color, glm::vec3(0.0f)), std::max(point.common.intensity, 0.0f));
        out.direction_spot = glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);
        out.axis_spot_outer = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
        out.up_shape_x = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        out.shape_attenuation = glm::vec4(
            0.0f,
            std::max(point.common.attenuation_power, 0.001f),
            std::max(point.common.attenuation_bias, 1e-5f),
            std::max(point.common.attenuation_cutoff, 0.0f));
        out.type_shape_flags = glm::uvec4(
            static_cast<uint32_t>(LightType::Point),
            static_cast<uint32_t>(LightCullingShape::Sphere),
            point.common.flags,
            static_cast<uint32_t>(point.common.attenuation_model));
        assign_light_cull_bounds(out, bounds);
        return out;
    }

    inline CullingLightGPU make_spot_culling_light(const SpotLight& spot)
    {
        CullingLightGPU out{};
        const Sphere bounds = spot_light_culling_sphere(spot);
        const glm::vec3 dir = normalize_or(spot.direction_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        const float inner = std::clamp(spot.inner_angle_rad, 0.01f, glm::half_pi<float>() - 0.01f);
        const float outer = std::clamp(std::max(inner + 0.001f, spot.outer_angle_rad), inner + 0.001f, glm::half_pi<float>() - 0.001f);
        const float inner_cos = std::cos(inner);
        const float outer_cos = std::cos(outer);

        out.position_range = glm::vec4(bounds.center, bounds.radius);
        out.color_intensity = glm::vec4(glm::max(spot.common.color, glm::vec3(0.0f)), std::max(spot.common.intensity, 0.0f));
        out.direction_spot = glm::vec4(dir, inner_cos);
        out.axis_spot_outer = glm::vec4(1.0f, 0.0f, 0.0f, outer_cos);
        out.up_shape_x = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        out.shape_attenuation = glm::vec4(
            0.0f,
            std::max(spot.common.attenuation_power, 0.001f),
            std::max(spot.common.attenuation_bias, 1e-5f),
            std::max(spot.common.attenuation_cutoff, 0.0f));
        out.type_shape_flags = glm::uvec4(
            static_cast<uint32_t>(LightType::Spot),
            static_cast<uint32_t>(LightCullingShape::Cone),
            spot.common.flags,
            static_cast<uint32_t>(spot.common.attenuation_model));
        assign_light_cull_bounds(out, bounds);
        return out;
    }

    inline CullingLightGPU make_rect_area_culling_light(const RectAreaLight& rect)
    {
        CullingLightGPU out{};
        const OBB obb = rect_area_light_culling_obb(rect);
        const Sphere bounds = rect_area_light_culling_sphere(rect);
        const glm::vec3 dir = -obb.axis_z;

        out.position_range = glm::vec4(bounds.center, bounds.radius);
        out.color_intensity = glm::vec4(glm::max(rect.common.color, glm::vec3(0.0f)), std::max(rect.common.intensity, 0.0f));
        out.direction_spot = glm::vec4(dir, 1.0f);
        out.axis_spot_outer = glm::vec4(obb.axis_x, 0.0f);
        out.up_shape_x = glm::vec4(obb.axis_y, obb.half_extents.x);
        out.shape_attenuation = glm::vec4(
            obb.half_extents.y,
            std::max(rect.common.attenuation_power, 0.001f),
            std::max(rect.common.attenuation_bias, 1e-5f),
            std::max(rect.common.attenuation_cutoff, 0.0f));
        out.type_shape_flags = glm::uvec4(
            static_cast<uint32_t>(LightType::RectArea),
            static_cast<uint32_t>(LightCullingShape::OrientedBox),
            rect.common.flags,
            static_cast<uint32_t>(rect.common.attenuation_model));
        assign_light_cull_bounds(out, bounds, aabb_from_obb(obb));
        return out;
    }

    inline CullingLightGPU make_tube_area_culling_light(const TubeAreaLight& tube)
    {
        CullingLightGPU out{};
        const Capsule cap = tube_area_light_culling_capsule(tube);
        const Sphere bounds = tube_area_light_culling_sphere(tube);
        const glm::vec3 axis = normalize_or(cap.b - cap.a, glm::vec3(1.0f, 0.0f, 0.0f));
        const float half_length = glm::length(cap.b - cap.a) * 0.5f;
        const float radius = cap.radius;

        out.position_range = glm::vec4(bounds.center, bounds.radius);
        out.color_intensity = glm::vec4(glm::max(tube.common.color, glm::vec3(0.0f)), std::max(tube.common.intensity, 0.0f));
        out.direction_spot = glm::vec4(axis, 1.0f);
        out.axis_spot_outer = glm::vec4(axis, 0.0f);
        out.up_shape_x = glm::vec4(0.0f, 1.0f, 0.0f, half_length);
        out.shape_attenuation = glm::vec4(
            radius,
            std::max(tube.common.attenuation_power, 0.001f),
            std::max(tube.common.attenuation_bias, 1e-5f),
            std::max(tube.common.attenuation_cutoff, 0.0f));
        out.type_shape_flags = glm::uvec4(
            static_cast<uint32_t>(LightType::TubeArea),
            static_cast<uint32_t>(LightCullingShape::Capsule),
            tube.common.flags,
            static_cast<uint32_t>(tube.common.attenuation_model));
        assign_light_cull_bounds(out, bounds, aabb_from_capsule(cap));
        return out;
    }
}
