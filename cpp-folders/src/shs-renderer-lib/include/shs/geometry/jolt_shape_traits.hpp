#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_shape_traits.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: C++20 concept болон consteval-д суурилсан Jolt shape trait-ууд.
            Compile-time-д shape-ийн шинж чанарыг шалгах,
            culling API-д type constraint тавих зорилготой.
*/

#include <concepts>
#include <cstdint>
#include <type_traits>

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <Jolt/Jolt.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include <glm/glm.hpp>
#include "shs/geometry/volumes.hpp"

namespace shs
{
    // =========================================================================
    //  Shape Kind Enum — compile-time shape category tag
    // =========================================================================

    enum class ShapeKind : uint8_t
    {
        Sphere,
        Box,
        Capsule,
        Cylinder,
        Cone,
        TaperedCapsule,
        ConvexHull,
        Mesh,
        Compound
    };


    // =========================================================================
    //  consteval shape property queries
    // =========================================================================

    consteval bool shape_kind_is_convex(ShapeKind k)
    {
        return k != ShapeKind::Mesh && k != ShapeKind::Compound;
    }

    consteval bool shape_kind_supports_exact_culling(ShapeKind k)
    {
        return k != ShapeKind::Compound && k != ShapeKind::Mesh;
    }

    consteval bool shape_kind_is_symmetric(ShapeKind k)
    {
        return k == ShapeKind::Sphere;
    }

    consteval bool shape_kind_needs_orientation(ShapeKind k)
    {
        return k != ShapeKind::Sphere;
    }

    consteval bool shape_kind_has_support_function(ShapeKind k)
    {
        return shape_kind_is_convex(k);
    }


    // =========================================================================
    //  C++20 Concepts
    // =========================================================================

    /// Anything that can produce a Jolt shape reference.
    template<typename T>
    concept JoltShapeSource = requires(const T& src) {
        { src.jolt_shape() } -> std::convertible_to<JPH::ShapeRefC>;
    };

    /// Anything that has a conservative bounding sphere (SHS LH space).
    template<typename T>
    concept HasBoundingSphere = requires(const T& src) {
        { src.bounding_sphere() } -> std::convertible_to<Sphere>;
    };

    /// Anything that has a world-space AABB (SHS LH space).
    template<typename T>
    concept HasWorldAABB = requires(const T& src) {
        { src.world_aabb() } -> std::convertible_to<AABB>;
    };

    /// Anything cullable: must have a Jolt shape + world transform.
    template<typename T>
    concept Cullable = requires(const T& obj) {
        { obj.jolt_shape() }       -> std::convertible_to<JPH::ShapeRefC>;
        { obj.world_transform() }  -> std::convertible_to<JPH::Mat44>;
    };

    /// Cullable object that also provides a fast bounding sphere.
    template<typename T>
    concept FastCullable = Cullable<T> && HasBoundingSphere<T>;
}

#endif // SHS_HAS_JOLT
