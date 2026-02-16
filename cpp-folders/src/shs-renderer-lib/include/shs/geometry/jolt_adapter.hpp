#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_adapter.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Jolt Physics номын сангийн интеграцийн суурь давхарга.
            SHS (LH, +Z forward) ↔ Jolt (RH, -Z forward) хоорондын
            бүх coordinate system conversion-ийг нэг цэгт нэгтгэнэ.

    CONVENTION:
        SHS:  Left-handed, Y-up, +Z = forward
        Jolt: Right-handed, Y-up, -Z = forward
        Conversion: negate Z for positions/directions.
                    negate X,Y for quaternions.
                    conjugate matrices by S = diag(1,1,-1,1).
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <Jolt/Jolt.h>
#include <Jolt/RegisterTypes.h>
#include <Jolt/Core/Factory.h>
#include <Jolt/Core/TempAllocator.h>
#include <Jolt/Core/JobSystemThreadPool.h>
#include <Jolt/Physics/PhysicsSettings.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Math/Vec3.h>
#include <Jolt/Math/Vec4.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Math/Quat.h>
#include <Jolt/Geometry/AABox.h>
#include <Jolt/Physics/Collision/Shape/Shape.h>

#include "shs/core/units.hpp"
#include "shs/geometry/aabb.hpp"
#include "shs/geometry/volumes.hpp"

namespace shs::jolt
{
    // =========================================================================
    //  Unit mapping
    //  SHS and Jolt are both configured to use SI-like units.
    //  Distance scaling is intentionally 1:1.
    // =========================================================================

    inline constexpr float kDistanceScaleShsToJolt = 1.0f;
    inline constexpr float kDistanceScaleJoltToShs = 1.0f;

    inline constexpr float to_jph_distance(float shs_distance_meters) noexcept
    {
        return shs_distance_meters * kDistanceScaleShsToJolt;
    }

    inline constexpr float to_shs_distance(float jolt_distance_units) noexcept
    {
        return jolt_distance_units * kDistanceScaleJoltToShs;
    }

    inline constexpr float to_jph_mass(float shs_mass_kg) noexcept
    {
        return shs_mass_kg;
    }

    inline constexpr float to_shs_mass(float jolt_mass_units) noexcept
    {
        return jolt_mass_units;
    }

    // =========================================================================
    //  Position / Direction conversion  (Z-negate)
    // =========================================================================

    inline JPH::Vec3 to_jph(const glm::vec3& v) noexcept
    {
        return JPH::Vec3(
            to_jph_distance(v.x),
            to_jph_distance(v.y),
            -to_jph_distance(v.z));
    }

    inline glm::vec3 to_glm(const JPH::Vec3& v) noexcept
    {
        return glm::vec3(
            to_shs_distance(v.GetX()),
            to_shs_distance(v.GetY()),
            -to_shs_distance(v.GetZ()));
    }


    // =========================================================================
    //  Quaternion conversion  (negate X,Y = Z-flip conjugation)
    // =========================================================================

    inline JPH::Quat to_jph(const glm::quat& q) noexcept
    {
        return JPH::Quat(-q.x, -q.y, q.z, q.w);
    }

    inline glm::quat to_glm(const JPH::Quat& q) noexcept
    {
        return glm::quat(q.GetW(), -q.GetX(), -q.GetY(), q.GetZ());
    }


    // =========================================================================
    //  4×4 Matrix conversion:  M_jolt = S · M_shs · S
    //  where S = diag(1, 1, -1, 1).  S is its own inverse.
    //  This negates row 2 and column 2 of the 4×4 matrix.
    // =========================================================================

    inline JPH::Mat44 to_jph(const glm::mat4& m) noexcept
    {
        // GLM is column-major: m[col][row]
        // After S·M·S conjugation:
        //   columns 0,1,3: negate row 2  (the z component)
        //   column 2: negate rows 0,1,3  (keep row 2)

        return JPH::Mat44(
            JPH::Vec4( m[0][0],  m[0][1], -m[0][2],  m[0][3]),   // col 0
            JPH::Vec4( m[1][0],  m[1][1], -m[1][2],  m[1][3]),   // col 1
            JPH::Vec4(-m[2][0], -m[2][1],  m[2][2], -m[2][3]),   // col 2
            JPH::Vec4( m[3][0],  m[3][1], -m[3][2],  m[3][3])    // col 3
        );
    }

    inline glm::mat4 to_glm(const JPH::Mat44& m) noexcept
    {
        // Reverse is identical: S·M·S with S = diag(1,1,-1,1)
        const JPH::Vec4 c0 = m.GetColumn4(0);
        const JPH::Vec4 c1 = m.GetColumn4(1);
        const JPH::Vec4 c2 = m.GetColumn4(2);
        const JPH::Vec4 c3 = m.GetColumn4(3);

        glm::mat4 out{};
        out[0] = glm::vec4( c0.GetX(),  c0.GetY(), -c0.GetZ(),  c0.GetW());
        out[1] = glm::vec4( c1.GetX(),  c1.GetY(), -c1.GetZ(),  c1.GetW());
        out[2] = glm::vec4(-c2.GetX(), -c2.GetY(),  c2.GetZ(), -c2.GetW());
        out[3] = glm::vec4( c3.GetX(),  c3.GetY(), -c3.GetZ(),  c3.GetW());
        return out;
    }


    // =========================================================================
    //  Plane conversion  (negate normal Z, keep distance)
    // =========================================================================

    inline Plane to_shs_plane(const JPH::Plane& p) noexcept
    {
        const JPH::Vec3 n = p.GetNormal();
        return Plane{
            glm::vec3(n.GetX(), n.GetY(), -n.GetZ()),
            to_shs_distance(p.GetConstant())
        };
    }

    inline JPH::Plane to_jph_plane(const Plane& p) noexcept
    {
        return JPH::Plane(
            JPH::Vec3(p.normal.x, p.normal.y, -p.normal.z),
            to_jph_distance(p.d));
    }


    // =========================================================================
    //  AABB conversion  (Z min/max swap after negation)
    // =========================================================================

    inline JPH::AABox to_jph(const AABB& b) noexcept
    {
        return JPH::AABox(
            JPH::Vec3(
                to_jph_distance(b.minv.x),
                to_jph_distance(b.minv.y),
                -to_jph_distance(b.maxv.z)),
            JPH::Vec3(
                to_jph_distance(b.maxv.x),
                to_jph_distance(b.maxv.y),
                -to_jph_distance(b.minv.z))
        );
    }

    inline AABB to_glm(const JPH::AABox& b) noexcept
    {
        AABB out{};
        out.minv = glm::vec3(
            to_shs_distance(b.mMin.GetX()),
            to_shs_distance(b.mMin.GetY()),
            -to_shs_distance(b.mMax.GetZ()));
        out.maxv = glm::vec3(
            to_shs_distance(b.mMax.GetX()),
            to_shs_distance(b.mMax.GetY()),
            -to_shs_distance(b.mMin.GetZ()));
        return out;
    }


    // =========================================================================
    //  Sphere conversion  (center Z-negate, radius unchanged)
    // =========================================================================

    inline JPH::Vec3 sphere_center_to_jph(const Sphere& s) noexcept
    {
        return to_jph(s.center);
    }

    inline Sphere to_shs_sphere(const JPH::Vec3& center_jph, float radius) noexcept
    {
        return Sphere{to_glm(center_jph), to_shs_distance(radius)};
    }


    // =========================================================================
    //  Init / Shutdown
    // =========================================================================

    inline bool jolt_initialized() noexcept
    {
        return JPH::Factory::sInstance != nullptr;
    }

    inline void init_jolt()
    {
        if (jolt_initialized()) return;

        JPH::RegisterDefaultAllocator();
        JPH::Factory::sInstance = new JPH::Factory();
        JPH::RegisterTypes();
    }

    inline void shutdown_jolt()
    {
        if (!jolt_initialized()) return;

        JPH::UnregisterTypes();
        delete JPH::Factory::sInstance;
        JPH::Factory::sInstance = nullptr;
    }
}

#else // !SHS_HAS_JOLT

namespace shs::jolt
{
    inline void init_jolt() {}
    inline void shutdown_jolt() {}
    inline bool jolt_initialized() noexcept { return false; }
}

#endif // SHS_HAS_JOLT
