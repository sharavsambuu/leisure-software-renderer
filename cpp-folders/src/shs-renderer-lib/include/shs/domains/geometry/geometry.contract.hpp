#pragma once

/*
    SHS RENDERER SAN

    FILE: geometry.contract.hpp
    MODULE: domains/geometry
    PURPOSE: CORE 1. TYPES — the geometry pod's discoverability seam (R4 P3.4).
             Shapes + value adapters are the pod spine; culling runtimes stay
             out until their own migration (R5: they are stateful execution
             candidates, not shape values). The rung-08 TBN operator rides here.
*/

#include "shs/domains/geometry/aabb.hpp"
#include "shs/domains/geometry/primitives.hpp"
#include "shs/domains/geometry/tangent_frame.hpp"
#include "shs/domains/geometry/volumes.hpp"

namespace shs::geometry
{
    // --- shape descriptors ---
    using shs::PlaneDesc;
    using shs::SphereDesc;
    using shs::BoxDesc;
    using shs::ConeDesc;

    // --- shape values ---
    using shs::AABB;
    using shs::Point3;
    using shs::LineSegment3;
    using shs::Ray3;
    using shs::Plane;
    using shs::Sphere;
    using shs::OBB;
    using shs::Capsule;
    using shs::Cone;
    using shs::Cylinder;
    using shs::Frustum;

    // --- rung-08 operator ---
    using shs::TangentFrame;
} // namespace shs::geometry
