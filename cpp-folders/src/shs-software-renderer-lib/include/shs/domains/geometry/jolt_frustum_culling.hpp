#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: jolt_frustum_culling.hpp
    МОДУЛЬ: geometry
    ЗОРИЛГО: Camera frustum-д суурилсан scene object culling.
            extract_frustum_planes()-ийг дахин ашиглаж,
            Cullable/FastCullable concept-ээр batch frustum cull хийнэ.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <span>

#include <glm/glm.hpp>

#include "shs/geometry/volumes.hpp"
#include "shs/geometry/frustum_culling.hpp"
#include "shs/geometry/jolt_culling.hpp"

namespace shs
{
    // =========================================================================
    //  Batch frustum cull for scene objects
    //  Extracts frustum from view-projection, then uses generic cull_vs_frustum.
    // =========================================================================

    template<FastCullable T>
    inline CullResult frustum_cull_scene(
        std::span<const T> objects,
        const glm::mat4& view_proj,
        const CullTolerance& tol = {})
    {
        const Frustum frustum = extract_frustum_planes(view_proj);
        return cull_vs_frustum(objects, frustum, tol);
    }


    // =========================================================================
    //  Single object frustum test
    // =========================================================================

    template<FastCullable T>
    inline bool is_visible_in_frustum(
        const T& obj,
        const glm::mat4& view_proj,
        const CullTolerance& tol = {})
    {
        const Frustum frustum = extract_frustum_planes(view_proj);
        const CullClass c = classify_vs_frustum(obj, frustum, tol);
        return c != CullClass::Outside;
    }
}

#endif // SHS_HAS_JOLT
