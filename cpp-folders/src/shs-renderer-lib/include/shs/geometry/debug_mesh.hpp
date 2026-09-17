#pragma once

/*
    SHS RENDERER SAN

    FILE: debug_mesh.hpp
    MODULE: geometry (value tier)
    PURPOSE: Simple indexed triangle mesh value for visualization/debug draw.

    Extraction note (R5c debug-draw seam ruling, 2026-09-17): DebugMesh was
    originally defined inside the Jolt debug-draw adapter
    (geometry/adapters/jolt/jolt_debug_draw.hpp). It is a pure value type
    (GLM-only, SDK-free), so it is owned here; Jolt-backed conversion
    functions in the adapter *produce* it, and the software debug-draw path
    (render/software/debug_draw.hpp) *consumes* it — render/software no
    longer reaches adapter code at all (include-graph gate R3 clean).
*/

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace geometry
    {
    struct DebugMesh
    {
        std::vector<glm::vec3> vertices{};
        std::vector<uint32_t>  indices{};

        void clear()
        {
            vertices.clear();
            indices.clear();
        }

        bool empty() const noexcept
        {
            return vertices.empty();
        }
    };

    } // inline namespace geometry
} // namespace shs
