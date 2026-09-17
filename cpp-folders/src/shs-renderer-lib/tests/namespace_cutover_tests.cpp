// Step-7 namespace cutover compatibility tests (step 7 item 2 of
// engine_domain_separation_migration.md).
//
// The cutover moved root `shs::` public symbols into owner namespaces behind
// `inline namespace` wrappers (and, for `shs::app` which pre-existed as a
// non-inline namespace, behind root using-declarations). These tests pin the
// compatibility contract:
//   1. old (`shs::X`) and new (`shs::<owner>::X`) spellings denote the SAME
//      entity (types, aliases, functions) — overload lookup is unchanged;
//   2. ADL keeps finding owner-namespace free functions through arguments
//      spelled either way;
//   3. serialization identifiers (stable enum ids + string mappings used by
//      persisted configs) are unchanged by the namespace move;
//   4. legacy compatibility includes (`shs/domains/...`) and canonical owner
//      headers can be mixed in one translation unit in either order.
//
// Namespace changes can break ABI even when names resolve: aliases do not
// guarantee binary compatibility (see the step-7 checklist).

#include "shs/domains/geometry/aabb.hpp" // legacy forwarder include
#include "shs/geometry/aabb.hpp"         // canonical owner include (same TU)

#include "shs/app/backend/backend_factory.hpp"
#include "shs/app/context.hpp"
#include "shs/app/session_orchestrator.hpp"
#include "shs/camera/camera_rig.hpp"
#include "shs/render/frame/backend_type.hpp"
#include "shs/rhi/core/backend.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <type_traits>

#include <glm/vec3.hpp>

// ---------------------------------------------------------------------------
// 1. Same-entity checks: old root spellings vs owner-namespace spellings.
// ---------------------------------------------------------------------------
static_assert(std::is_same_v<shs::AABB, shs::geometry::AABB>,
    "root AABB spelling must denote shs::geometry::AABB");
static_assert(std::is_same_v<shs::RenderBackendType, shs::render::RenderBackendType>,
    "root RenderBackendType spelling must denote shs::render::RenderBackendType");
static_assert(std::is_same_v<shs::IRenderBackend, shs::rhi::IRenderBackend>,
    "root IRenderBackend spelling must denote shs::rhi::IRenderBackend");
static_assert(std::is_same_v<shs::Context, shs::app::Context>,
    "root Context spelling must denote the app-owned shs::app::Context");
static_assert(std::is_same_v<shs::RuntimeState, shs::app::SessionState>,
    "root RuntimeState alias must denote shs::app::SessionState");
static_assert(std::is_same_v<shs::app::RuntimeState, shs::app::SessionState>,
    "app-namespaced RuntimeState alias must denote shs::app::SessionState");
static_assert(std::is_same_v<shs::CameraRig, shs::camera::CameraRig>,
    "root CameraRig spelling must denote shs::camera::CameraRig");

namespace cutover_adl_probe
{
    // 2. ADL: unqualified call from a foreign namespace. The argument type
    // shs::render::RenderBackendType must pull in the owner-namespace free
    // function without any using-directive or qualification.
    const char* backend_name(shs::render::RenderBackendType type)
    {
        return render_backend_type_name(type);
    }
}

int main()
{
    bool ok = true;

    // -----------------------------------------------------------------
    // 3. Overload lookup through old and new spellings picks the same
    //    overloads and returns equivalent results.
    // -----------------------------------------------------------------
    {
        // Old root spelling, enum overload.
        const auto via_root_enum = shs::create_render_backend(shs::RenderBackendType::Software);
        // New owner spelling, string overload (exercises the to_lower_ascii
        // parse path inside the same owner namespace).
        const auto via_app_text = shs::app::create_render_backend("SOFTWARE");
        // Mixed: new spelling function, old spelling argument type.
        const auto mixed = shs::app::create_render_backend(shs::RenderBackendType::Software);

        ok = via_root_enum.backend != nullptr &&
             via_app_text.backend != nullptr &&
             mixed.backend != nullptr &&
             via_root_enum.active == shs::render::RenderBackendType::Software &&
             via_app_text.active == shs::render::RenderBackendType::Software &&
             mixed.active == shs::render::RenderBackendType::Software &&
             via_root_enum.backend->type() == shs::RenderBackendType::Software &&
             via_app_text.backend->type() == shs::render::RenderBackendType::Software &&
             ok;

        // The backend factory reports the requested id unchanged across
        // spellings (serialization identifier stability, see below).
        ok = via_root_enum.requested == shs::render::RenderBackendType::Software &&
             via_app_text.requested == shs::RenderBackendType::Software &&
             ok;
    }

    // -----------------------------------------------------------------
    // 4. Serialization identifiers: stable enum ids and the persisted
    //    string<->id mapping must be unchanged by the namespace move.
    // -----------------------------------------------------------------
    {
        static_assert(std::is_same_v<std::underlying_type_t<shs::RenderBackendType>, std::uint8_t>,
            "RenderBackendType underlying type must stay uint8_t");
        static_assert(static_cast<std::uint8_t>(shs::RenderBackendType::Software) == 0,
            "Software id drifted");
        static_assert(static_cast<std::uint8_t>(shs::RenderBackendType::OpenGL) == 1,
            "OpenGL id drifted");
        static_assert(static_cast<std::uint8_t>(shs::RenderBackendType::Vulkan) == 2,
            "Vulkan id drifted");

        const shs::RenderBackendType ids[] = {
            shs::RenderBackendType::Software,
            shs::RenderBackendType::OpenGL,
            shs::RenderBackendType::Vulkan,
        };
        const char* const names[] = {"software", "opengl", "vulkan"};
        for (int i = 0; i < 3; ++i)
        {
            // Old spelling for the name mapping, new spelling for the parse.
            const char* named = shs::render_backend_type_name(ids[i]);
            const auto parsed = shs::app::parse_render_backend_type(
                names[i], shs::render::RenderBackendType::Software);
            ok = std::string(named) == names[i] && ok;
            ok = static_cast<std::uint8_t>(parsed) == static_cast<std::uint8_t>(ids[i]) && ok;
        }
    }

    // -----------------------------------------------------------------
    // 5. ADL through a foreign namespace (probe above).
    // -----------------------------------------------------------------
    ok = std::string(cutover_adl_probe::backend_name(shs::RenderBackendType::Vulkan)) == "vulkan" && ok;
    ok = std::string(cutover_adl_probe::backend_name(shs::render::RenderBackendType::OpenGL)) == "opengl" && ok;

    // -----------------------------------------------------------------
    // 6. Mixed old/new spellings compose in a single expression.
    // -----------------------------------------------------------------
    {
        shs::camera::CameraRig rig{};
        shs::CameraRig& same_rig = rig; // old spelling binds to the same type
        static_cast<void>(same_rig);

        shs::RuntimeState state{};
        shs::app::SessionState& same_state = state;
        static_cast<void>(same_state);

        shs::geometry::AABB box{};
        box.expand(glm::vec3(1.0f, 2.0f, 3.0f));
        shs::AABB& same_box = box;
        ok = same_box.center().x == 1.0f && same_box.center().y == 2.0f && ok;
    }

    std::fprintf(stderr, "[namespace-cutover] %s\n", ok ? "pass" : "FAIL");
    return ok ? 0 : 1;
}
