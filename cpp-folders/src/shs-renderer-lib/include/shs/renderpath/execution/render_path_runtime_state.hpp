#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_runtime_state.hpp
    MODULE: pipeline
    PURPOSE: Runtime toggles/state contract used by dynamic render path recipes.
*/


namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    struct RenderPathRuntimeState
    {
        bool view_occlusion_enabled = true;
        bool shadow_occlusion_enabled = false;
        bool debug_aabb = false;
        bool lit_mode = true;
        bool enable_shadows = true;

        // Value semantics (pod test kit requires snapshot equality).
        bool operator==(const RenderPathRuntimeState&) const = default;

        void reset_defaults()
        {
            *this = RenderPathRuntimeState{};
        }
    };

    } // inline namespace renderpath
}

