#pragma once

/*
    SHS RENDERER SAN

    FILE: frame.contract.hpp
    MODULE: domains/frame
    PURPOSE: CORE 1. TYPES — the frame pod's discoverability seam (R3).
             FrameParams and friends are pure configuration values rebuilt per
             frame by planners; this contract names them in one place.
*/

#include "shs/render/frame/frame_params.hpp"
#include "shs/render/frame/technique_mode.hpp"

namespace shs::frame
{
    using shs::FrameParams;
    using shs::PassParamBlocks;
    using shs::TonemapParams;
    using shs::ShadowPassParams;
    using shs::LightShaftsPassParams;
    using shs::MotionVectorParams;
    using shs::MotionBlurPassParams;
    using shs::HybridPipelineParams;
    using shs::TechniqueParams;
    using shs::TechniqueMode;
    using shs::DebugViewMode;
    using shs::CullMode;
    using shs::ShadingModel;
} // namespace shs::frame
