#pragma once

/*
    SHS RENDERER SAN

    FILE: lighting.contract.hpp
    MODULE: domains/lighting
    PURPOSE: CORE 1. TYPES — the lighting pod's discoverability seam (R4 P3.5).
             Light vocabulary + sets are the pod spine; culling runtimes stay
             out until their own migration (R5). The rung-08 shading terms ride here.
*/

#include "shs/lighting/light_culling_mode.hpp"
#include "shs/lighting/light_set.hpp"
#include "shs/lighting/light_types.hpp"
#include "shs/lighting/shading_terms.hpp"

namespace shs::lighting
{
    // --- light vocabulary ---
    using shs::LightType;
    using shs::LightCullingShape;
    using shs::LightAttenuationModel;
    using shs::LocalLightCommon;
    using shs::PointLight;
    using shs::SpotLight;
    using shs::RectAreaLight;
    using shs::TubeAreaLight;
    using shs::CullingLightGPU;

    // --- sets + modes ---
    using shs::LightSet;
    using shs::LightCullingMode;
} // namespace shs::lighting
