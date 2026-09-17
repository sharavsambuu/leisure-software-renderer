#pragma once

/*
    SHS RENDERER SAN

    FILE: gfx.contract.hpp
    MODULE: domains/gfx
    PURPOSE: CORE 1. TYPES — the gfx pod's discoverability seam (R5b P3.8).
             Handle + pixel-buffer values are the pod spine (plain data, the
             currency between passes and backends). RTRegistry (unordered_map
             keyed store, method-mutated) is an edge candidate — visible but
             migrating with the other registries in the R5b convergence.
*/

#include "shs/render/targets/resource_handles.hpp"
#include "shs/render/targets/rt_handle.hpp"
#include "shs/render/targets/storage/rt_registry.hpp"
#include "shs/render/targets/rt_shadow.hpp"
#include "shs/render/targets/rt_types.hpp"

namespace shs::gfx
{
    // --- handle values (pod spine) ---
    using shs::RTHandle;
    using shs::RT_Color;
    using shs::RT_Depth;
    using shs::RT_Motion;
    using shs::RT_Shadow;

    // --- pixel values ---
    using shs::Color;
    using shs::ColorF;
    using shs::Motion2f;
    using shs::PixelBuffer2D;
    using shs::RT_ColorLDR;
    using shs::RT_ColorHDR;
    using shs::RT_DepthBuffer;
    using shs::RT_ColorDepth;
    using shs::RT_ColorDepthVelocity;

    // --- store class (visible, migrating in R5b convergence) ---
    using shs::RTRegistry;
} // namespace shs::gfx
