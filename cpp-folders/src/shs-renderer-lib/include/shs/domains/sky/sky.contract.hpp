#pragma once

/*
    SHS RENDERER SAN

    FILE: sky.contract.hpp
    MODULE: domains/sky
    PURPOSE: CORE 1. TYPES — the sky pod's discoverability seam (R5a P3.9).
             Sky MODELS are values (procedural params, cubemap data); the
             ISkyModel virtual interface is flagged debt — virtual dispatch in
             the per-pixel sampling path violates the hot-path law (Const II
             §4.4); R5b replaces it with a closed variant or concept +
             free-function sample(). Kept visible here until then.
*/

#include "shs/domains/sky/cubemap_sky.hpp"
#include "shs/domains/sky/procedural_sky.hpp"
#include "shs/domains/sky/sky_model.hpp"

namespace shs::sky
{
    using shs::ProceduralSky;
    using shs::CubemapData;

    // Edge-debt interface (visible, migrating in R5b — see above).
    using shs::ISkyModel;
} // namespace shs::sky
