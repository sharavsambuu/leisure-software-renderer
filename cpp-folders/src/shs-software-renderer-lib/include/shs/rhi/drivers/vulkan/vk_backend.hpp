#pragma once

/*
    SHS RENDERER SAN

    FILE: vk_backend.hpp (facade shim)
    MODULE: rhi/drivers/vulkan
    PURPOSE: Canonical path forward for the aspirational backend_factory
            include — resolves to the P2 value-desc driver pod. This is a
            stable re-export (not a deprecation shim): the pod-first driver
            lives under shs/execution/rhi/drivers/vulkan/.
*/

#include "shs/execution/rhi/drivers/vulkan/vk_backend.hpp"
