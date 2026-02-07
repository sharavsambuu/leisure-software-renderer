#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sync_desc.hpp
    МОДУЛЬ: rhi/sync
    ЗОРИЛГО: Fence/Semaphore/barrier descriptor-ууд.
            Vulkan synchronization загварыг backend-neutral түвшинд илэрхийлнэ.
*/


#include <cstdint>

namespace shs
{
    enum class RHIPipelineStage : uint8_t
    {
        Top = 0,
        DrawIndirect = 1,
        VertexInput = 2,
        VertexShader = 3,
        FragmentShader = 4,
        ColorOutput = 5,
        ComputeShader = 6,
        Transfer = 7,
        Bottom = 8
    };

    struct RHISemaphoreSignalDesc
    {
        uint64_t semaphore_id = 0;
        uint64_t value = 0;
        RHIPipelineStage stage = RHIPipelineStage::Bottom;
    };

    struct RHISemaphoreWaitDesc
    {
        uint64_t semaphore_id = 0;
        uint64_t value = 0;
        RHIPipelineStage stage = RHIPipelineStage::Top;
    };

    struct RHIFenceDesc
    {
        uint64_t fence_id = 0;
        bool signaled = false;
    };

    enum class RHIAccess : uint16_t
    {
        None = 0,
        Read = 1,
        Write = 2,
        ReadWrite = 3
    };

    struct RHIMemoryBarrierDesc
    {
        RHIPipelineStage src_stage = RHIPipelineStage::Top;
        RHIPipelineStage dst_stage = RHIPipelineStage::Bottom;
        RHIAccess src_access = RHIAccess::None;
        RHIAccess dst_access = RHIAccess::None;
    };
}

