#pragma once

#include <shs/core/context.hpp>
#include <shs/passes/rt_types.hpp>

namespace shs
{
    struct PassContext
    {
        Context*  ctx = nullptr;
        DefaultRT* rt  = nullptr;
        int frame_index = 0;
        float dt = 0.0f;
    };
}