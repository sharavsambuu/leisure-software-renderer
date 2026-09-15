#pragma once

/*
    SHS RENDERER SAN

    FILE: value_input_latch.hpp
    MODULE: input
    PURPOSE: Value-oriented reducer for runtime input latch state.
*/


#include <span>
#include <vector>

namespace shs
{
    struct RuntimeInputLatch
    {
        bool forward = false;
        bool backward = false;
        bool left = false;
        bool right = false;
        bool ascend = false;
        bool descend = false;
        bool boost = false;
        bool left_mouse_down = false;
        bool right_mouse_down = false;
        float mouse_dx_accum = 0.0f;
        float mouse_dy_accum = 0.0f;
        bool quit_requested = false;
    };

    enum class RuntimeInputEventType : unsigned char
    {
        SetForward = 0,
        SetBackward = 1,
        SetLeft = 2,
        SetRight = 3,
        SetAscend = 4,
        SetDescend = 5,
        SetBoost = 6,
        SetLeftMouseDown = 7,
        SetRightMouseDown = 8,
        AddMouseDelta = 9,
        RequestQuit = 10
    };

    struct RuntimeInputEvent
    {
        RuntimeInputEventType type = RuntimeInputEventType::SetForward;
        bool bool_value = false;
        float x = 0.0f;
        float y = 0.0f;
    };

    inline RuntimeInputEvent make_bool_input_event(RuntimeInputEventType type, bool value)
    {
        RuntimeInputEvent out{};
        out.type = type;
        out.bool_value = value;
        return out;
    }

    inline RuntimeInputEvent make_mouse_delta_input_event(float dx, float dy)
    {
        RuntimeInputEvent out{};
        out.type = RuntimeInputEventType::AddMouseDelta;
        out.x = dx;
        out.y = dy;
        return out;
    }

    inline RuntimeInputEvent make_quit_input_event()
    {
        RuntimeInputEvent out{};
        out.type = RuntimeInputEventType::RequestQuit;
        return out;
    }

    inline RuntimeInputLatch reduce_runtime_input_latch(
        RuntimeInputLatch state,
        std::span<const RuntimeInputEvent> events)
    {
        for (const RuntimeInputEvent& e : events)
        {
            switch (e.type)
            {
                case RuntimeInputEventType::SetForward:
                    state.forward = e.bool_value;
                    break;
                case RuntimeInputEventType::SetBackward:
                    state.backward = e.bool_value;
                    break;
                case RuntimeInputEventType::SetLeft:
                    state.left = e.bool_value;
                    break;
                case RuntimeInputEventType::SetRight:
                    state.right = e.bool_value;
                    break;
                case RuntimeInputEventType::SetAscend:
                    state.ascend = e.bool_value;
                    break;
                case RuntimeInputEventType::SetDescend:
                    state.descend = e.bool_value;
                    break;
                case RuntimeInputEventType::SetBoost:
                    state.boost = e.bool_value;
                    break;
                case RuntimeInputEventType::SetLeftMouseDown:
                    state.left_mouse_down = e.bool_value;
                    break;
                case RuntimeInputEventType::SetRightMouseDown:
                    state.right_mouse_down = e.bool_value;
                    break;
                case RuntimeInputEventType::AddMouseDelta:
                    state.mouse_dx_accum += e.x;
                    state.mouse_dy_accum += e.y;
                    break;
                case RuntimeInputEventType::RequestQuit:
                    state.quit_requested = true;
                    break;
            }
        }
        return state;
    }

    inline RuntimeInputLatch clear_runtime_input_frame_deltas(RuntimeInputLatch state)
    {
        state.mouse_dx_accum = 0.0f;
        state.mouse_dy_accum = 0.0f;
        return state;
    }

    inline void append_runtime_input_event(
        std::vector<RuntimeInputEvent>& out,
        RuntimeInputEventType type,
        bool value)
    {
        out.push_back(make_bool_input_event(type, value));
    }
}
