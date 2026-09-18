#pragma once

/*
    tier0 common — execution-neutral fixed-function pass policy (AD2).

    One vocabulary for the fixed-function state both twins need. The *_sw
    rasterizer and the *_vk pipeline builder consume THIS type; neither keeps a
    parallel semantic copy of the other's state. Before AD2 the same four
    concepts existed twice (SwState and the depth/blend/stencil fields of
    VkPipelineSetup) and could drift apart silently.

    In scope: depth test, depth write, straight-alpha blending, stencil mode +
    reference, scissor. Out of scope on purpose: shader paths, vertex layout,
    Vulkan handles, and framebuffer/depth/stencil STORAGE — the executors own
    their own storage (the software rasterizer its z/stencil vectors, the Vulkan
    harness its attachments) and mutate it only inside a draw.

    Defaults (documented, never implicit): depth test ON with a LESS comparison,
    depth write ON, blending OFF, stencil Disabled (buffer untouched), no
    scissor (full target). Every command states the policy it means, so state
    from one draw cannot leak into the next.

    Capability differences between the twins — kept visible rather than papered
    over by a looser abstraction:
      * Scissor. The software rasterizer applies the policy per draw, so one
        frame may mix scissored and unscissored draws. Vulkan applies a single
        dynamic scissor (VK_DYNAMIC_STATE_SCISSOR) per render() call, so the
        *_vk twin issues one band for the whole pass; the twins agree for the
        band-per-pass use every current demo has.
      * Stencil facing. Vulkan configures front and back faces identically; the
        software rasterizer has a single stencil value per pixel and no facing.
      * Stencil mode is a closed set. "invert" without "test" used to be two
        independent bools in both backends, and both silently ignored it. That
        combination is no longer representable.
      * Blending. Both use SRC_ALPHA / ONE_MINUS_SRC_ALPHA source-over against a
        float target; the software path computes it in float and quantizes once
        at the store.
      * Culling is deliberately not part of this policy: both twins rasterize
        both orientations (curriculum demos draw back faces on purpose).
*/

#include <cstdint>

namespace adventures
{
    // Closed stencil vocabulary. WriteRef is the "ALWAYS + REPLACE" pair the
    // text used to spell as two bools; the two test modes are EQUAL/NOT_EQUAL
    // against stencil_ref with KEEP on failure.
    enum class StencilMode : uint8_t
    {
        Disabled,     // buffer untouched: no test, no write
        WriteRef,     // covered pixels store ref
        TestEqual,    // keep pixels whose stencil == ref
        TestNotEqual, // keep pixels whose stencil != ref
    };

    inline bool stencil_enabled(StencilMode m) { return m != StencilMode::Disabled; }
    inline bool stencil_writes(StencilMode m) { return m == StencilMode::WriteRef; }
    inline bool stencil_tests(StencilMode m)
    {
        return m == StencilMode::TestEqual || m == StencilMode::TestNotEqual;
    }
    inline bool stencil_inverts(StencilMode m) { return m == StencilMode::TestNotEqual; }

    // Half-open rectangle, x1/y1 exclusive, in framebuffer pixels, origin
    // top-left (the same orientation as Frame and the Vulkan scissor).
    struct ScissorRect
    {
        int x0 = 0;
        int y0 = 0;
        int x1 = 0;
        int y1 = 0;

        bool empty() const { return x1 <= x0 || y1 <= y0; }
    };

    struct PassPolicy
    {
        bool        depth_test     = true;
        bool        depth_write    = true;
        bool        blend          = false;
        StencilMode stencil        = StencilMode::Disabled;
        uint8_t     stencil_ref    = 1;
        bool        scissor_enabled = false;
        ScissorRect scissor{};
    };

    // Pixel-space scissor test. Disabled scissor allows everything, and an
    // enabled but empty rectangle allows nothing (a band that clips away is a
    // statement, not an error).
    inline bool scissor_allows(const PassPolicy& policy, int x, int y)
    {
        if (!policy.scissor_enabled) return true;
        return x >= policy.scissor.x0 && x < policy.scissor.x1 &&
               y >= policy.scissor.y0 && y < policy.scissor.y1;
    }

    // Intersect a policy scissor with a w x h target. Vulkan requires the
    // dynamic scissor to lie inside the framebuffer, so the *_vk adapter clamps
    // through here; the software rasterizer needs no clamp because it tests
    // scissor_allows() per pixel. An empty result means "draw nothing", the
    // same thing an empty policy rectangle means on the software side.
    inline ScissorRect clamp_scissor(const ScissorRect& r, int w, int h)
    {
        ScissorRect out{};
        out.x0 = r.x0 < 0 ? 0 : (r.x0 > w ? w : r.x0);
        out.y0 = r.y0 < 0 ? 0 : (r.y0 > h ? h : r.y0);
        out.x1 = r.x1 < out.x0 ? out.x0 : (r.x1 > w ? w : r.x1);
        out.y1 = r.y1 < out.y0 ? out.y0 : (r.y1 > h ? h : r.y1);
        return out;
    }
}
