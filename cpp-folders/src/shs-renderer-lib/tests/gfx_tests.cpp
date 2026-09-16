#include <cstdio>
#include <memory_resource>
#include <variant>
#include <vector>

#include "shs/domains/gfx/gfx.contract.hpp"
#include "shs/domains/gfx/gfx.gateway.hpp"
#include "shs/domains/pod_test_kit.hpp"

// Headless tests for the gfx pod (R5b P3.8: handle/buffer pins + identity).
// Links only shs::renderer-values + glm. Registry allocation is exercised
// only via its current API as migration spec (edge-candidate until R5b).
namespace
{
    auto run_gateway = [](shs::gfx::GfxState& s,
                             std::span<const shs::gfx::GfxCommand> a,
                             const shs::gfx::GfxContext& in,
                             std::pmr::vector<shs::gfx::GfxEvent>& e)
    {
        shs::gfx::gfx_gateway(s, a, in, e);
    };

    // Null handle is invalid; any nonzero id is valid (generation-free).
    bool test_handle_validity()
    {
        const shs::RTHandle null{};
        if (null.valid()) return false;
        shs::RTHandle h{};
        h.id = 41;
        if (!h.valid()) return false;
        return true;
    }

    // Pixel buffers clear uniformly and address row-major.
    bool test_pixel_buffer()
    {
        shs::PixelBuffer2D<shs::Color> buf(4, 2, shs::Color{9, 8, 7, 255});
        if (buf.w != 4 || buf.h != 2) return false;
        if (buf.at(3, 1).r != 9 || buf.at(0, 0).b != 7) return false;
        buf.at(1, 0).g = 200;
        return buf.at(1, 0).g == 200 && buf.at(2, 0).g == 8;
    }

    bool test_identity_stable()
    {
        return shs::pod_test::empty_log_is_stable<shs::gfx::GfxState,
            shs::gfx::GfxCommand, shs::gfx::GfxContext,
            shs::gfx::GfxEvent>(
            run_gateway, shs::gfx::GfxState{}, shs::gfx::GfxContext{});
    }

    bool test_replay_deterministic()
    {
        const shs::gfx::GfxState s0{};
        const std::vector<shs::gfx::GfxCommand> none{};
        return shs::pod_test::replay_is_deterministic<shs::gfx::GfxState,
            shs::gfx::GfxCommand, shs::gfx::GfxContext,
            shs::gfx::GfxEvent>(
            run_gateway, s0,
            std::span<const shs::gfx::GfxCommand>{none.data(), none.size()},
            shs::gfx::GfxContext{});
    }
} // namespace

int main()
{
    bool ok = true;
    auto run = [&](const char* name, bool result)
    {
        std::fprintf(stderr, "[gfx-tests] %s: %s\n", name, result ? "pass" : "FAIL");
        ok = result && ok;
    };

    run("handle_validity", test_handle_validity());
    run("pixel_buffer", test_pixel_buffer());
    run("identity_stable", test_identity_stable());
    run("replay_deterministic", test_replay_deterministic());

    if (!ok)
    {
        std::fprintf(stderr, "[gfx-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[gfx-tests] all tests passed\n");
    return 0;
}
