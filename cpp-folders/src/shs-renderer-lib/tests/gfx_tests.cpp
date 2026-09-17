#include <cstdio>

#include "shs/render/targets/gfx.contract.hpp"

// Headless tests for the gfx pod (R5b P3.8: handle/buffer pins). The identity
// gateway scaffolding was retired (migration step 4.5): the gfx pod owns
// value types only until real intents arrive (empty command vocabulary by
// law §6.1). Links only shs::renderer-values + glm. Registry allocation is
// exercised only via its current API as migration spec (edge-candidate until
// R5b).
namespace
{
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

    if (!ok)
    {
        std::fprintf(stderr, "[gfx-tests] FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "[gfx-tests] all tests passed\n");
    return 0;
}
