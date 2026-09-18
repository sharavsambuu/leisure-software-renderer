#include <cstdio>
#include <cstdint>
#include <fstream>
#include <span>
#include <vector>

#include "shs/app/backend/backend_factory.hpp"
#include "shs/app/context.hpp"
#include "shs/rhi/core/offscreen_execution.hpp"

#ifdef SHS_OFFSCREEN_SHADER_DIR
static std::vector<uint32_t> read_spirv(const char* path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return {};
    const auto bytes = file.tellg();
    if (bytes <= 0 || bytes % 4 != 0) return {};
    std::vector<uint32_t> words(static_cast<size_t>(bytes) / 4);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(words.data()), bytes)) return {};
    return words;
}
#endif

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// G3 acceptance: factory-facing execution.
//
// The minimal scene must execute when the backend is obtained from
// create_render_backend() and driven through the generic IRenderBackend /
// IOffscreenExecution surface. This translation unit never names
// VulkanRenderBackend, includes no Vulkan header and touches no Vk* type — that
// is the whole point of the gate.
//
// Known-answer independence: the expected pixels are derived from the authored
// scene, not from a parity run. offscreen_pipeline.slang holds the triangle at
// NDC (-0.5,-0.5) (0.5,-0.5) (0.0,0.5) and fs_main() returns the constant
// float4(1.0, 0.25, 0.0, 1.0). On a 32x32 target the viewport maps NDC
// (-0.5,-0.5) (0.5,-0.5) (0.0,0.5) to the pixel triangle (8,8) (24,8) (16,24) —
// base edge on row 8, apex on row 24 — so (16,12) is covered with RGBA8
// (255,64,0,255) and (16,28), below the apex, keeps the pass clear value
// (0,0,0,0).
//
// An unavailable device, pipeline or generic surface is reported as SKIP (77),
// never as a pass.
int main()
{
    using namespace shs;

    // 1. Factory selection: the backend comes from the factory, never a local
    //    concrete instance.
    auto created = app::create_render_backend(RenderBackendType::Vulkan);
    CHECK(created.backend != nullptr);
    CHECK(created.requested == RenderBackendType::Vulkan);

    // 2. Registration through the app-owned context: the consumer path resolves
    //    the backend by type, exactly as the renderpath executor does.
    app::Context ctx{};
    ctx.register_backend(created.backend.get());
    ctx.set_primary_backend(created.backend.get());
    IRenderBackend* generic = ctx.backend(RenderBackendType::Vulkan);
    CHECK(generic != nullptr);
    CHECK(generic->type() == created.backend->type());

    // 3. The generic offscreen surface is optional. Null means "fall back or
    //    skip" — never "passed".
    IOffscreenExecution* offscreen = generic->offscreen_execution();
    if (offscreen == nullptr)
    {
        std::fprintf(stderr, "SKIP: %s exposes no generic offscreen surface. %s\n",
            generic->name(), created.note.c_str());
        return 77;
    }

#ifndef SHS_OFFSCREEN_SHADER_DIR
    std::fprintf(stderr, "SKIP: this build has no minimal-scene SPIR-V\n");
    return 77;
#else
    // 4. Open the device through the generic surface — no dynamic_cast, no
    //    Vulkan type. An unavailable device is a skip, never a pass.
    if (!offscreen->initialize_device())
    {
        std::fprintf(stderr, "SKIP: no Vulkan device available for offscreen work\n");
        return 77;
    }

    auto vs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_vs.spv");
    auto fs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_fs.spv");
    if (vs.empty() || fs.empty())
    {
        std::fprintf(stderr, "SKIP: minimal-scene SPIR-V unavailable\n");
        return 77;
    }

    RHIImageDesc target{};
    target.width = 32;
    target.height = 32;
    target.format = RHIFormat::RGBA8_UNorm;
    target.usage = RHIImageUsage_ColorAttachment | RHIImageUsage_TransferSrc;

    RHIGraphicsPipelineDesc pipeline{};
    pipeline.vs = {RHIShaderStage::Vertex, vs.data(), vs.size() * 4, "vs_main"};
    pipeline.fs = {RHIShaderStage::Fragment, fs.data(), fs.size() * 4, "fs_main"};
    pipeline.rt.has_depth = false;
    pipeline.depth = {false, false};
    pipeline.raster.cull = RHICullMode::None;

    // Unsupported descriptors are rejected through the generic surface too: a
    // target without TransferSrc cannot be read back.
    auto unreadable = target;
    unreadable.usage = RHIImageUsage_ColorAttachment;
    CHECK(offscreen->prepare_offscreen(unreadable, pipeline) == 0);
    CHECK(offscreen->offscreen_target() == 0);

    const uint64_t prepared = offscreen->prepare_offscreen(target, pipeline);
    if (prepared == 0)
    {
        std::fprintf(stderr, "SKIP: generic offscreen preparation unavailable (device or capability)\n");
        return 77;
    }
    const uint64_t resolved_target = offscreen->offscreen_target();
    CHECK(resolved_target != 0);

    const RHICmd stream[] = {rhi_cmd_begin_pass({resolved_target, 0, true, false}),
        rhi_cmd_bind_pipeline(prepared), rhi_cmd_draw({3}), rhi_cmd_end_pass()};
    std::vector<uint8_t> pixels(32 * 32 * 4, 0xCD);
    CHECK(offscreen->execute_offscreen(stream, pixels));
    const auto pixel_is = [&](int x, int y, int r, int g, int b, int a) {
        const auto* p = pixels.data() + (y * 32 + x) * 4;
        return p[0] == r && p[1] == g && p[2] == b && p[3] == a;
    };
    CHECK(pixel_is(16, 12, 255, 64, 0, 255)); // authored interior sample
    CHECK(pixel_is(16, 28, 0, 0, 0, 0));      // outside the base edge
    CHECK(pixel_is(1, 1, 0, 0, 0, 0));        // clear value, proves the readback wrote

    // Repeated execution through the generic surface is stable.
    CHECK(offscreen->execute_offscreen(stream, pixels));
    CHECK(pixel_is(16, 12, 255, 64, 0, 255));

    // Rejections are reported rather than silently accepted.
    const RHICmd missing_binding[] = {stream[0], rhi_cmd_draw({3}), stream[3]};
    CHECK(!offscreen->execute_offscreen(missing_binding, pixels));
    CHECK(!offscreen->execute_offscreen({}, pixels));
    CHECK(!offscreen->execute_offscreen(stream, std::span<uint8_t>(pixels).first(1)));

    // Reset invalidates the surface until it is prepared again.
    offscreen->reset_offscreen();
    CHECK(offscreen->offscreen_target() == 0);
    CHECK(!offscreen->execute_offscreen(stream, pixels));
    CHECK(offscreen->prepare_offscreen(target, pipeline) != 0);
    CHECK(offscreen->execute_offscreen(stream, pixels));
    CHECK(pixel_is(16, 12, 255, 64, 0, 255));
    offscreen->reset_offscreen();

    std::fprintf(stderr,
        "PASS: factory-facing generic offscreen execution — known pixels, rejection and reset\n");
    return 0;
#endif
}
