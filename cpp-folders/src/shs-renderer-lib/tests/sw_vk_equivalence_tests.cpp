// G4 acceptance: library SW/Vulkan equivalence through the real execution path.
//
// Both halves obtain their backend from create_render_backend() and drive it
// through the generic IRenderBackend / IOffscreenExecution surface, with the
// SAME consumer code and the SAME command stream. This translation unit never
// names a concrete backend type and includes no GPU header.
//
// Three separate claims, deliberately not conflated:
//   1. Known-answer independence — each half's pixels are checked against the
//      AUTHORED scene (tests/shaders/offscreen_pipeline.slang), never against the
//      other half's output.
//   2. Documented per-output tolerances — see TOLERANCE notes below.
//   3. Equivalence over the declared slice — the two readbacks are compared under
//      those tolerances.
//
// Skip discipline: the software half is always asserted (a CPU rasterizer needs
// no device, so a refusal there is a FAIL, not a skip). Only the Vulkan half may
// report itself unavailable, and then the process exits 77 (CTest
// SKIP_RETURN_CODE) after the software half has passed — never a silent pass.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <span>
#include <string>
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

namespace
{
    constexpr int W = 32;
    constexpr int H = 32;

    // TOLERANCE (color, both covered): the authored fragment color is
    // float4(1.0, 0.25, 0.0, 1.0) and both paths store RGBA8, so agreement is
    // exact up to one 8-bit quantum (0.25 * 255 = 63.75 -> 64).
    constexpr int kChannelTolerance = 1;
    // TOLERANCE (coverage): the two rasterizers use different screen mappings —
    // the CPU path samples the (W-1)/(H-1) extent, the GPU path the w/h extent of
    // its viewport — so boundary pixels of a shared edge may disagree while
    // interior pixels never do. The budget below is the measured cost of that
    // convention gap at this 32x32 fixture; it is not a blanket color allowance.
    constexpr int kEdgeBudget = 16;

    struct Probe
    {
        bool ok = true;
        std::string message{};

        void expect(bool condition, const char* what)
        {
            if (!condition && ok)
            {
                ok = false;
                message = what;
            }
        }
    };

    struct HalfResult
    {
        bool available = false;
        const char* skip_reason = nullptr;
        Probe probe{};
        std::vector<uint8_t> first{};
        bool produced = false;
    };
    // The one consumer path both halves share: factory -> app context -> generic
    // IRenderBackend -> generic IOffscreenExecution. Every probe below is policy
    // the two realizations must agree on, not an implementation detail.
    HalfResult run_half(shs::RenderBackendType which, const std::vector<uint32_t>& vs,
                        const std::vector<uint32_t>& fs, const char* label)
    {
        using namespace shs;
        HalfResult r{};

        auto created = app::create_render_backend(which);
        std::fprintf(stderr, "[%s] factory: requested=%s active=%s note=\"%s\"\n", label,
            render_backend_type_name(created.requested), render_backend_type_name(created.active),
            created.note.c_str());
        if (!created.backend || created.active != which)
        {
            r.skip_reason = "this build cannot activate the requested backend";
            return r;
        }

        app::Context ctx{};
        ctx.register_backend(created.backend.get());
        ctx.set_primary_backend(created.backend.get());
        IRenderBackend* generic = ctx.backend(which);
        if (!generic)
        {
            r.skip_reason = "context lost the registered backend";
            return r;
        }
        // Diagnostics retained from the concrete backend through the generic
        // surface: name + capability claims, never a Vk* handle.
        std::fprintf(stderr, "[%s] backend=\"%s\" supports_offscreen=%d supports_present=%d\n", label,
            generic->name(), generic->capabilities().supports_offscreen ? 1 : 0,
            generic->capabilities().supports_present ? 1 : 0);

        IOffscreenExecution* offscreen = generic->offscreen_execution();
        if (!offscreen)
        {
            r.skip_reason = "backend exposes no generic offscreen surface";
            return r;
        }
        if (!offscreen->initialize_device())
        {
            r.skip_reason = "no device available for offscreen work";
            return r;
        }
        r.available = true;

        RHIImageDesc target{};
        target.width = W;
        target.height = H;
        target.format = RHIFormat::RGBA8_UNorm;
        target.usage = RHIImageUsage_ColorAttachment | RHIImageUsage_TransferSrc;

        RHIGraphicsPipelineDesc pipeline{};
        pipeline.vs = {RHIShaderStage::Vertex, vs.data(), vs.size() * 4, "vs_main"};
        pipeline.fs = {RHIShaderStage::Fragment, fs.data(), fs.size() * 4, "fs_main"};
        pipeline.rt.has_depth = false;
        pipeline.depth = {false, false};
        pipeline.raster.cull = RHICullMode::None;

        // A target that cannot be read back is refused, with no id left behind.
        auto unreadable = target;
        unreadable.usage = RHIImageUsage_ColorAttachment;
        r.probe.expect(offscreen->prepare_offscreen(unreadable, pipeline) == 0,
            "target without TransferSrc was accepted");
        r.probe.expect(offscreen->offscreen_target() == 0,
            "a refused preparation still reported a target id");

        // A module outside the fixed ABI envelope is refused.
        auto wrong_stage = pipeline;
        wrong_stage.fs.stage = RHIShaderStage::Vertex;
        r.probe.expect(offscreen->prepare_offscreen(target, wrong_stage) == 0,
            "a fragment module declared as a vertex stage was accepted");

        const uint64_t prepared = offscreen->prepare_offscreen(target, pipeline);
        if (prepared == 0)
        {
            r.skip_reason = "preparation refused (capability or device)";
            r.available = false;
            return r;
        }
        const uint64_t resolved = offscreen->offscreen_target();
        r.probe.expect(resolved != 0, "a successful preparation reported target id 0");
        r.probe.expect(offscreen->prepare_offscreen(target, pipeline) == 0,
            "re-preparation without reset was accepted");

        const RHICmd stream[] = {rhi_cmd_begin_pass({resolved, 0, true, false}),
            rhi_cmd_bind_pipeline(prepared), rhi_cmd_draw({3}), rhi_cmd_end_pass()};

        std::vector<uint8_t> pixels(static_cast<size_t>(W) * H * 4, 0xCD);
        r.probe.expect(offscreen->execute_offscreen(stream, pixels),
            "the canonical minimal-scene stream was refused");
        r.first = pixels;
        r.produced = true;


        // Refusals, not approximations.
        const RHICmd no_pipeline[] = {stream[0], rhi_cmd_draw({3}), stream[3]};
        r.probe.expect(!offscreen->execute_offscreen(no_pipeline, pixels),
            "a draw without a bound pipeline was accepted");
        r.probe.expect(!offscreen->execute_offscreen({}, pixels), "an empty stream was accepted");
        r.probe.expect(!offscreen->execute_offscreen(stream, std::span<uint8_t>(pixels).first(1)),
            "an undersized readback buffer was accepted");
        const RHICmd foreign_target[] = {rhi_cmd_begin_pass({resolved + 7, 0, true, false}),
            rhi_cmd_bind_pipeline(prepared), rhi_cmd_draw({3}), rhi_cmd_end_pass()};
        r.probe.expect(!offscreen->execute_offscreen(foreign_target, pixels),
            "a stream naming a foreign target id was accepted");
        r.probe.expect(!offscreen->execute_offscreen(std::span<const RHICmd>(stream).first(3), pixels),
            "a stream with no end pass was accepted");

        // Repeated execution is stable — byte for byte.
        std::vector<uint8_t> again(pixels.size(), 0xCD);
        r.probe.expect(offscreen->execute_offscreen(stream, again), "repeated execution was refused");
        r.probe.expect(again == r.first, "repeated execution did not reproduce the same pixels");

        // Reset invalidates the surface, and re-preparing the same descriptor
        // resolves the same ids so the recorded stream stays valid.
        offscreen->reset_offscreen();
        r.probe.expect(offscreen->offscreen_target() == 0, "target id survived reset");
        r.probe.expect(!offscreen->execute_offscreen(stream, pixels), "execution survived reset");
        r.probe.expect(offscreen->prepare_offscreen(target, pipeline) == prepared,
            "re-preparation resolved a different pipeline id for the same descriptor");
        r.probe.expect(offscreen->offscreen_target() == resolved,
            "re-preparation resolved a different target id for the same descriptor");
        r.probe.expect(offscreen->execute_offscreen(stream, pixels),
            "the recorded stream stopped working after a reset/re-prepare cycle");

        // A different in-contract pipeline descriptor resolves a different id.
        offscreen->reset_offscreen();
        auto culled = pipeline;
        culled.raster.cull = RHICullMode::Back;
        const uint64_t culled_id = offscreen->prepare_offscreen(target, culled);
        r.probe.expect(culled_id != 0, "a legal cull-mode change was refused");
        r.probe.expect(culled_id != prepared, "a different pipeline descriptor reused a pipeline id");
        offscreen->reset_offscreen();

        return r;
    }

    // Known-answer check, run independently for each half: expectations come from
    // the authored scene, never from the other backend's output.
    // offscreen_pipeline.slang holds the triangle at NDC (-0.5,-0.5) (0.5,-0.5)
    // (0.0,0.5) and fs_main() returns the constant float4(1.0, 0.25, 0.0, 1.0).
    // Both paths map NDC y downward, so that is the pixel triangle (8,8) (24,8)
    // (16,24): base edge on row 8, apex on row 24. Pixel (16,12) is interior and
    // must read RGBA8 (255,64,0,255); (16,28) is below the apex and (1,1) is
    // outside both edges, so both keep the pass clear value (0,0,0,0).
    Probe known_answer(const std::vector<uint8_t>& pixels)
    {
        Probe p{};
        const auto is = [&](int x, int y, int r, int g, int b, int a) {
            const uint8_t* q = pixels.data() + (static_cast<size_t>(y) * W + x) * 4u;
            return q[0] == r && q[1] == g && q[2] == b && q[3] == a;
        };
        p.expect(is(16, 12, 255, 64, 0, 255), "authored interior sample is not (255,64,0,255)");
        p.expect(is(16, 28, 0, 0, 0, 0), "a pixel below the apex is not the clear value");
        p.expect(is(1, 1, 0, 0, 0, 0), "the top-left corner is not the clear value");
        return p;
    }

    // Equivalence over the declared slice, under the two documented tolerances.
    // Coverage is read from alpha: the RHI clear is transparent black and the
    // authored fragment alpha is 1, so alpha separates "covered" from "cleared"
    // on both paths without a hand-tuned colour heuristic.
    Probe compare_readbacks(const std::vector<uint8_t>& cpu, const std::vector<uint8_t>& gpu)
    {
        Probe p{};
        int mismatches = 0, max_delta = 0, cpu_covered = 0, gpu_covered = 0;
        int mm_both = 0, mm_gpu_only = 0, mm_cpu_only = 0;
        for (int y = 0; y < H; ++y)
        {
            for (int x = 0; x < W; ++x)
            {
                const size_t at = (static_cast<size_t>(y) * W + x) * 4u;
                const uint8_t* c = cpu.data() + at;
                const uint8_t* g = gpu.data() + at;
                const bool c_cov = c[3] != 0;
                const bool g_cov = g[3] != 0;
                cpu_covered += c_cov ? 1 : 0;
                gpu_covered += g_cov ? 1 : 0;
                int delta = 0;
                for (int ch = 0; ch < 4; ++ch)
                    delta = std::max(delta, std::abs(int(c[ch]) - int(g[ch])));
                max_delta = std::max(max_delta, delta);
                if (delta <= kChannelTolerance) continue;
                ++mismatches;
                if (c_cov && g_cov) ++mm_both;
                else if (g_cov) ++mm_gpu_only;
                else ++mm_cpu_only;
            }
        }
        std::fprintf(stderr,
            "equivalence: cpu_covered=%d gpu_covered=%d mismatches=%d "
            "(both=%d gpu_only=%d cpu_only=%d) max_delta=%d\n",
            cpu_covered, gpu_covered, mismatches, mm_both, mm_gpu_only, mm_cpu_only, max_delta);
        // Where both rasterizers covered a pixel, the stored colour must agree
        // within one 8-bit quantum. The GPU may only ADD coverage (its viewport
        // uses the w/h extent, the CPU path the (W-1)/(H-1) extent), never blank
        // a pixel the CPU covered; the remaining boundary disagreement is bounded
        // by the documented budget.
        p.expect(mm_both == 0,
            "both rasterizers covered a pixel but disagreed beyond the colour tolerance");
        p.expect(mm_cpu_only == 0, "the GPU left a CPU-covered pixel blank");
        p.expect(mismatches <= kEdgeBudget, "coverage disagreement exceeded the documented edge budget");
        p.expect(cpu_covered > 16 && gpu_covered > 16,
            "one rasterizer covered too little of the target to be a meaningful comparison");
        return p;
    }
} // namespace

// G4 acceptance, in three separate stages so a failure names its own claim.
//
// Stage 1/2 run the SOFTWARE half unconditionally: a CPU rasterizer needs no
// device, so a refusal anywhere in it is a hard FAIL. Only stage 3 may report the
// Vulkan half unavailable, and then this process exits 77 — the software half is
// proven, the equivalence claim is explicitly NOT made, and CTest records a skip
// rather than a pass.
int main()
{
    using namespace shs;

#ifndef SHS_OFFSCREEN_SHADER_DIR
    // Both realizations are gated by the same descriptor envelope, which carries
    // the SPIR-V module, so neither half can run without it.
    std::fprintf(stderr, "SKIP: this build has no minimal-scene SPIR-V\n");
    return 77;
#else
    const auto vs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_vs.spv");
    const auto fs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_fs.spv");
    if (vs.empty() || fs.empty())
    {
        std::fprintf(stderr, "SKIP: minimal-scene SPIR-V unavailable\n");
        return 77;
    }

    // --- stage 1: the software realization, the same consumer path as the GPU --
    const HalfResult software = run_half(RenderBackendType::Software, vs, fs, "cpu");
    if (!software.available || !software.produced)
    {
        std::fprintf(stderr, "FAIL: the software backend exposes no usable generic offscreen "
            "surface (%s)\n", software.skip_reason ? software.skip_reason : "no pixels produced");
        return 1;
    }
    if (!software.probe.ok)
    {
        std::fprintf(stderr, "FAIL: software half: %s\n", software.probe.message.c_str());
        return 1;
    }

    // --- stage 2: independent known-answer check of the CPU readback ----------
    const Probe sw_known = known_answer(software.first);
    if (!sw_known.ok)
    {
        std::fprintf(stderr, "FAIL: software known-answer: %s\n", sw_known.message.c_str());
        return 1;
    }
    std::fprintf(stderr, "known-answer: CPU readback matches the authored scene\n");

    // --- stage 3: the Vulkan realization, then the equivalence claim ----------
    const HalfResult vulkan = run_half(RenderBackendType::Vulkan, vs, fs, "gpu");
    if (!vulkan.available)
    {
        std::fprintf(stderr, "SKIP: Vulkan half unavailable (%s). The software half passed, but "
            "library SW/Vulkan equivalence was NOT established.\n",
            vulkan.skip_reason ? vulkan.skip_reason : "unknown reason");
        return 77;
    }
    if (!vulkan.probe.ok)
    {
        std::fprintf(stderr, "FAIL: Vulkan half: %s\n", vulkan.probe.message.c_str());
        return 1;
    }
    if (!vulkan.produced)
    {
        std::fprintf(stderr, "FAIL: the Vulkan half produced no pixels\n");
        return 1;
    }
    const Probe vk_known = known_answer(vulkan.first);
    if (!vk_known.ok)
    {
        std::fprintf(stderr, "FAIL: Vulkan known-answer: %s\n", vk_known.message.c_str());
        return 1;
    }
    std::fprintf(stderr, "known-answer: GPU readback matches the authored scene\n");

    const Probe equivalence = compare_readbacks(software.first, vulkan.first);
    if (!equivalence.ok)
    {
        std::fprintf(stderr, "FAIL: equivalence: %s\n", equivalence.message.c_str());
        return 1;
    }

    std::fprintf(stderr,
        "PASS: library SW/Vulkan equivalence through the generic execution path "
        "(32x32 authored triangle, colour tolerance %d/255, edge budget %d)\n",
        kChannelTolerance, kEdgeBudget);
    return 0;
#endif
}
