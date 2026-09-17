#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <vector>

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
#include "shs/rhi/vulkan/value/vk_backend.hpp"
#include "shs/render/software/rasterizer.hpp"

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; } } while (false)

// Step 6 (engine_domain_separation_migration.md): BOUNDED software/Vulkan
// parity. The same fixed NDC triangle with the same flat color runs through
// (a) the real Vulkan offscreen path (GPU raster, RGBA8 readback) and
// (b) the software rasterizer (CPU, HDR float buffer). Comparison is bounded:
// per-channel tolerance of one 8-bit quantum and a small edge-pixel budget
// (the two rasterizers' fill rules may differ on boundary pixels). Every
// unavailable/skipped configuration is EXPLICITLY reported on stderr and
// exits 77 (CTest SKIP_RETURN_CODE) — never a silent pass.
namespace
{
    constexpr int W = 32;
    constexpr int H = 32;

    // Same NDC triangle and flat fragment output as offscreen_pipeline.slang
    // (vs_uploaded positions + fs_main color).
    const shs::MeshData& parity_triangle()
    {
        static const shs::MeshData mesh = [] {
            shs::MeshData m{};
            // Same coordinates as offscreen_pipeline.slang vs_main (the GPU
            // path is attribute-less and uses exactly these).
            m.positions = {
                glm::vec3(-0.5f, -0.5f, 0.0f),
                glm::vec3(0.5f, -0.5f, 0.0f),
                glm::vec3(0.0f, 0.5f, 0.0f),
            };
            m.indices = {0u, 1u, 2u};
            return m;
        }();
        return mesh;
    }

    shs::ShaderProgram parity_program()
    {
        shs::ShaderProgram program{};
        program.vs = [](const shs::ShaderVertex& v, const shs::ShaderUniforms&) {
            shs::VertexOut out{};
            out.clip = glm::vec4(v.position.x, v.position.y, 0.0f, 1.0f); // direct clip coords
            return out;
        };
        program.fs = [](const shs::FragmentIn&, const shs::ShaderUniforms&) {
            shs::FragmentOut out{};
            out.color = shs::ColorF{1.0f, 0.25f, 0.0f, 1.0f};
            return out;
        };
        return program;
    }
}

int main()
{
    // --- software reference: same triangle, CPU raster --------------------
    shs::RT_ColorHDR target(W, H, shs::ColorF{0.0f, 0.0f, 0.0f, 0.0f});
    shs::ShaderUniforms uniforms{};
    const auto stats = shs::rasterize_mesh(parity_triangle(), parity_program(), uniforms,
        shs::RasterizerTarget{&target, nullptr},
        shs::RasterizerConfig{shs::RasterizerCullMode::None, true});
    std::fprintf(stderr, "sw stats: tri_input=%lu tri_after_clip=%lu tri_raster=%lu\n",
        (unsigned long)stats.tri_input, (unsigned long)stats.tri_after_clip,
        (unsigned long)stats.tri_raster);

    // Software-side known pixels (independent of Vulkan availability): the
    // triangle interior carries the flat color; the corner stays clear.
    const auto sw_at = [&](int x, int y) {
        const shs::ColorF& p = target.color.at(x, y);
        return (p.r == 1.0f && p.g == 0.25f && p.b == 0.0f);
    };
    int sw_covered = 0;
    int sw_ymean = 0;
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            if (sw_at(x, y))
            {
                ++sw_covered;
                sw_ymean += y;
            }
        }
    }
    if (sw_covered > 0) sw_ymean /= sw_covered;
    std::fprintf(stderr, "software path: exercised (%d covered pixels, y-mean %d)\n",
        sw_covered, sw_ymean);
    CHECK(sw_covered > 16);
    CHECK(sw_at(16, 12));   // interior (same probe pixel as the VK harness)
    CHECK(!sw_at(1, 1));    // clear corner

#ifndef SHS_HAS_VULKAN
    std::fprintf(stderr, "SKIP: SHS_HAS_VULKAN undefined — GPU/software parity "
        "not exercised in this configuration\n");
    return 77;
#else
    shs::VulkanRenderBackend backend;
    if (!backend.initialize_device())
    {
        std::fprintf(stderr, "SKIP: Vulkan device unavailable — GPU/software "
            "parity not exercised\n");
        return 77;
    }
#ifndef SHS_OFFSCREEN_SHADER_DIR
    std::fprintf(stderr, "SKIP: offscreen SPIR-V unavailable (slangc not "
        "configured) — GPU/software parity not exercised\n");
    return 77;
#else
    const auto vs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_vs.spv");
    const auto fs = read_spirv(SHS_OFFSCREEN_SHADER_DIR "/offscreen_fs.spv");
    if (vs.empty() || fs.empty())
    {
        std::fprintf(stderr, "SKIP: offscreen SPIR-V missing — parity not exercised\n");
        return 77;
    }

    shs::RHIImageDesc desc{};
    desc.width = W;
    desc.height = H;
    desc.format = shs::RHIFormat::RGBA8_UNorm;
    desc.usage = shs::RHIImageUsage_ColorAttachment | shs::RHIImageUsage_TransferSrc;
    shs::RHIGraphicsPipelineDesc pd{};
    // The offscreen path is attribute-less: the triangle comes from vs_main
    // (SV_VulkanVertexID), so no vertex buffer is bound on the GPU side.
    pd.vs = {shs::RHIShaderStage::Vertex, vs.data(), vs.size() * 4, "vs_main"};
    pd.fs = {shs::RHIShaderStage::Fragment, fs.data(), fs.size() * 4, "fs_main"};
    pd.rt.has_depth = false;
    pd.depth = {false, false};
    pd.raster.cull = shs::RHICullMode::None;
    auto prepared = backend.prepare_offscreen(desc, pd);
    CHECK(prepared);

    const shs::RHICmd stream[] = {
        shs::rhi_cmd_begin_pass({backend.offscreen_target(), 0, true, false}),
        shs::rhi_cmd_bind_pipeline(*prepared),
        shs::rhi_cmd_draw({3}),
        shs::rhi_cmd_end_pass(),
    };
    std::vector<uint8_t> gpu_pixels(size_t(W) * H * 4, 0u);
    CHECK(backend.execute_offscreen(stream, gpu_pixels));

    // --- bounded comparison: per-channel tolerance 1, edge budget 8 pixels
    int mismatches = 0;
    int max_delta = 0;
    int gpu_covered = 0;
    int gpu_minx = W, gpu_maxx = -1, gpu_miny = H, gpu_maxy = -1;
    int mm_gpu_only = 0, mm_sw_only = 0, mm_both = 0, match = 0;
    for (int y = 0; y < H; ++y)
    {
        for (int x = 0; x < W; ++x)
        {
            const uint8_t* gp = gpu_pixels.data() + (size_t(y) * W + x) * 4;
            if (gp[0] != 0)
            {
                ++gpu_covered;
                gpu_minx = std::min(gpu_minx, x);
                gpu_maxx = std::max(gpu_maxx, x);
                gpu_miny = std::min(gpu_miny, y);
                gpu_maxy = std::max(gpu_maxy, y);
            }
            const shs::ColorF& sw = target.color.at(x, y);
            const bool sw_written = (sw.r != 0.0f || sw.g != 0.0f || sw.b != 0.0f);
            const int sw_c[4] = {
                int(glm::clamp(sw.r, 0.0f, 1.0f) * 255.0f + 0.5f),
                int(glm::clamp(sw.g, 0.0f, 1.0f) * 255.0f + 0.5f),
                int(glm::clamp(sw.b, 0.0f, 1.0f) * 255.0f + 0.5f),
                sw_written ? 255 : 0};
            bool differs = false;
            for (int c = 0; c < 4; ++c)
            {
                const int delta = std::abs(int(gp[c]) - sw_c[c]);
                max_delta = std::max(max_delta, delta);
                if (delta > 1)
                {
                    differs = true;
                    break;
                }
            }
            if (!differs) { ++match; continue; }
            ++mismatches;
            if (gp[0] != 0 && sw_written) ++mm_both;
            else if (gp[0] != 0) ++mm_gpu_only;
            else ++mm_sw_only;
        }
    }
    std::fprintf(stderr,
        "parity: gpu_covered=%d bbox=(%d..%d, %d..%d) sw_covered=%d mismatches=%d "
        "(both=%d gpu_only=%d sw_only=%d) max_delta=%d\n",
        gpu_covered, gpu_minx, gpu_maxx, gpu_miny, gpu_maxy, sw_covered, mismatches,
        mm_both, mm_gpu_only, mm_sw_only, max_delta);
    // Where BOTH rasterizers covered a pixel, colors must be identical
    // (bounded by one 8-bit quantum). Coverage may differ only on boundary
    // pixels (fill rules): budget 16 at this 32x32 fixture, and the GPU may
    // only ADD coverage — it must never leave a SW-covered pixel blank.
    CHECK(mm_both == 0);
    CHECK(mm_sw_only == 0);
    CHECK(mismatches <= 16);
    CHECK(gpu_covered > 16 && sw_covered > 16);
    std::fprintf(stderr,
        "PASS: software/Vulkan bounded parity (32x32 flat triangle, "
        "tolerance 1/255, edge budget 16)\n");
    return 0;
#endif
#endif
}