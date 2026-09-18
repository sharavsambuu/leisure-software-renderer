#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: sw_offscreen.hpp
    МОДУЛЬ: rhi/software
    ЗОРИЛГО: Software backend-ийн offscreen execution хэрэгжүүлэлт.
            IRenderBackend::offscreen_execution() энэ объектыг буцаана.
            GPU тал (rhi/vulkan) ижил гэрээг Vulkan pipeline-аар гүйцэтгэдэг;
            энэ нь тэр гэрээний CPU бодит хэрэгжүүлэлт (equivalence-ийн нэг тал).
*/

#include <cstdint>
#include <span>
#include <string_view>

#include <glm/glm.hpp>

#include "shs/rhi/core/offscreen_execution.hpp"
#include "shs/render/shader/program.hpp"
#include "shs/render/shader/shader_identity.hpp"
#include "shs/render/software/rasterizer.hpp"
#include "shs/render/targets/rt_types.hpp"
#include "shs/resources/mesh.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace rhi
    {
    // The CPU counterpart of the authored minimal-scene recipe
    // (tests/shaders/offscreen_pipeline.slang). The GPU fetches its positions by
    // SV_VulkanVertexID *inside* the vertex shader; the CPU has no vertex-ID
    // input, so the identical fetch is realized as an explicit triangle-list mesh
    // and the vertex program only has to pass the fetched NDC position through.
    // This decomposition is behaviorally equivalent for a triangle list, and it is
    // why the CPU realization is bound to an authored recipe rather than to
    // arbitrary SPIR-V.
    namespace sw_offscreen_recipe
    {
        // Authored positions of vs_main(), in source order.
        inline const glm::vec3* authored_positions()
        {
            static const glm::vec3 positions[3] = {
                glm::vec3(-0.5f, -0.5f, 0.0f),
                glm::vec3(0.5f, -0.5f, 0.0f),
                glm::vec3(0.0f, 0.5f, 0.0f),
            };
            return positions;
        }

        // Entry names this build has a CPU realization for. A descriptor whose
        // modules name anything else is rejected by prepare_offscreen().
        // Single-sourced from the shader identity vocabulary (Slang plan P1.5):
        // the same constants the authored offscreen_pipeline.slang module's
        // entry points are declared under, so the pair cannot drift apart.
        inline constexpr std::string_view vertex_entry = shs::render::kShaderEntryVsMain;
        inline constexpr std::string_view fragment_entry = shs::render::kShaderEntryFsMain;

        // Realizes vs_main()/vs_uploaded(): the fetched position already is clip
        // space (the authored z/w are 0/1).
        inline auto triangle_vs()
        {
            return [](const shs::render::ShaderVertex& v, const shs::render::ShaderUniforms&) {
                shs::render::VertexOut out{};
                out.clip = glm::vec4(v.position.x, v.position.y, 0.0f, 1.0f);
                return out;
            };
        }

        // Realizes fs_main(): the authored flat color.
        inline auto flat_fs()
        {
            return [](const shs::render::FragmentIn&, const shs::render::ShaderUniforms&) {
                shs::render::FragmentOut out{};
                out.color = shs::render::ColorF{1.0f, 0.25f, 0.0f, 1.0f};
                return out;
            };
        }

        // Non-erased concrete program (renderer-lib review R1): rasterize_mesh
        // inlines both callables directly, no std::function in the hot path.
        inline auto flat_triangle_program()
        {
            return shs::render::ShaderProgramFn<decltype(triangle_vs()), decltype(flat_fs())>{
                triangle_vs(), flat_fs()};
        }

        // The same program, type-erased for the identity layer only. The
        // execution path keeps using flat_triangle_program() above, so the
        // per-pixel call stays inlinable (R1) while identity, resolution and
        // tests get one handle.
        inline shs::render::ShaderProgram erased_flat_triangle_program()
        {
            return shs::render::make_erased_program(flat_triangle_program());
        }

        // One identity, two realizations: this CPU program, and the authored
        // tests/shaders/offscreen_pipeline.slang module whose entry points are
        // the very constants above. The module name is the same stem the CMake
        // slangc step compiles.
        inline shs::render::ShaderDesc offscreen_pipeline_shader_desc()
        {
            shs::render::ShaderDesc d{};
            d.name = shs::render::shader_id_builtin_name(shs::render::ShaderId::OffscreenPipeline);
            d.entries = shs::render::ShaderEntryPoints{vertex_entry, fragment_entry, {}};
            d.module = "offscreen_pipeline";
            d.realization_mask = static_cast<uint8_t>(
                shs::render::kShaderRealizationSoftware | shs::render::kShaderRealizationVulkan);
            d.cpp_impl = &erased_flat_triangle_program;
            return d;
        }

        // Caller-owned and built on demand: no ambient registry decides what the
        // CPU is able to execute.
        inline shs::render::ShaderManifest offscreen_shader_manifest()
        {
            shs::render::ShaderManifest m{};
            const shs::render::ShaderDesc d = offscreen_pipeline_shader_desc();
            (void)m.register_shader(shs::render::ShaderId::OffscreenPipeline, d.name, d);
            return m;
        }
        // Stable, descriptor-derived identity: identical descriptors must resolve
        // to the same id across reset and re-preparation, exactly as the GPU
        // path's hash-keyed registries do, so a recorded stream stays valid across
        // a reset/re-prepare cycle.
        inline uint64_t id_mix(uint64_t h, uint64_t v)
        {
            h ^= v;
            h *= 1099511628211ull; // FNV-1a prime
            return h == 0 ? 1ull : h;
        }

        inline uint64_t target_id_of(const RHIImageDesc& d)
        {
            uint64_t h = 1469598103934665603ull; // FNV-1a offset basis
            h = id_mix(h, static_cast<uint64_t>(d.type));
            h = id_mix(h, static_cast<uint64_t>(d.width));
            h = id_mix(h, static_cast<uint64_t>(d.height));
            h = id_mix(h, static_cast<uint64_t>(d.format));
            h = id_mix(h, static_cast<uint64_t>(d.usage));
            h = id_mix(h, static_cast<uint64_t>(d.mip_levels));
            h = id_mix(h, static_cast<uint64_t>(d.layers));
            return h;
        }

        // Keyed by the shape the CPU realization actually uses — layout, cull,
        // front face and the resolved entry names — not by bytecode the CPU
        // cannot execute.
        inline uint64_t pipeline_id_of(const RHIGraphicsPipelineDesc& d)
        {
            uint64_t h = 1099511628211ull;
            h = id_mix(h, static_cast<uint64_t>(d.vertex_layout));
            h = id_mix(h, static_cast<uint64_t>(d.raster.cull));
            h = id_mix(h, static_cast<uint64_t>(d.raster.front_face));
            for (const char* c = d.vs.entry; c && *c; ++c)
                h = id_mix(h, static_cast<unsigned char>(*c));
            for (const char* c = d.fs.entry; c && *c; ++c)
                h = id_mix(h, static_cast<unsigned char>(*c));
            return h;
        }

        // SV_VulkanVertexID % 3 realized as an explicit triangle list. Vertices
        // beyond the last whole triangle are dropped, exactly as triangle-list
        // topology drops them on the GPU.
        inline shs::resources::MeshData procedural_mesh(uint32_t vertex_count)
        {
            shs::resources::MeshData mesh{};
            const uint32_t drawn = (vertex_count / 3u) * 3u;
            const glm::vec3* positions = authored_positions();
            mesh.positions.reserve(drawn);
            mesh.indices.reserve(drawn);
            for (uint32_t i = 0; i < drawn; ++i)
            {
                mesh.positions.push_back(positions[i % 3u]);
                mesh.indices.push_back(i);
            }
            return mesh;
        }

        // RGBA8 store of the CPU HDR buffer, round-half-up and clamped: the same
        // quantum a UNORM GPU write lands on. Alpha is stored as-is (the authored
        // fragment alpha is 1, the pass clear alpha is 0).
        inline void store_rgba8(const shs::render::RT_ColorHDR& target, std::span<uint8_t> pixels)
        {
            for (int y = 0; y < target.h; ++y)
            {
                for (int x = 0; x < target.w; ++x)
                {
                    const shs::render::ColorF& c = target.color.at(x, y);
                    uint8_t* out = pixels.data() + (static_cast<size_t>(y) *
                        static_cast<size_t>(target.w) + static_cast<size_t>(x)) * 4u;
                    out[0] = static_cast<uint8_t>(glm::clamp(c.r, 0.0f, 1.0f) * 255.0f + 0.5f);
                    out[1] = static_cast<uint8_t>(glm::clamp(c.g, 0.0f, 1.0f) * 255.0f + 0.5f);
                    out[2] = static_cast<uint8_t>(glm::clamp(c.b, 0.0f, 1.0f) * 255.0f + 0.5f);
                    out[3] = static_cast<uint8_t>(glm::clamp(c.a, 0.0f, 1.0f) * 255.0f + 0.5f);
                }
            }
        }
    } // namespace sw_offscreen_recipe

    // CPU realization of the generic offscreen contract. Owns no device, so
    // initialize_device() always succeeds; prepare_offscreen() is the single
    // point that can still reject. The realized slice is deliberately narrow and
    // is exactly the authored recipe: one RGBA8 target cleared to transparent
    // black, plus an attribute-less (Procedural) triangle-list pipeline whose
    // entry points have a CPU counterpart. Anything outside that slice is
    // rejected by return 0 / false — never approximated.
    //
    // Rejection is stricter than the Vulkan realization in one direction only:
    // commands with no CPU realization (vertex/index binding, indexed draws,
    // instancing, first-vertex offsets, dispatch) are refused here even where a
    // GPU could execute them, because the generic contract exposes no
    // buffer-creation surface yet. The accepted set of this class is therefore a
    // subset of the Vulkan realization's; see the G4 evidence note.
    class SoftwareOffscreenExecution final : public IOffscreenExecution
    {
    public:
        SoftwareOffscreenExecution() = default;
        SoftwareOffscreenExecution(const SoftwareOffscreenExecution&) = delete;
        SoftwareOffscreenExecution& operator=(const SoftwareOffscreenExecution&) = delete;

        [[nodiscard]] bool initialize_device() override
        {
            device_open_ = true;
            return true;
        }

        [[nodiscard]] uint64_t prepare_offscreen(const RHIImageDesc& target,
                                                 const RHIGraphicsPipelineDesc& pipeline) override
        {
            // Re-preparation without reset is refused, as on the GPU path.
            if (!device_open_ || target_id_ != 0) return 0;
            if (!target_supported(target) || !rhi_graphics_pipeline_desc_supported(pipeline))
                return 0;
            // The generic contract has no buffer-creation entry point, so only the
            // attribute-less Procedural layout is realizable on the CPU.
            if (pipeline.vertex_layout != RHIVertexLayout::Procedural) return 0;
            // The CPU cannot execute arbitrary SPIR-V: the descriptor's entry
            // points must resolve to a registered shader identity realized on
            // this backend, so an unknown name is a rejection, never a silent
            // approximation. The law lives in the shader identity layer (Slang
            // plan P1.5) rather than in a pair of literals here.
            if (!pipeline.vs.entry || !pipeline.fs.entry) return 0;
            const auto shader_manifest = sw_offscreen_recipe::offscreen_shader_manifest();
            const auto resolved = shader_manifest.resolve(
                shs::render::ShaderId::OffscreenPipeline,
                shs::render::RenderBackendType::Software,
                shs::render::ShaderEntryPoints{
                    std::string_view(pipeline.vs.entry),
                    std::string_view(pipeline.fs.entry),
                    {}});
            if (!resolved) return 0;

            target_id_ = sw_offscreen_recipe::target_id_of(target);
            pipeline_id_ = sw_offscreen_recipe::pipeline_id_of(pipeline);
            width_ = target.width;
            height_ = target.height;
            cull_ = pipeline.raster.cull;
            front_face_ccw_ = pipeline.raster.front_face == RHIFrontFace::CCW;
            color_ = shs::render::RT_ColorHDR(width_, height_,
                shs::render::ColorF{0.0f, 0.0f, 0.0f, 0.0f});
            return pipeline_id_;
        }

        [[nodiscard]] uint64_t offscreen_target() const override { return target_id_; }

        [[nodiscard]] bool execute_offscreen(std::span<const RHICmd> commands,
                                             std::span<uint8_t> pixels) override
        {
            if (target_id_ == 0) return false;
            if (commands.empty() ||
                !std::holds_alternative<RHICmdBeginPassDesc>(commands.front().payload) ||
                !std::holds_alternative<RHICmdEndPassDesc>(commands.back().payload))
                return false;
            // Same readback contract as the GPU path: the buffer size is exact.
            if (pixels.size() != static_cast<size_t>(width_) * static_cast<size_t>(height_) * 4u)
                return false;

            color_.clear(shs::render::ColorF{0.0f, 0.0f, 0.0f, 0.0f});
            shs::render::RasterizerConfig config{};
            config.cull_mode = cull_ == RHICullMode::Back
                ? shs::render::RasterizerCullMode::Back
                : (cull_ == RHICullMode::Front ? shs::render::RasterizerCullMode::Front
                                               : shs::render::RasterizerCullMode::None);
            config.front_face_ccw = front_face_ccw_;
            config.job_system = nullptr;
            const auto program = sw_offscreen_recipe::flat_triangle_program();
            const shs::render::ShaderUniforms uniforms{};
            const shs::render::RasterizerTarget raster_target{&color_, nullptr};

            bool inside_pass = false;
            bool pipeline_bound = false;
            bool did_clear = false;
            for (const RHICmd& cmd : commands)
            {
                if (const auto* d = std::get_if<RHICmdBeginPassDesc>(&cmd.payload))
                {
                    // Mirrors the GPU pass acceptance: this target, no depth
                    // attachment, and a mandatory clear (transparent black — the
                    // RHI carries no configurable clear value).
                    if (inside_pass || d->color_target != target_id_ || d->depth_target ||
                        !d->clear_color || d->clear_depth)
                        return false;
                    inside_pass = true;
                    pipeline_bound = false;
                    did_clear = true;
                }
                else if (const auto* d = std::get_if<RHICmdBindPipelineDesc>(&cmd.payload))
                {
                    if (d->pipeline != pipeline_id_) return false;
                    pipeline_bound = true;
                }
                else if (const auto* d = std::get_if<RHICmdDrawDesc>(&cmd.payload))
                {
                    if (!inside_pass || !pipeline_bound || d->instance_count != 1 ||
                        d->first_instance != 0 || d->first_vertex != 0)
                        return false;
                    const auto mesh = sw_offscreen_recipe::procedural_mesh(d->vertex_count);
                    if (mesh.positions.empty()) continue;
                    (void)shs::render::rasterize_mesh(mesh, program, uniforms, raster_target, config);
                }
                else if (std::holds_alternative<RHICmdBarrierDesc>(cmd.payload))
                {
                    // CPU work is already complete at the call site: nothing is in
                    // flight for a barrier to order.
                }
                else if (std::holds_alternative<RHICmdEndPassDesc>(cmd.payload))
                {
                    if (!inside_pass) return false;
                    inside_pass = false;
                }
                else
                {
                    // bind_vertex_buffer / bind_index_buffer / draw_indexed /
                    // dispatch: outside the generic CPU slice.
                    return false;
                }
            }
            if (inside_pass || !did_clear) return false;

            sw_offscreen_recipe::store_rgba8(color_, pixels);
            return true;
        }

        void reset_offscreen() override
        {
            target_id_ = 0;
            pipeline_id_ = 0;
            width_ = 0;
            height_ = 0;
            cull_ = RHICullMode::None;
            front_face_ccw_ = true;
            color_ = shs::render::RT_ColorHDR{};
        }

        [[nodiscard]] bool device_open() const { return device_open_; }

    private:
        [[nodiscard]] static bool target_supported(const RHIImageDesc& d)
        {
            return d.type == RHIImageType::Tex2D && d.format == RHIFormat::RGBA8_UNorm &&
                d.width > 0 && d.height > 0 && d.mip_levels == 1 && d.layers == 1 &&
                (d.usage & RHIImageUsage_ColorAttachment) != 0 &&
                (d.usage & RHIImageUsage_TransferSrc) != 0;
        }

        bool device_open_ = false;
        // Descriptor-derived, so identical descriptors keep the same ids across a
        // reset/re-prepare cycle (the GPU path's registries behave the same way).
        uint64_t target_id_ = 0;
        uint64_t pipeline_id_ = 0;
        int width_ = 0;
        int height_ = 0;
        RHICullMode cull_ = RHICullMode::None;
        bool front_face_ccw_ = true;
        shs::render::RT_ColorHDR color_{};
    };

    } // inline namespace rhi
}
