#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>

#include "shs/render/shader/builtin_shader_manifest.hpp"
#include "shs/rhi/software/sw_offscreen.hpp"

// Shader identity gate (Slang plan P1.5).
//
// Proves that shader identity is *data*: one ShaderId names one authored
// shader, each backend resolves it to its own realization (software: a C++
// ShaderProgram; Vulkan: an authored module + entry points), and a backend with
// no realization refuses loudly instead of silently approximating. The
// software realization's entry-name law is exercised through the identity layer
// rather than through duplicated literals.
//
// GPU-free: no device, no Context, no SPIR-V execution.
namespace
{
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok)
    {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

using namespace shs;

std::string read_text_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
} // namespace

int main()
{
    // --- Vocabulary law: Unknown and the Count sentinel are never ids -------
    check(shader_id_builtin_name(ShaderId::Unknown).empty(), "Unknown has a canonical name");
    check(!shader_id_is_registerable(ShaderId::Unknown), "Unknown is registerable");
    check(!shader_id_is_registerable(ShaderId::Count), "the Count sentinel is registerable");
    for (uint16_t i = 1; i < static_cast<uint16_t>(ShaderId::Count); ++i)
    {
        const auto id = static_cast<ShaderId>(i);
        check(shader_id_is_registerable(id), "a builtin id is not registerable");
        check(!shader_id_builtin_name(id).empty(), "a builtin id has no canonical name");
    }

    // --- Range law: the vocabulary is open, the old reading is unchanged -----
    // Opening the identity space added an acceptance (the open registered
    // range); it must not have moved any value the closed vocabulary could
    // already hold. This is the parity guard for that claim.
    for (uint16_t i = 0; i <= static_cast<uint16_t>(ShaderId::Count); ++i)
    {
        const auto id = static_cast<ShaderId>(i);
        const bool legacy = i != static_cast<uint16_t>(ShaderId::Unknown) &&
                            i < static_cast<uint16_t>(ShaderId::Count);
        check(shader_id_is_registerable(id) == legacy, "the builtin registerability reading changed");
        check(shader_id_is_builtin(id) == legacy, "the builtin classification changed");
        check(!shader_id_is_open(id), "a builtin-range value read as an open id");
    }
    check(shader_id_is_open(static_cast<ShaderId>(kShaderIdOpenBase)), "the open base is not open");
    check(shader_id_in_valid_range(static_cast<ShaderId>(kShaderIdOpenMax)),
        "the open max is not a valid id");
    check(!shader_id_in_valid_range(static_cast<ShaderId>(kShaderIdReserved)),
        "the reserved top slot is a valid id");
    check(shader_id_name_or_null(ShaderId::BlinnPhong) != nullptr, "a builtin lost its static name");
    check(shader_id_name_or_null(static_cast<ShaderId>(kShaderIdOpenBase)) == nullptr,
        "an open id claimed a static name");
    check(parse_shader_id(shader_id_builtin_name(ShaderId::PbrMetallicRoughness)) ==
              ShaderId::PbrMetallicRoughness,
        "parse_shader_id does not invert shader_id_builtin_name");
    check(shader_id_in_valid_range(ShaderId::Unknown) == false, "Unknown reads as a valid id");

    // --- Registration is verified pairing, and every refusal is named -------
    ShaderManifest m{};
    ShaderDesc soft{};
    soft.name = shader_id_builtin_name(ShaderId::BlinnPhong);
    soft.realization_mask = kShaderRealizationSoftware;
    soft.cpp_impl = &erased_blinn_phong_program;

    auto r = m.register_shader(ShaderId::BlinnPhong, "blinn_phong_typo", soft);
    check(!r && r.error() == ShaderIdentityError::NameMismatch, "an id/name mismatch was accepted");

    r = m.register_shader(ShaderId::BlinnPhong, soft.name, soft);
    check(static_cast<bool>(r), "a correct (id, name) pairing was refused");

    r = m.register_shader(ShaderId::BlinnPhong, soft.name, soft);
    check(!r && r.error() == ShaderIdentityError::AlreadyRegistered, "a duplicate registration was accepted");

    r = m.register_shader(ShaderId::Unknown, "whatever", soft);
    check(!r && r.error() == ShaderIdentityError::UnknownShader, "Unknown was registerable");

    r = m.register_shader(ShaderId::Count, "whatever", soft);
    check(!r && r.error() == ShaderIdentityError::UnknownShader, "the Count sentinel was registerable");

    ShaderDesc no_impl = soft;
    no_impl.name = shader_id_builtin_name(ShaderId::LitDefault);
    no_impl.cpp_impl = nullptr;
    r = m.register_shader(ShaderId::LitDefault, no_impl.name, no_impl);
    check(!r && r.error() == ShaderIdentityError::MissingCppImpl,
        "a software realization without a program factory was accepted");

    ShaderDesc no_module{};
    no_module.name = shader_id_builtin_name(ShaderId::LitDefault);
    no_module.realization_mask = kShaderRealizationVulkan;
    no_module.entries = ShaderEntryPoints{kShaderEntryVsMain, kShaderEntryFsMain, {}};
    r = m.register_shader(ShaderId::LitDefault, no_module.name, no_module);
    check(!r && r.error() == ShaderIdentityError::MissingModule,
        "a Vulkan realization without an authored module was accepted");

    ShaderDesc no_entries = no_module;
    no_entries.module = "somewhere";
    no_entries.entries = ShaderEntryPoints{}; // entry points deliberately absent
    r = m.register_shader(ShaderId::LitDefault, no_entries.name, no_entries);
    check(!r && r.error() == ShaderIdentityError::MissingEntryPoints,
        "a Vulkan realization without entry points was accepted");

    ShaderDesc none{};
    none.name = shader_id_builtin_name(ShaderId::LitDefault);
    r = m.register_shader(ShaderId::LitDefault, none.name, none);
    check(!r && r.error() == ShaderIdentityError::NoRealization,
        "a descriptor claiming no realization was accepted");

    // --- An unregistered id is a hard miss, never an alias ------------------
    const auto miss = m.resolve(ShaderId::PbrMetallicRoughness, RenderBackendType::Software);
    check(!miss && miss.error() == ShaderIdentityError::UnknownShader, "an unregistered id resolved");

    // --- Builtin value-tier identities: software realizations only ----------
    const ShaderManifest value = builtin_value_shader_manifest();
    check(value.size() == 6, "the builtin value manifest size drifted");
    // The open half must be untouched by builtins: opening the range added
    // capacity, not identities.
    check(value.open_count() == 0, "a builtin manifest carries a minted open slot");
    check(value.ids().empty(), "a builtin manifest minted an open name");
    check(ShaderManifest::builtin_capacity() == kShaderIdBuiltinCount,
        "the builtin slot extent drifted");
    for (ShaderId id : {ShaderId::BlinnPhong, ShaderId::PbrMetallicRoughness, ShaderId::LitDefault,
                        ShaderId::DebugViewAlbedo, ShaderId::DebugViewNormal, ShaderId::DebugViewDepth})
    {
        check(value.has(id), "a builtin value identity is not registered");
        const auto sw = value.resolve(id, RenderBackendType::Software);
        check(static_cast<bool>(sw) && sw->has_program && sw->program.valid(),
            "a software builtin did not resolve to a valid program");
        const auto vk = value.resolve(id, RenderBackendType::Vulkan);
        check(!vk && vk.error() == ShaderIdentityError::BackendNotRealized,
            "a software-only builtin did not refuse Vulkan");
        const auto gl = value.resolve(id, RenderBackendType::OpenGL);
        check(!gl && gl.error() == ShaderIdentityError::BackendNotRealized,
            "a software-only builtin did not refuse OpenGL");
    }

    // --- One identity, two realizations (the authored offscreen pipeline) ---
    const ShaderManifest off = sw_offscreen_recipe::offscreen_shader_manifest();
    check(off.has(ShaderId::OffscreenPipeline), "the offscreen identity is not registered");
    check(off.size() == 1, "the offscreen manifest size drifted");
    check(off.open_count() == 0, "the offscreen manifest carries a minted open slot");
    const ShaderDesc* od = off.get(ShaderId::OffscreenPipeline);
    check(od != nullptr && od->name == "offscreen_pipeline", "the offscreen identity name drifted");
    check(od != nullptr && od->module == "offscreen_pipeline", "the offscreen module name drifted");

    const auto off_sw = off.resolve(ShaderId::OffscreenPipeline, RenderBackendType::Software);
    check(static_cast<bool>(off_sw) && off_sw->has_program && off_sw->program.valid(),
        "the software offscreen resolution carries no program");
    check(static_cast<bool>(off_sw) && off_sw->module.empty(),
        "the software offscreen resolution carried a GPU module");

    const auto off_vk = off.resolve(ShaderId::OffscreenPipeline, RenderBackendType::Vulkan);
    check(static_cast<bool>(off_vk), "the Vulkan offscreen resolution failed");
    check(static_cast<bool>(off_vk) && !off_vk->has_program,
        "the Vulkan offscreen resolution carried a CPU program");
    check(static_cast<bool>(off_vk) && off_vk->module == "offscreen_pipeline",
        "the Vulkan resolution lost the authored module");
    check(static_cast<bool>(off_vk) && off_vk->entries.vs == kShaderEntryVsMain &&
        off_vk->entries.fs == kShaderEntryFsMain, "the Vulkan entry points drifted");

    // OpenGL has no realization today. It must say so, not quietly run the CPU
    // program behind the caller's back.
    const auto off_gl = off.resolve(ShaderId::OffscreenPipeline, RenderBackendType::OpenGL);
    check(!off_gl && off_gl.error() == ShaderIdentityError::BackendNotRealized,
        "OpenGL resolved a realization it does not have");

    // --- Entry points are single-sourced, not duplicated literals -----------
    check(sw_offscreen_recipe::vertex_entry == kShaderEntryVsMain,
        "the software vertex entry no longer shares the canonical constant");
    check(sw_offscreen_recipe::fragment_entry == kShaderEntryFsMain,
        "the software fragment entry no longer shares the canonical constant");
    check(od != nullptr && od->entries.vs == sw_offscreen_recipe::vertex_entry &&
        od->entries.fs == sw_offscreen_recipe::fragment_entry,
        "the manifest entry points diverge from the CPU realization");

    // --- Declared entry points are checked, preserving the old refusal law --
    const auto ok_decl = off.resolve(ShaderId::OffscreenPipeline, RenderBackendType::Software,
        ShaderEntryPoints{sw_offscreen_recipe::vertex_entry, sw_offscreen_recipe::fragment_entry, {}});
    check(static_cast<bool>(ok_decl), "declared registered entry points were refused");

    const auto bad_decl = off.resolve(ShaderId::OffscreenPipeline, RenderBackendType::Software,
        ShaderEntryPoints{"vs_not_registered", sw_offscreen_recipe::fragment_entry, {}});
    check(!bad_decl && bad_decl.error() == ShaderIdentityError::EntryPointMismatch,
        "an unknown entry name resolved");

    const auto uploaded_decl = off.resolve(ShaderId::OffscreenPipeline, RenderBackendType::Software,
        ShaderEntryPoints{kShaderEntryVsUploaded, sw_offscreen_recipe::fragment_entry, {}});
    check(!uploaded_decl && uploaded_decl.error() == ShaderIdentityError::EntryPointMismatch,
        "the uploaded-entry GPU variant was accepted as a CPU realization");

    // --- Known answers through the resolved program -------------------------
    // The identity layer must hand back a program that really is the authored
    // CPU realization, not merely something non-null.
    const FragmentIn fin{};
    const ShaderUniforms uniforms{};
    const FragmentOut frag = off_sw->program.fs(fin, uniforms);
    check(std::fabs(frag.color.r - 1.0f) < 1e-6f && std::fabs(frag.color.g - 0.25f) < 1e-6f &&
        frag.color.b < 1e-6f && std::fabs(frag.color.a - 1.0f) < 1e-6f,
        "the resolved offscreen fragment is not the authored flat color");

    ShaderVertex vin{};
    vin.position = glm::vec3(-0.5f, -0.5f, 0.0f);
    const VertexOut vert = off_sw->program.vs(vin, uniforms);
    check(std::fabs(vert.clip.x + 0.5f) < 1e-6f && std::fabs(vert.clip.y + 0.5f) < 1e-6f &&
        std::fabs(vert.clip.w - 1.0f) < 1e-6f,
        "the resolved offscreen vertex does not pass the authored position through");

    // The erased form is built *from* the concrete program, so the two cannot drift.
    const auto concrete = sw_offscreen_recipe::flat_triangle_program();
    const auto erased = make_erased_program(concrete);
    const FragmentOut c_out = concrete.fs(fin, uniforms);
    const FragmentOut e_out = erased.fs(fin, uniforms);
    check(c_out.color.r == e_out.color.r && c_out.color.g == e_out.color.g &&
        c_out.color.b == e_out.color.b,
        "the erased program drifted from the concrete one");

    // --- Caller-owned, comparable, no ambient state -------------------------
    const ShaderManifest a = sw_offscreen_recipe::offscreen_shader_manifest();
    const ShaderManifest b = sw_offscreen_recipe::offscreen_shader_manifest();
    check(a == b, "two identical builders produced different manifests");
    ShaderManifest c = a;
    check(c == a, "a copy differs from its source");
    check(c != value, "the offscreen and value manifests are not independent");

    // --- The real consumer: prepare_offscreen now enforces the identity law -
    rhi::SoftwareOffscreenExecution exec{};
    check(exec.initialize_device(), "the software offscreen device did not open");

    alignas(uint32_t) static const uint32_t fake_spirv[5] = {0x07230203u, 0x00010300u, 0u, 1u, 0u};

    RHIImageDesc target{};
    target.width = 32;
    target.height = 32;
    target.format = RHIFormat::RGBA8_UNorm;
    target.usage = RHIImageUsage_ColorAttachment | RHIImageUsage_TransferSrc;

    const char* const vs_entry = sw_offscreen_recipe::vertex_entry.data();
    const char* const fs_entry = sw_offscreen_recipe::fragment_entry.data();

    RHIGraphicsPipelineDesc pipeline{};
    pipeline.vs = {RHIShaderStage::Vertex, fake_spirv, sizeof(fake_spirv), vs_entry};
    pipeline.fs = {RHIShaderStage::Fragment, fake_spirv, sizeof(fake_spirv), fs_entry};
    pipeline.rt.has_depth = false;
    pipeline.depth = {false, false};

    const uint64_t prepared = exec.prepare_offscreen(target, pipeline);
    check(prepared != 0, "the registered entry pair did not prepare");
    if (prepared != 0)
        check(exec.offscreen_target() != 0, "a prepared target was not reported");

    exec.reset_offscreen();
    RHIGraphicsPipelineDesc unknown_entry = pipeline;
    unknown_entry.fs.entry = "fs_not_registered";
    check(exec.prepare_offscreen(target, unknown_entry) == 0,
        "an unregistered entry name prepared: the identity law is not enforced");

    // --- Verified against the authored source -------------------------------
    // The Vulkan side is only a claim until it is checked against the file that
    // is actually compiled. This is the cheap half of the plan's P2 reflection
    // idea: identity vs. source of truth, in a GPU-free gate.
#ifdef SHS_SHADER_SOURCE_DIR
    if (od != nullptr)
    {
        const std::string src = read_text_file(
            std::string(SHS_SHADER_SOURCE_DIR) + "/" + std::string(od->module) + ".slang");
        check(!src.empty(), "the registered module stem does not name a real authored source file");
        check(src.find(od->entries.vs) != std::string::npos,
            "the declared vertex entry is absent from the authored module");
        check(src.find(od->entries.fs) != std::string::npos,
            "the declared fragment entry is absent from the authored module");
        check(src.find(kShaderEntryVsUploaded) != std::string::npos,
            "the authored module lost its uploaded-entry variant");
    }
#else
    std::fprintf(stderr, "NOTE: SHS_SHADER_SOURCE_DIR undefined - authored-source verification skipped\n");
#endif

    if (failures != 0)
    {
        std::fprintf(stderr, "shader identity gate: %d check(s) failed\n", failures);
        return 1;
    }
    std::printf("shader identity gate: all checks passed\n");
    return 0;
}
