#define SDL_MAIN_HANDLED

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <shs/core/context.hpp>
#include <shs/core/units.hpp>
#include <shs/camera/camera_math.hpp>
#include <shs/camera/convention.hpp>
#include <shs/camera/light_camera.hpp>
#include <shs/app/runtime_state.hpp>
#include <shs/frame/technique_mode.hpp>
#include <shs/geometry/jolt_adapter.hpp>
#include <shs/geometry/culling_software.hpp>
#include <shs/geometry/jolt_culling.hpp>
#include <shs/geometry/jolt_debug_draw.hpp>
#include <shs/geometry/jolt_shapes.hpp>
#include <shs/geometry/scene_shape.hpp>
#include <shs/job/thread_pool_job_system.hpp>
#include <shs/job/wait_group.hpp>
#include <shs/lighting/light_culling_mode.hpp>
#include <shs/lighting/light_runtime.hpp>
#include <shs/lighting/light_set.hpp>
#include <shs/lighting/shadow_technique.hpp>
#include <shs/input/value_actions.hpp>
#include <shs/input/value_input_latch.hpp>
#include <shs/gfx/rt_shadow.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/pipeline/render_path_compiler.hpp>
#include <shs/pipeline/render_composition_presets.hpp>
#include <shs/pipeline/pass_adapters.hpp>
#include <shs/pipeline/pass_contract_registry.hpp>
#include <shs/pipeline/pluggable_pipeline.hpp>
#include <shs/pipeline/render_path_executor.hpp>
#include <shs/pipeline/render_path_pass_dispatch.hpp>
#include <shs/pipeline/render_path_presets.hpp>
#include <shs/pipeline/render_path_recipe.hpp>
#include <shs/pipeline/render_path_barrier_plan.hpp>
#include <shs/pipeline/render_path_runtime_layout.hpp>
#include <shs/pipeline/vk_render_path_pass_context.hpp>
#include <shs/pipeline/vk_standard_pass_execution.hpp>
#include <shs/pipeline/render_path_standard_pass_routing.hpp>
#include <shs/pipeline/render_technique_presets.hpp>
#include <shs/pipeline/render_path_temporal.hpp>
#include <shs/pipeline/technique_profile.hpp>
#include <shs/resources/loaders/primitive_import.hpp>
#include <shs/resources/resource_registry.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/rhi/drivers/vulkan/vk_backend.hpp>
#include <shs/rhi/drivers/vulkan/vk_cmd_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_frame_ownership.hpp>
#include <shs/rhi/drivers/vulkan/vk_memory_utils.hpp>
#include <shs/rhi/drivers/vulkan/vk_render_path_barrier_mapping.hpp>
#include <shs/rhi/drivers/vulkan/vk_render_path_descriptors.hpp>
#include <shs/rhi/drivers/vulkan/vk_render_path_temporal_resources.hpp>
#include <shs/rhi/drivers/vulkan/vk_shader_utils.hpp>
#include <shs/scene/scene_types.hpp>

namespace
{
constexpr int kDefaultW = 1280;
constexpr int kDefaultH = 720;
constexpr uint32_t kDefaultTileSize = 16;
constexpr uint32_t kMaxLightsPerTile = 128;
constexpr uint32_t kMaxLights = 768;
constexpr uint32_t kDefaultLightCount = 384;
constexpr int kSceneOccW = 320;
constexpr int kSceneOccH = 180;
constexpr int kLightOccW = 320;
constexpr int kLightOccH = 180;
constexpr float kTechniqueSwitchPeriodSec = 8.0f;
constexpr uint32_t kDefaultClusterZSlices = 16;
constexpr float kShadowNearZ = 0.05f;
constexpr float kDemoNearZ = 0.05f;
constexpr float kDemoFarZ = 180.0f;
constexpr float kDemoFloorSizeM = 64.0f * shs::units::meter;
constexpr uint32_t kSunShadowMapSize = 2048;
constexpr uint32_t kLocalShadowMapSize = 1024;
constexpr uint32_t kMaxSpotShadowMaps = 8;
constexpr uint32_t kMaxPointShadowLights = 2;
constexpr uint32_t kPointShadowFaceCount = 6;
constexpr uint32_t kMaxLocalShadowLayers = kMaxSpotShadowMaps + (kMaxPointShadowLights * kPointShadowFaceCount);
constexpr uint32_t kWorkerPoolRingSize = 2;
constexpr uint32_t kMaxGpuPassTimestampQueries = 128;
constexpr const char* kAppName = "HelloRenderingPaths";

struct Vertex
{
    glm::vec3 pos{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
};

struct DrawPush
{
    glm::mat4 model{1.0f};
    glm::vec4 base_color{1.0f};
    glm::vec4 material_params{0.0f, 0.5f, 1.0f, 0.0f}; // x: metallic, y: roughness, z: ao
};

struct ShadowPush
{
    glm::mat4 light_view_proj{1.0f};
    glm::mat4 model{1.0f};
};

struct alignas(16) CameraUBO
{
    glm::mat4 view{1.0f};
    glm::mat4 proj{1.0f};
    glm::mat4 view_proj{1.0f};
    glm::vec4 camera_pos_time{0.0f};
    glm::vec4 sun_dir_intensity{0.0f, -1.0f, 0.0f, 1.0f};
    glm::uvec4 screen_tile_lightcount{0u}; // x: width, y: height, z: tiles_x, w: light_count
    glm::uvec4 params{0u};                  // x: tiles_y, y: max_per_tile, z: tile_size, w: culling_mode
    glm::uvec4 culling_params{0u};          // x: cluster_z_slices, y: lighting_technique, z: semantic_debug_mode, w: semantic_id
    glm::vec4 depth_params{kDemoNearZ, kDemoFarZ, 0.0f, 0.0f}; // x: near, y: far
    glm::vec4 exposure_gamma{1.0f, 2.2f, 0.0f, 0.0f};
    glm::mat4 sun_shadow_view_proj{1.0f};
    glm::vec4 sun_shadow_params{1.0f, 0.0008f, 0.0015f, 2.0f}; // x: strength, y: bias_const, z: bias_slope, w: pcf_radius
    glm::vec4 sun_shadow_filter{1.0f, 1.0f, 0.0f, 0.0f};       // x: pcf_step, y: enabled
    glm::vec4 temporal_params{0.0f};                            // x: temporal-enable, y: history-valid, z: history-blend
};

struct alignas(16) ShadowLightGPU
{
    glm::mat4 light_view_proj{1.0f};
    glm::vec4 position_range{0.0f}; // xyz: light pos, w: range/far
    glm::vec4 shadow_params{0.0f};  // x: strength, y: bias_const, z: bias_slope, w: pcf_radius
    glm::uvec4 meta{0u};            // x: ShadowTechnique, y: layer base, z: reserved, w: enabled
};
static_assert(sizeof(ShadowLightGPU) % 16 == 0, "ShadowLightGPU must be std430 compatible");

struct Instance
{
    enum class MeshKind : uint8_t
    {
        Sphere = 0,
        Box = 1,
        Cone = 2,
        Capsule = 3,
        Cylinder = 4
    };

    glm::vec3 base_pos{0.0f};
    glm::vec4 base_color{1.0f};
    glm::vec3 base_rot{0.0f};
    glm::vec3 rot_speed{0.0f};
    float scale = 1.0f;
    float phase = 0.0f;
    float metallic = 0.08f;
    float roughness = 0.36f;
    float ao = 1.0f;
    MeshKind mesh_kind = MeshKind::Sphere;
};

struct GpuBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void* mapped = nullptr;
};

struct FrameResources
{
    GpuBuffer camera_buffer{};
    GpuBuffer light_buffer{};
    GpuBuffer shadow_light_buffer{};
    GpuBuffer tile_counts_buffer{};
    GpuBuffer tile_indices_buffer{};
    GpuBuffer tile_depth_ranges_buffer{};
    VkDescriptorSet global_set = VK_NULL_HANDLE;
};

struct DepthTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0;
    uint32_t h = 0;
};

struct GBufferAttachment
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
};

struct GBufferTarget
{
    std::array<GBufferAttachment, 4> colors{};
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    uint32_t w = 0;
    uint32_t h = 0;
};

struct AmbientOcclusionTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0;
    uint32_t h = 0;
};

struct PostColorTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0;
    uint32_t h = 0;
};

struct LayeredDepthTarget
{
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView sampled_view = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::vector<VkImageView> layer_views{};
    std::vector<VkFramebuffer> framebuffers{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t w = 0;
    uint32_t h = 0;
    uint32_t layers = 0;
};

struct WorkerPool
{
    std::array<VkCommandPool, kWorkerPoolRingSize> pools{VK_NULL_HANDLE, VK_NULL_HANDLE};
};

struct GpuPassTimestampSample
{
    std::string pass_id{};
    shs::PassId pass_kind = shs::PassId::Unknown;
    uint32_t begin_query = UINT32_MAX;
    uint32_t end_query = UINT32_MAX;
    bool success = false;
};

struct GpuPassTimestampFrameState
{
    std::vector<GpuPassTimestampSample> samples{};
    uint32_t query_count = 0;
    bool pending = false;
};

struct PhaseFBenchmarkConfig
{
    bool enabled = false;
    uint32_t warmup_frames = 90u;
    uint32_t sample_frames = 180u;
    bool include_post_variants = true;
    bool include_full_cycle = false;
    bool capture_snapshots = true;
    uint32_t max_entries = 0u;
    std::string output_path = "artifacts/phase_f_baseline_metrics.jsonl";
    std::string snapshot_dir = "artifacts/phase_f_snapshots";
};

enum class PhaseFBenchmarkStage : uint8_t
{
    Disabled = 0,
    Warmup = 1,
    Sample = 2,
    AwaitSnapshot = 3
};

struct PhaseFBenchmarkAccumulator
{
    uint32_t sampled_frames = 0u;
    double frame_ms_sum = 0.0;
    double frame_ms_min = std::numeric_limits<double>::max();
    double frame_ms_max = 0.0;
    double dispatch_cpu_ms_sum = 0.0;
    double gpu_ms_sum = 0.0;
    uint32_t gpu_valid_frames = 0u;
    uint32_t gpu_zero_sample_frames = 0u;
    uint64_t gpu_sample_count_sum = 0u;
    uint64_t gpu_rejected_sample_count_sum = 0u;
    uint64_t visible_lights_sum = 0u;
    uint64_t active_lights_sum = 0u;
    uint32_t gbuffer_frames = 0u;
    uint32_t ssao_frames = 0u;
    uint32_t deferred_frames = 0u;
    uint32_t taa_frames = 0u;
    uint32_t motion_frames = 0u;
    uint32_t dof_frames = 0u;

    void reset()
    {
        sampled_frames = 0u;
        frame_ms_sum = 0.0;
        frame_ms_min = std::numeric_limits<double>::max();
        frame_ms_max = 0.0;
        dispatch_cpu_ms_sum = 0.0;
        gpu_ms_sum = 0.0;
        gpu_valid_frames = 0u;
        gpu_zero_sample_frames = 0u;
        gpu_sample_count_sum = 0u;
        gpu_rejected_sample_count_sum = 0u;
        visible_lights_sum = 0u;
        active_lights_sum = 0u;
        gbuffer_frames = 0u;
        ssao_frames = 0u;
        deferred_frames = 0u;
        taa_frames = 0u;
        motion_frames = 0u;
        dof_frames = 0u;
    }
};

struct PhaseGSoakConfig
{
    bool enabled = false;
    uint32_t duration_sec = 180u;
    uint32_t cycle_frames = 240u;
    uint32_t log_interval_frames = 120u;
    uint32_t toggle_interval_cycles = 2u;
    std::string output_path = "artifacts/phase_g_soak_metrics.jsonl";
    double accept_max_avg_frame_ms = 50.0;
    uint32_t accept_max_render_target_rebuild_delta = 24u;
    uint32_t accept_max_pipeline_rebuild_delta = 24u;
    uint32_t accept_max_swapchain_generation_delta = 24u;
    uint32_t accept_max_cycle_failures = 0u;
};

struct PhaseGSoakState
{
    bool started = false;
    bool finished = false;
    uint64_t frame_counter = 0u;
    uint64_t cycles = 0u;
    uint64_t toggle_events = 0u;
    uint64_t last_cycle_frame = 0u;
    uint64_t last_log_frame = 0u;
    float elapsed_sec = 0.0f;
    uint64_t rebuild_target_start = 0u;
    uint64_t rebuild_pipeline_start = 0u;
    uint64_t swapchain_gen_start = 0u;
    uint64_t cycle_apply_failures = 0u;
    double frame_ms_sum = 0.0;
    double frame_ms_min = std::numeric_limits<double>::max();
    double frame_ms_max = 0.0;
};

struct PhaseIParityConfig
{
    bool enabled = false;
    bool include_resource_validation = true;
    std::string output_path = "artifacts/phase_i_backend_parity.jsonl";
    bool runtime_sw_execute = true;
    uint32_t runtime_warmup_frames = 2u;
    uint32_t runtime_sample_frames = 6u;
    uint32_t runtime_width = 320u;
    uint32_t runtime_height = 180u;
};

struct PhaseISoftwareRuntimeSample
{
    bool attempted = false;
    bool configured = false;
    bool executed = false;
    bool report_valid = false;
    uint32_t sampled_frames = 0u;
    double avg_frame_ms = 0.0;
    uint64_t ldr_hash = 0u;
    std::string error{};
    std::string warning{};
};

struct CompositionParityEntry
{
    size_t index = 0u;
    std::string name{};
    shs::RenderPathPreset path_preset = shs::RenderPathPreset::Forward;
    shs::RenderTechniquePreset technique_preset = shs::RenderTechniquePreset::PBR;
    shs::RenderCompositionPostStackPreset post_stack = shs::RenderCompositionPostStackPreset::Default;

    bool vk_plan_valid = false;
    bool vk_resource_valid = false;
    bool vk_barrier_valid = false;
    bool vk_valid = false;
    size_t vk_pass_count = 0u;
    size_t vk_barrier_edges = 0u;
    uint32_t vk_layout_transitions = 0u;
    size_t vk_alias_classes = 0u;
    uint32_t vk_alias_slots = 0u;
    std::string vk_plan_error{};
    std::string vk_resource_error{};
    std::string vk_barrier_error{};
    std::string vk_warning{};

    bool sw_plan_valid = false;
    bool sw_resource_valid = false;
    bool sw_barrier_valid = false;
    bool sw_valid = false;
    size_t sw_pass_count = 0u;
    std::string sw_plan_error{};
    std::string sw_resource_error{};
    std::string sw_barrier_error{};
    std::string sw_warning{};

    bool has_ssao = false;
    bool has_taa = false;
    bool has_motion = false;
    bool has_dof = false;

    PhaseISoftwareRuntimeSample sw_runtime{};
};

struct LocalShadowCaster
{
    uint32_t light_index = 0;
    shs::ShadowTechnique technique = shs::ShadowTechnique::None;
    uint32_t layer_base = 0;
    glm::vec3 position_ws{0.0f};
    float range = 1.0f;
    glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
    float outer_angle_rad = glm::radians(35.0f);
    float strength = 1.0f;
};

struct FreeCamera
{
    glm::vec3 pos{0.0f, 5.5f, -22.0f};
    float yaw = glm::half_pi<float>();
    float pitch = -0.18f;
    float move_speed = 8.0f;
    float look_speed = 0.003f;
    static constexpr float kMouseSpikeThreshold = 240.0f;
    static constexpr float kMouseDeltaClamp = 90.0f;

    void update(
        bool move_forward,
        bool move_backward,
        bool move_left,
        bool move_right,
        bool move_up,
        bool move_down,
        bool boost,
        bool left_mouse_down,
        bool right_mouse_down,
        float mouse_dx,
        float mouse_dy,
        float dt)
    {
        if (left_mouse_down || right_mouse_down)
        {
            float mdx = mouse_dx;
            float mdy = mouse_dy;
            if (std::abs(mdx) > kMouseSpikeThreshold || std::abs(mdy) > kMouseSpikeThreshold)
            {
                mdx = 0.0f;
                mdy = 0.0f;
            }
            mdx = std::clamp(mdx, -kMouseDeltaClamp, kMouseDeltaClamp);
            mdy = std::clamp(mdy, -kMouseDeltaClamp, kMouseDeltaClamp);
            yaw -= mdx * look_speed;
            pitch -= mdy * look_speed;
            pitch = std::clamp(pitch, -glm::half_pi<float>() + 0.01f, glm::half_pi<float>() - 0.01f);
        }

        const glm::vec3 fwd = shs::forward_from_yaw_pitch(yaw, pitch);
        const glm::vec3 right = shs::right_from_forward(fwd);
        const glm::vec3 up(0.0f, 1.0f, 0.0f);
        const float speed = move_speed * (boost ? 2.0f : 1.0f);
        if (move_forward) pos += fwd * speed * dt;
        if (move_backward) pos -= fwd * speed * dt;
        if (move_left) pos -= right * speed * dt;
        if (move_right) pos += right * speed * dt;
        if (move_up) pos += up * speed * dt;
        if (move_down) pos -= up * speed * dt;
    }

    glm::mat4 view_matrix() const
    {
        return shs::look_at_lh(pos, pos + shs::forward_from_yaw_pitch(yaw, pitch), glm::vec3(0.0f, 1.0f, 0.0f));
    }
};

enum class DebugVolumeMeshKind : uint8_t
{
    Sphere = 0,
    Cone = 1,
    Box = 2
};

struct LightVolumeDebugDraw
{
    DebugVolumeMeshKind mesh = DebugVolumeMeshKind::Sphere;
    glm::mat4 model{1.0f};
    glm::vec4 color{1.0f};
};

enum class VulkanCullerBackend : uint8_t
{
    GpuCompute = 0,
    Disabled = 1
};

enum class FramebufferDebugPreset : uint8_t
{
    FinalComposite = 0,
    Albedo = 1,
    Normal = 2,
    Material = 3,
    Depth = 4,
    AmbientOcclusion = 5,
    LightGrid = 6,
    LightClusters = 7,
    Shadow = 8,
    ColorHDR = 9,
    ColorLDR = 10,
    Motion = 11,
    DoFCircleOfConfusion = 12,
    DoFBlur = 13,
    DoFFactor = 14
};

const char* lighting_technique_name(shs::RenderTechniquePreset tech)
{
    return shs::render_technique_preset_name(tech);
}

const char* vulkan_culler_backend_name(VulkanCullerBackend backend)
{
    switch (backend)
    {
        case VulkanCullerBackend::GpuCompute: return "gpu";
        case VulkanCullerBackend::Disabled: return "off";
    }
    return "gpu";
}

const char* framebuffer_debug_preset_name(FramebufferDebugPreset preset)
{
    switch (preset)
    {
        case FramebufferDebugPreset::FinalComposite: return "final";
        case FramebufferDebugPreset::Albedo: return "albedo";
        case FramebufferDebugPreset::Normal: return "normal";
        case FramebufferDebugPreset::Material: return "material";
        case FramebufferDebugPreset::Depth: return "depth";
        case FramebufferDebugPreset::AmbientOcclusion: return "ao";
        case FramebufferDebugPreset::LightGrid: return "light_grid";
        case FramebufferDebugPreset::LightClusters: return "light_clusters";
        case FramebufferDebugPreset::Shadow: return "shadow";
        case FramebufferDebugPreset::ColorHDR: return "hdr";
        case FramebufferDebugPreset::ColorLDR: return "ldr";
        case FramebufferDebugPreset::Motion: return "motion";
        case FramebufferDebugPreset::DoFCircleOfConfusion: return "dof_coc";
        case FramebufferDebugPreset::DoFBlur: return "dof_blur";
        case FramebufferDebugPreset::DoFFactor: return "dof_factor";
    }
    return "final";
}

bool framebuffer_debug_preset_requires_motion_pass(FramebufferDebugPreset preset)
{
    return preset == FramebufferDebugPreset::Motion;
}

bool framebuffer_debug_preset_requires_dof_pass(FramebufferDebugPreset preset)
{
    return preset == FramebufferDebugPreset::DoFCircleOfConfusion ||
        preset == FramebufferDebugPreset::DoFBlur ||
        preset == FramebufferDebugPreset::DoFFactor;
}

uint32_t semantic_debug_mode_for_framebuffer_preset(FramebufferDebugPreset preset)
{
    switch (preset)
    {
        case FramebufferDebugPreset::FinalComposite:
            return 0u;
        case FramebufferDebugPreset::Albedo:
            return 1u;
        case FramebufferDebugPreset::Normal:
            return 2u;
        case FramebufferDebugPreset::Depth:
            return 3u;
        case FramebufferDebugPreset::Material:
            return 4u;
        case FramebufferDebugPreset::AmbientOcclusion:
            return 5u;
        case FramebufferDebugPreset::LightGrid:
            return 6u;
        case FramebufferDebugPreset::LightClusters:
            return 7u;
        case FramebufferDebugPreset::Shadow:
            return 8u;
        case FramebufferDebugPreset::ColorHDR:
            return 10u;
        case FramebufferDebugPreset::ColorLDR:
            return 11u;
        case FramebufferDebugPreset::Motion:
            return 12u;
        case FramebufferDebugPreset::DoFCircleOfConfusion:
            return 13u;
        case FramebufferDebugPreset::DoFBlur:
            return 14u;
        case FramebufferDebugPreset::DoFFactor:
            return 15u;
    }
    return 0u;
}

uint32_t semantic_debug_mode_for_semantic(shs::PassSemantic semantic)
{
    // Shared with fp_stress_scene.frag semantic debug switch.
    switch (semantic)
    {
        case shs::PassSemantic::Albedo:
            return 1u;
        case shs::PassSemantic::Normal:
            return 2u;
        case shs::PassSemantic::Depth:
        case shs::PassSemantic::HistoryDepth:
            return 3u;
        case shs::PassSemantic::Material:
            return 4u;
        case shs::PassSemantic::AmbientOcclusion:
            return 5u;
        case shs::PassSemantic::LightGrid:
        case shs::PassSemantic::LightIndexList:
            return 6u;
        case shs::PassSemantic::LightClusters:
            return 7u;
        case shs::PassSemantic::ShadowMap:
            return 8u;
        case shs::PassSemantic::ColorHDR:
            return 10u;
        case shs::PassSemantic::ColorLDR:
        case shs::PassSemantic::HistoryColor:
            return 11u;
        case shs::PassSemantic::MotionVectors:
        case shs::PassSemantic::HistoryMotion:
            return 12u;
        default:
            break;
    }
    return 0u;
}

glm::vec3 safe_perp_axis(const glm::vec3& v)
{
    if (std::abs(v.y) < 0.9f) return glm::vec3(0.0f, 1.0f, 0.0f);
    return glm::vec3(0.0f, 0.0f, 1.0f);
}

void basis_from_axis(
    const glm::vec3& axis_y,
    glm::vec3& out_x,
    glm::vec3& out_y,
    glm::vec3& out_z)
{
    out_y = shs::normalize_or(axis_y, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec3 up_hint = safe_perp_axis(out_y);
    out_x = shs::normalize_or(glm::cross(up_hint, out_y), glm::vec3(1.0f, 0.0f, 0.0f));
    out_z = shs::normalize_or(glm::cross(out_y, out_x), glm::vec3(0.0f, 0.0f, 1.0f));
}

glm::mat4 model_from_basis_and_scale(
    const glm::vec3& position,
    const glm::vec3& axis_x,
    const glm::vec3& axis_y,
    const glm::vec3& axis_z,
    const glm::vec3& scale_xyz)
{
    glm::mat4 m(1.0f);
    m[0] = glm::vec4(axis_x * scale_xyz.x, 0.0f);
    m[1] = glm::vec4(axis_y * scale_xyz.y, 0.0f);
    m[2] = glm::vec4(axis_z * scale_xyz.z, 0.0f);
    m[3] = glm::vec4(position, 1.0f);
    return m;
}

bool profile_has_pass(const shs::TechniqueProfile& profile, shs::PassId pass_id)
{
    if (!shs::pass_id_is_standard(pass_id)) return false;
    for (const auto& p : profile.passes)
    {
        if (p.pass_id == pass_id) return true;
        if (shs::parse_pass_id(p.id) == pass_id) return true;
    }
    return false;
}

class HelloRenderingPathsApp
{
public:
    ~HelloRenderingPathsApp()
    {
        cleanup();
    }

    void run()
    {
        shs::jolt::init_jolt();
        configure_phase_f_from_env();
        configure_phase_g_from_env();
        configure_phase_i_from_env();
        init_sdl();
        init_backend();
        configure_vulkan_culler_backend_from_env();
        init_jobs();
        init_scene_data();
        initialize_phase_i_parity_report();
        init_gpu_resources();
        initialize_phase_f_benchmark();
        initialize_phase_g_soak();
        print_controls();
        main_loop();
    }

    void cleanup()
    {
        if (cleaned_up_) return;
        cleaned_up_ = true;

        if (vk_) vk_->wait_idle();

        destroy_gpu_pass_timestamp_resources();
        destroy_pipelines();
        destroy_depth_target();
        destroy_gbuffer_target();
        destroy_ao_target();
        destroy_post_color_target(post_target_a_);
        destroy_post_color_target(post_target_b_);
        shs::vk_destroy_render_path_temporal_resources(vk_->device(), temporal_resources_);
        destroy_layered_depth_target(sun_shadow_target_);
        destroy_layered_depth_target(local_shadow_target_);

        destroy_worker_pools();
        if (jobs_) jobs_.reset();
        
        destroy_buffer(vertex_buffer_);
        destroy_buffer(index_buffer_);
        destroy_buffer(floor_vertex_buffer_);
        destroy_buffer(floor_index_buffer_);
        destroy_buffer(cone_vertex_buffer_);
        destroy_buffer(cone_index_buffer_);
        destroy_buffer(box_vertex_buffer_);
        destroy_buffer(box_index_buffer_);
        destroy_buffer(sphere_line_index_buffer_);
        destroy_buffer(cone_line_index_buffer_);
        destroy_buffer(box_line_index_buffer_);
        destroy_buffer(capsule_vertex_buffer_);
        destroy_buffer(capsule_index_buffer_);
        destroy_buffer(cylinder_vertex_buffer_);
        destroy_buffer(cylinder_index_buffer_);
        destroy_buffer(phase_f_snapshot_readback_buffer_);

        for (auto& fr : frame_resources_)
        {
            destroy_buffer(fr.camera_buffer);
            destroy_buffer(fr.light_buffer);
            destroy_buffer(fr.shadow_light_buffer);
            destroy_buffer(fr.tile_counts_buffer);
            destroy_buffer(fr.tile_depth_ranges_buffer);
            destroy_buffer(fr.tile_indices_buffer);
            fr.global_set = VK_NULL_HANDLE;
        }

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            if (depth_sampler_ != VK_NULL_HANDLE)
            {
                vkDestroySampler(vk_->device(), depth_sampler_, nullptr);
                depth_sampler_ = VK_NULL_HANDLE;
            }
            if (descriptor_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(vk_->device(), descriptor_pool_, nullptr);
                descriptor_pool_ = VK_NULL_HANDLE;
            }
            if (deferred_descriptor_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorPool(vk_->device(), deferred_descriptor_pool_, nullptr);
                deferred_descriptor_pool_ = VK_NULL_HANDLE;
                deferred_set_ = VK_NULL_HANDLE;
                deferred_post_a_set_ = VK_NULL_HANDLE;
                deferred_post_b_set_ = VK_NULL_HANDLE;
            }
            if (deferred_set_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_->device(), deferred_set_layout_, nullptr);
                deferred_set_layout_ = VK_NULL_HANDLE;
            }
            if (global_set_layout_ != VK_NULL_HANDLE)
            {
                vkDestroyDescriptorSetLayout(vk_->device(), global_set_layout_, nullptr);
                global_set_layout_ = VK_NULL_HANDLE;
            }
        }

        keep_.clear();
        vk_ = nullptr;

        if (win_)
        {
            SDL_DestroyWindow(win_);
            win_ = nullptr;
        }
        if (sdl_ready_)
        {
            SDL_Quit();
            sdl_ready_ = false;
        }

        if (phase_f_metrics_stream_.is_open())
        {
            phase_f_metrics_stream_.close();
        }
        if (phase_g_metrics_stream_.is_open())
        {
            phase_g_metrics_stream_.close();
        }

        shs::jolt::shutdown_jolt();
    }

private:
    void print_controls() const
    {
        std::fprintf(stderr, "\n[%s] Controls\n", kAppName);
        std::fprintf(stderr, "  Esc        : quit\n");
        std::fprintf(stderr, "  F1         : toggle recording mode (inline / MT-secondary)\n");
        std::fprintf(stderr, "  F2         : cycle rendering path (Forward/Forward+/Deferred/TiledDeferred/ClusteredForward)\n");
        std::fprintf(stderr, "  F3         : cycle composed presets ({path + technique + post-stack variant})\n");
        std::fprintf(stderr, "  F4         : cycle rendering-technique recipe (PBR/Blinn)\n");
        std::fprintf(stderr, "  F5         : cycle framebuffer debug preset (final/albedo/normal/material/depth/ao/light-grid/light-clusters/shadow/hdr/ldr/motion/dof-coc/dof-blur/dof-factor)\n");
        std::fprintf(stderr, "  Tab        : cycle rendering path (alias)\n");
        std::fprintf(stderr, "  F6         : toggle Vulkan culler backend (gpu / disabled)\n");
        std::fprintf(stderr, "  F7         : toggle light debug wireframe draw\n");
        std::fprintf(stderr, "  F8         : cycle semantic debug target from active resource plan\n");
        std::fprintf(stderr, "  F9         : toggle temporal accumulation (history blend + jitter, when TAA pass exists)\n");
        std::fprintf(stderr, "  F10        : print controls/help + composition catalog (includes VK/SW parity)\n");
        std::fprintf(stderr, "  F11        : toggle auto lighting-technique switching\n");
        std::fprintf(stderr, "  F12        : toggle directional (sun) shadow contribution\n");
        std::fprintf(stderr, "  Drag LMB/RMB: free-look camera (WSL spike-filtered)\n");
        std::fprintf(stderr, "  W/A/S/D + Q/E: move camera, Shift: boost\n");
        std::fprintf(stderr, "  1/2        : orbit radius scale -/+\n");
        std::fprintf(stderr, "  3/4        : light height bias -/+\n");
        std::fprintf(stderr, "  5/6        : light range scale -/+\n");
        std::fprintf(stderr, "  7/8        : light intensity scale -/+\n");
        std::fprintf(stderr, "  9/0        : sun shadow strength -/+ (when F12 is on)\n");
        std::fprintf(stderr, "  R          : reset light tuning\n");
        std::fprintf(stderr, "  +/-        : decrease/increase active light count\n");
        std::fprintf(stderr, "  Title bar  : shows composition, path/technique state, culling/debug stats, CPU/GPU pass timing state, rebuild counters, and frame ms\n\n");
        std::fprintf(stderr, "  Phase-I    : set `SHS_PHASE_I=1` for VK/SW parity JSONL (includes SW runtime sampling by default)\n\n");
        std::fprintf(stderr, "  Phase-F    : set `SHS_PHASE_F=1` for auto matrix benchmark -> JSONL artifacts (+ optional PPM snapshots)\n\n");
        std::fprintf(stderr, "  Phase-G    : set `SHS_PHASE_G=1` for timed soak auto-cycle -> JSONL churn/rebuild metrics + acceptance verdict\n\n");
    }

    std::vector<CompositionParityEntry> collect_composition_parity_entries(
        bool include_resource_validation = true) const
    {
        std::vector<CompositionParityEntry> out{};
        out.reserve(composition_cycle_order_.size());
        if (composition_cycle_order_.empty()) return out;

        const shs::RenderPathCompiler compiler{};
        shs::BackendCapabilities software_caps{};
        software_caps.supports_offscreen = true;
        software_caps.supports_present = false;
        const shs::RenderPathCapabilitySet software_capset =
            shs::make_render_path_capability_set(shs::RenderBackendType::Software, software_caps);

        for (size_t i = 0; i < composition_cycle_order_.size(); ++i)
        {
            const shs::RenderCompositionRecipe& c = composition_cycle_order_[i];
            const shs::RenderCompositionResolved resolved =
                shs::resolve_builtin_render_composition_recipe(
                    c,
                    shs::RenderBackendType::Vulkan,
                    "render_path_vk",
                    "render_tech_vk");

            CompositionParityEntry entry{};
            entry.index = i;
            entry.name = c.name;
            entry.path_preset = c.path_preset;
            entry.technique_preset = c.technique_preset;
            entry.post_stack = c.post_stack;

            const shs::RenderPathExecutionPlan vk_plan =
                compiler.compile(resolved.path_recipe, ctx_, &pass_contract_registry_);
            entry.vk_plan_valid = vk_plan.valid;
            entry.vk_pass_count = vk_plan.pass_chain.size();
            if (!vk_plan.errors.empty()) entry.vk_plan_error = vk_plan.errors.front();
            if (!vk_plan.warnings.empty()) entry.vk_warning = vk_plan.warnings.front();
            entry.has_ssao = shs::render_path_plan_has_pass(vk_plan, shs::PassId::SSAO);
            entry.has_taa = shs::render_path_plan_has_pass(vk_plan, shs::PassId::TAA);
            entry.has_motion = shs::render_path_plan_has_pass(vk_plan, shs::PassId::MotionBlur);
            entry.has_dof = shs::render_path_plan_has_pass(vk_plan, shs::PassId::DepthOfField);

            if (include_resource_validation)
            {
                const shs::RenderPathResourcePlan vk_resource_plan =
                    shs::compile_render_path_resource_plan(vk_plan, resolved.path_recipe, &pass_contract_registry_);
                const shs::RenderPathBarrierPlan vk_barrier_plan =
                    shs::compile_render_path_barrier_plan(vk_plan, vk_resource_plan, &pass_contract_registry_);
                entry.vk_resource_valid = vk_resource_plan.valid;
                entry.vk_barrier_valid = vk_barrier_plan.valid;
                entry.vk_barrier_edges = vk_barrier_plan.edges.size();
                entry.vk_layout_transitions = shs::render_path_barrier_layout_transition_count(vk_barrier_plan);
                entry.vk_alias_classes = vk_barrier_plan.alias_classes.size();
                entry.vk_alias_slots = shs::render_path_alias_slot_count(vk_barrier_plan);
                if (!vk_resource_plan.errors.empty()) entry.vk_resource_error = vk_resource_plan.errors.front();
                if (!vk_barrier_plan.errors.empty()) entry.vk_barrier_error = vk_barrier_plan.errors.front();
            }
            else
            {
                entry.vk_resource_valid = true;
                entry.vk_barrier_valid = true;
            }
            entry.vk_valid = entry.vk_plan_valid && entry.vk_resource_valid && entry.vk_barrier_valid;

            shs::RenderPathRecipe sw_recipe = resolved.path_recipe;
            sw_recipe.backend = shs::RenderBackendType::Software;
            sw_recipe.name = c.name + "__path_sw";
            const shs::RenderPathExecutionPlan sw_plan =
                compiler.compile(sw_recipe, software_capset, &pass_contract_registry_sw_);
            entry.sw_plan_valid = sw_plan.valid;
            entry.sw_pass_count = sw_plan.pass_chain.size();
            if (!sw_plan.errors.empty()) entry.sw_plan_error = sw_plan.errors.front();
            if (!sw_plan.warnings.empty()) entry.sw_warning = sw_plan.warnings.front();

            if (include_resource_validation)
            {
                const shs::RenderPathResourcePlan sw_resource_plan =
                    shs::compile_render_path_resource_plan(sw_plan, sw_recipe, &pass_contract_registry_sw_);
                const shs::RenderPathBarrierPlan sw_barrier_plan =
                    shs::compile_render_path_barrier_plan(sw_plan, sw_resource_plan, &pass_contract_registry_sw_);
                entry.sw_resource_valid = sw_resource_plan.valid;
                entry.sw_barrier_valid = sw_barrier_plan.valid;
                if (!sw_resource_plan.errors.empty()) entry.sw_resource_error = sw_resource_plan.errors.front();
                if (!sw_barrier_plan.errors.empty()) entry.sw_barrier_error = sw_barrier_plan.errors.front();
            }
            else
            {
                entry.sw_resource_valid = true;
                entry.sw_barrier_valid = true;
            }
            entry.sw_valid = entry.sw_plan_valid && entry.sw_resource_valid && entry.sw_barrier_valid;

            out.push_back(std::move(entry));
        }
        return out;
    }

    void print_composition_catalog() const
    {
        if (composition_cycle_order_.empty())
        {
            std::fprintf(stderr, "[render-path][composition] No registered compositions.\n");
            return;
        }

        const std::vector<CompositionParityEntry> entries = collect_composition_parity_entries(true);
        std::fprintf(
            stderr,
            "[render-path][composition] Cycle catalog (%zu entries):\n",
            entries.size());
        for (const CompositionParityEntry& e : entries)
        {
            std::fprintf(
                stderr,
                "  [%02zu] %-42s path:%-17s technique:%-7s post:%-8s bk[vk:%-7s sw:%-7s] pass[v:%2zu s:%2zu] post[s:%c t:%c m:%c d:%c] br:%zu lay:%u al:%zu/%u%s\n",
                e.index,
                e.name.c_str(),
                shs::render_path_preset_name(e.path_preset),
                shs::render_technique_preset_name(e.technique_preset),
                shs::render_composition_post_stack_preset_name(e.post_stack),
                e.vk_valid ? "ok" : "invalid",
                e.sw_valid ? "ok" : "invalid",
                e.vk_pass_count,
                e.sw_pass_count,
                e.has_ssao ? 'Y' : '-',
                e.has_taa ? 'Y' : '-',
                e.has_motion ? 'Y' : '-',
                e.has_dof ? 'Y' : '-',
                e.vk_barrier_edges,
                e.vk_layout_transitions,
                e.vk_alias_classes,
                e.vk_alias_slots,
                (e.index == active_composition_index_) ? "  <active>" : "");
            if (!e.vk_plan_error.empty())
            {
                std::fprintf(stderr, "        plan-error: %s\n", e.vk_plan_error.c_str());
            }
            if (!e.vk_resource_error.empty())
            {
                std::fprintf(stderr, "        resource-error: %s\n", e.vk_resource_error.c_str());
            }
            if (!e.vk_barrier_error.empty())
            {
                std::fprintf(stderr, "        barrier-error: %s\n", e.vk_barrier_error.c_str());
            }
            if (!e.sw_plan_error.empty())
            {
                std::fprintf(stderr, "        sw-plan-error: %s\n", e.sw_plan_error.c_str());
            }
            if (!e.sw_resource_error.empty())
            {
                std::fprintf(stderr, "        sw-resource-error: %s\n", e.sw_resource_error.c_str());
            }
            if (!e.sw_barrier_error.empty())
            {
                std::fprintf(stderr, "        sw-barrier-error: %s\n", e.sw_barrier_error.c_str());
            }
        }
    }

    static std::string json_escape(const std::string& in)
    {
        std::string out{};
        out.reserve(in.size() + 8u);
        for (unsigned char c : in)
        {
            switch (c)
            {
                case '\\': out += "\\\\"; break;
                case '"': out += "\\\""; break;
                case '\b': out += "\\b"; break;
                case '\f': out += "\\f"; break;
                case '\n': out += "\\n"; break;
                case '\r': out += "\\r"; break;
                case '\t': out += "\\t"; break;
                default:
                    if (c < 0x20u)
                    {
                        char buf[7]{};
                        std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned int>(c));
                        out += buf;
                    }
                    else
                    {
                        out.push_back(static_cast<char>(c));
                    }
                    break;
            }
        }
        return out;
    }

    static uint64_t hash_ldr_color_buffer(const shs::RT_ColorLDR& ldr)
    {
        static constexpr uint64_t kFNVOffset = 1469598103934665603ull;
        static constexpr uint64_t kFNVPrime = 1099511628211ull;
        uint64_t h = kFNVOffset;
        for (const shs::Color& px : ldr.color.data)
        {
            h ^= static_cast<uint64_t>(px.r); h *= kFNVPrime;
            h ^= static_cast<uint64_t>(px.g); h *= kFNVPrime;
            h ^= static_cast<uint64_t>(px.b); h *= kFNVPrime;
            h ^= static_cast<uint64_t>(px.a); h *= kFNVPrime;
        }
        return h;
    }

    PhaseISoftwareRuntimeSample run_phase_i_software_runtime_sample(
        const shs::RenderPathRecipe& sw_recipe,
        shs::RenderTechniquePreset technique_preset) const
    {
        PhaseISoftwareRuntimeSample out{};
        out.attempted = true;

        const uint32_t w = std::max(16u, phase_i_config_.runtime_width);
        const uint32_t h = std::max(16u, phase_i_config_.runtime_height);
        const uint32_t warmup_frames = phase_i_config_.runtime_warmup_frames;
        const uint32_t sample_frames = std::max(1u, phase_i_config_.runtime_sample_frames);
        const uint32_t total_frames = warmup_frames + sample_frames;

        auto backend_result = shs::create_render_backend("software");
        if (!backend_result.backend)
        {
            out.error = "software backend create failed";
            return out;
        }

        shs::Context sw_ctx{};
        std::vector<std::unique_ptr<shs::IRenderBackend>> keepalive{};
        keepalive.reserve(1u + backend_result.auxiliary_backends.size());
        keepalive.push_back(std::move(backend_result.backend));
        for (auto& aux : backend_result.auxiliary_backends)
        {
            if (aux) keepalive.push_back(std::move(aux));
        }
        if (keepalive.empty() || !keepalive.front())
        {
            out.error = "software backend unavailable";
            return out;
        }
        sw_ctx.set_primary_backend(keepalive.front().get());
        for (size_t i = 1u; i < keepalive.size(); ++i)
        {
            if (keepalive[i]) sw_ctx.register_backend(keepalive[i].get());
        }

        shs::ResourceRegistry resources{};
        shs::RTRegistry rtr{};
        shs::PluggablePipeline pipeline{};
        pipeline.set_strict_graph_validation(true);

        shs::RT_ShadowDepth shadow_rt{256, 256};
        shs::RT_ColorHDR hdr_rt{static_cast<int>(w), static_cast<int>(h)};
        shs::RT_ColorDepthMotion motion_rt{static_cast<int>(w), static_cast<int>(h), kDemoNearZ, kDemoFarZ};
        shs::RT_ColorLDR ldr_rt{static_cast<int>(w), static_cast<int>(h)};
        shs::RT_ColorLDR shafts_tmp_rt{static_cast<int>(w), static_cast<int>(h)};
        shs::RT_ColorLDR motion_blur_tmp_rt{static_cast<int>(w), static_cast<int>(h)};

        const shs::RT_Shadow rt_shadow_h = rtr.reg<shs::RT_Shadow>(&shadow_rt);
        const shs::RTHandle rt_hdr_h = rtr.reg<shs::RTHandle>(&hdr_rt);
        const shs::RT_Motion rt_motion_h = rtr.reg<shs::RT_Motion>(&motion_rt);
        const shs::RTHandle rt_ldr_h = rtr.reg<shs::RTHandle>(&ldr_rt);
        const shs::RTHandle rt_shafts_tmp_h = rtr.reg<shs::RTHandle>(&shafts_tmp_rt);
        const shs::RTHandle rt_motion_blur_tmp_h = rtr.reg<shs::RTHandle>(&motion_blur_tmp_rt);

        const shs::PassFactoryRegistry pass_registry = shs::make_standard_pass_factory_registry(
            rt_shadow_h,
            rt_hdr_h,
            rt_motion_h,
            rt_ldr_h,
            rt_shafts_tmp_h,
            rt_motion_blur_tmp_h);

        const shs::RenderPathCompiler compiler{};
        shs::BackendCapabilities software_caps{};
        software_caps.supports_offscreen = true;
        software_caps.supports_present = false;
        const shs::RenderPathCapabilitySet software_capset =
            shs::make_render_path_capability_set(shs::RenderBackendType::Software, software_caps);
        const shs::RenderPathExecutionPlan sw_plan =
            compiler.compile(sw_recipe, software_capset, &pass_registry);
        if (!sw_plan.valid)
        {
            out.error = sw_plan.errors.empty() ? "software plan invalid" : sw_plan.errors.front();
            return out;
        }

        std::vector<std::string> missing{};
        out.configured = pipeline.configure_from_render_path_plan(pass_registry, sw_plan, &missing);
        if (!out.configured)
        {
            out.error = missing.empty() ? "software pipeline configure failed" : ("missing pass: " + missing.front());
            return out;
        }

        shs::Scene scene{};
        scene.resources = &resources;
        scene.cam.pos = glm::vec3(0.0f, 2.2f, 6.5f);
        scene.cam.target = glm::vec3(0.0f, 0.6f, 0.0f);
        scene.cam.up = glm::vec3(0.0f, 1.0f, 0.0f);
        scene.cam.znear = kDemoNearZ;
        scene.cam.zfar = kDemoFarZ;
        scene.cam.fov_y_radians = glm::radians(60.0f);
        scene.cam.view = shs::look_at_lh(scene.cam.pos, scene.cam.target, scene.cam.up);
        scene.cam.proj = shs::perspective_lh_no(
            scene.cam.fov_y_radians,
            static_cast<float>(w) / static_cast<float>(h),
            scene.cam.znear,
            scene.cam.zfar);
        scene.cam.viewproj = scene.cam.proj * scene.cam.view;
        scene.cam.prev_viewproj = scene.cam.viewproj;
        scene.sun.dir_ws = glm::normalize(glm::vec3(-0.35f, -1.0f, -0.25f));
        scene.sun.color = glm::vec3(1.0f, 0.97f, 0.92f);
        scene.sun.intensity = 2.0f;

        shs::FrameParams fp{};
        fp.w = static_cast<int>(w);
        fp.h = static_cast<int>(h);
        fp.dt = 1.0f / 60.0f;
        fp.time = 0.0f;
        fp.debug_view = shs::DebugViewMode::Final;
        fp.cull_mode = shs::CullMode::Back;
        fp.technique.mode = sw_recipe.technique_mode;
        fp.technique.active_modes_mask = shs::technique_mode_mask_all();
        fp.pass.shadow.enable = sw_recipe.runtime_defaults.enable_shadows;
        fp.enable_shadows = fp.pass.shadow.enable;
        fp.hybrid.allow_cross_backend_passes = false;
        fp.hybrid.strict_backend_availability = true;
        fp.hybrid.emulate_vulkan_runtime = false;
        const shs::RenderTechniqueRecipe tech_recipe =
            shs::make_builtin_render_technique_recipe(technique_preset, "phase_i_sw_runtime");
        shs::apply_render_technique_recipe_to_frame_params(tech_recipe, fp);

        double sampled_frame_ms_sum = 0.0;
        uint32_t sampled_count = 0u;
        using clock = std::chrono::steady_clock;
        for (uint32_t frame = 0u; frame < total_frames; ++frame)
        {
            const auto t0 = clock::now();
            pipeline.execute(sw_ctx, scene, fp, rtr);
            const auto t1 = clock::now();
            const auto& report = pipeline.execution_report();
            if (!report.valid)
            {
                out.report_valid = false;
                out.error = report.errors.empty() ? "software execution report invalid" : report.errors.front();
                return out;
            }
            if (!report.warnings.empty() && out.warning.empty())
            {
                out.warning = report.warnings.front();
            }
            out.report_valid = true;
            out.executed = true;

            if (frame >= warmup_frames)
            {
                sampled_frame_ms_sum +=
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                ++sampled_count;
            }

            fp.time += fp.dt;
            scene.cam.prev_viewproj = scene.cam.viewproj;
        }

        out.sampled_frames = sampled_count;
        out.avg_frame_ms = (sampled_count > 0u)
            ? (sampled_frame_ms_sum / static_cast<double>(sampled_count))
            : 0.0;
        out.ldr_hash = hash_ldr_color_buffer(ldr_rt);
        return out;
    }

    static bool parse_env_bool(const char* value, bool fallback)
    {
        if (!value || *value == '\0') return fallback;
        std::string v(value);
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (v == "1" || v == "true" || v == "on" || v == "yes") return true;
        if (v == "0" || v == "false" || v == "off" || v == "no") return false;
        return fallback;
    }

    static uint32_t parse_env_u32(const char* value, uint32_t fallback, uint32_t min_value = 1u)
    {
        if (!value || *value == '\0') return fallback;
        char* end = nullptr;
        const unsigned long parsed = std::strtoul(value, &end, 10);
        if (end == value) return fallback;
        const uint32_t out = static_cast<uint32_t>(std::min<unsigned long>(
            parsed,
            static_cast<unsigned long>(std::numeric_limits<uint32_t>::max())));
        return std::max(min_value, out);
    }

    static double parse_env_f64(const char* value, double fallback, double min_value = 0.0)
    {
        if (!value || *value == '\0') return fallback;
        char* end = nullptr;
        const double parsed = std::strtod(value, &end);
        if (end == value || !std::isfinite(parsed)) return fallback;
        return std::max(min_value, parsed);
    }

    static std::string sanitize_file_component(const std::string& in)
    {
        std::string out{};
        out.reserve(in.size());
        for (char c : in)
        {
            if ((c >= 'a' && c <= 'z') ||
                (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') ||
                c == '-' || c == '_' || c == '.')
            {
                out.push_back(c);
            }
            else
            {
                out.push_back('_');
            }
        }
        if (out.empty()) out = "composition";
        return out;
    }

    static double safe_div(double numerator, uint32_t denominator)
    {
        return (denominator > 0u) ? (numerator / static_cast<double>(denominator)) : 0.0;
    }

    void configure_phase_i_from_env()
    {
        const char* phase_i_env = std::getenv("SHS_PHASE_I");
        const char* phase_i_enabled_env = std::getenv("SHS_PHASE_I_ENABLED");
        phase_i_config_.enabled = parse_env_bool(phase_i_enabled_env, parse_env_bool(phase_i_env, false));
        if (!phase_i_config_.enabled) return;

        phase_i_config_.include_resource_validation = parse_env_bool(
            std::getenv("SHS_PHASE_I_INCLUDE_RESOURCE_VALIDATION"),
            phase_i_config_.include_resource_validation);
        phase_i_config_.runtime_sw_execute = parse_env_bool(
            std::getenv("SHS_PHASE_I_RUNTIME_SW"),
            phase_i_config_.runtime_sw_execute);
        phase_i_config_.runtime_warmup_frames = parse_env_u32(
            std::getenv("SHS_PHASE_I_RUNTIME_WARMUP_FRAMES"),
            phase_i_config_.runtime_warmup_frames,
            0u);
        phase_i_config_.runtime_sample_frames = parse_env_u32(
            std::getenv("SHS_PHASE_I_RUNTIME_SAMPLE_FRAMES"),
            phase_i_config_.runtime_sample_frames,
            1u);
        phase_i_config_.runtime_width = parse_env_u32(
            std::getenv("SHS_PHASE_I_RUNTIME_WIDTH"),
            phase_i_config_.runtime_width,
            16u);
        phase_i_config_.runtime_height = parse_env_u32(
            std::getenv("SHS_PHASE_I_RUNTIME_HEIGHT"),
            phase_i_config_.runtime_height,
            16u);
        if (const char* output_env = std::getenv("SHS_PHASE_I_OUTPUT"))
        {
            if (*output_env != '\0') phase_i_config_.output_path = output_env;
        }
    }

    void initialize_phase_i_parity_report() const
    {
        if (!phase_i_config_.enabled) return;

        std::vector<CompositionParityEntry> entries =
            collect_composition_parity_entries(phase_i_config_.include_resource_validation);
        if (entries.empty())
        {
            std::fprintf(stderr, "[phase-i] No compositions available for parity report.\n");
            return;
        }

        std::filesystem::path output_path(phase_i_config_.output_path);
        if (output_path.has_parent_path())
        {
            std::error_code ec{};
            std::filesystem::create_directories(output_path.parent_path(), ec);
        }

        std::ofstream out(phase_i_config_.output_path, std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            std::fprintf(stderr, "[phase-i] Failed to open output: %s\n", phase_i_config_.output_path.c_str());
            return;
        }

        size_t full_parity_count = 0u;
        size_t vk_only_count = 0u;
        size_t sw_only_count = 0u;
        size_t sw_runtime_attempted = 0u;
        size_t sw_runtime_executed = 0u;
        double sw_runtime_frame_ms_sum = 0.0;
        uint32_t sw_runtime_frame_ms_count = 0u;
        for (const CompositionParityEntry& e : entries)
        {
            if (e.vk_valid && e.sw_valid) ++full_parity_count;
            else if (e.vk_valid && !e.sw_valid) ++vk_only_count;
            else if (!e.vk_valid && e.sw_valid) ++sw_only_count;
        }

        out << "{"
            << "\"event\":\"phase_i_begin\","
            << "\"composition_count\":" << entries.size() << ","
            << "\"resource_validation\":" << (phase_i_config_.include_resource_validation ? "true" : "false") << ","
            << "\"runtime_sw_execute\":" << (phase_i_config_.runtime_sw_execute ? "true" : "false")
            << "}\n";

        for (CompositionParityEntry& e : entries)
        {
            if (phase_i_config_.runtime_sw_execute &&
                e.sw_plan_valid &&
                e.index < composition_cycle_order_.size())
            {
                const shs::RenderCompositionRecipe& c = composition_cycle_order_[e.index];
                const shs::RenderCompositionResolved sw_resolved =
                    shs::resolve_builtin_render_composition_recipe(
                        c,
                        shs::RenderBackendType::Software,
                        "render_path_sw",
                        "render_tech_sw");
                shs::RenderPathRecipe sw_recipe = sw_resolved.path_recipe;
                sw_recipe.backend = shs::RenderBackendType::Software;
                sw_recipe.name = c.name + "__phase_i_runtime_sw";
                e.sw_runtime = run_phase_i_software_runtime_sample(sw_recipe, c.technique_preset);
                if (e.sw_runtime.attempted) ++sw_runtime_attempted;
                if (e.sw_runtime.executed)
                {
                    ++sw_runtime_executed;
                    sw_runtime_frame_ms_sum += e.sw_runtime.avg_frame_ms;
                    ++sw_runtime_frame_ms_count;
                }
            }

            out << "{";
            out << "\"event\":\"phase_i_composition\",";
            out << "\"index\":" << e.index << ",";
            out << "\"composition\":\"" << json_escape(e.name) << "\",";
            out << "\"path\":\"" << shs::render_path_preset_name(e.path_preset) << "\",";
            out << "\"technique\":\"" << shs::render_technique_preset_name(e.technique_preset) << "\",";
            out << "\"post_stack\":\"" << shs::render_composition_post_stack_preset_name(e.post_stack) << "\",";
            out << "\"vk_valid\":" << (e.vk_valid ? "true" : "false") << ",";
            out << "\"vk_plan_valid\":" << (e.vk_plan_valid ? "true" : "false") << ",";
            out << "\"vk_resource_valid\":" << (e.vk_resource_valid ? "true" : "false") << ",";
            out << "\"vk_barrier_valid\":" << (e.vk_barrier_valid ? "true" : "false") << ",";
            out << "\"vk_pass_count\":" << e.vk_pass_count << ",";
            out << "\"vk_barrier_edges\":" << e.vk_barrier_edges << ",";
            out << "\"vk_layout_transitions\":" << e.vk_layout_transitions << ",";
            out << "\"vk_alias_classes\":" << e.vk_alias_classes << ",";
            out << "\"vk_alias_slots\":" << e.vk_alias_slots << ",";
            out << "\"sw_valid\":" << (e.sw_valid ? "true" : "false") << ",";
            out << "\"sw_plan_valid\":" << (e.sw_plan_valid ? "true" : "false") << ",";
            out << "\"sw_resource_valid\":" << (e.sw_resource_valid ? "true" : "false") << ",";
            out << "\"sw_barrier_valid\":" << (e.sw_barrier_valid ? "true" : "false") << ",";
            out << "\"sw_pass_count\":" << e.sw_pass_count << ",";
            out << "\"post_ssao\":" << (e.has_ssao ? "true" : "false") << ",";
            out << "\"post_taa\":" << (e.has_taa ? "true" : "false") << ",";
            out << "\"post_motion\":" << (e.has_motion ? "true" : "false") << ",";
            out << "\"post_dof\":" << (e.has_dof ? "true" : "false") << ",";
            out << "\"vk_plan_error\":\"" << json_escape(e.vk_plan_error) << "\",";
            out << "\"vk_resource_error\":\"" << json_escape(e.vk_resource_error) << "\",";
            out << "\"vk_barrier_error\":\"" << json_escape(e.vk_barrier_error) << "\",";
            out << "\"vk_warning\":\"" << json_escape(e.vk_warning) << "\",";
            out << "\"sw_plan_error\":\"" << json_escape(e.sw_plan_error) << "\",";
            out << "\"sw_resource_error\":\"" << json_escape(e.sw_resource_error) << "\",";
            out << "\"sw_barrier_error\":\"" << json_escape(e.sw_barrier_error) << "\",";
            out << "\"sw_warning\":\"" << json_escape(e.sw_warning) << "\",";
            out << "\"sw_runtime_attempted\":" << (e.sw_runtime.attempted ? "true" : "false") << ",";
            out << "\"sw_runtime_configured\":" << (e.sw_runtime.configured ? "true" : "false") << ",";
            out << "\"sw_runtime_executed\":" << (e.sw_runtime.executed ? "true" : "false") << ",";
            out << "\"sw_runtime_report_valid\":" << (e.sw_runtime.report_valid ? "true" : "false") << ",";
            out << "\"sw_runtime_sampled_frames\":" << e.sw_runtime.sampled_frames << ",";
            out << "\"sw_runtime_avg_frame_ms\":" << e.sw_runtime.avg_frame_ms << ",";
            out << "\"sw_runtime_ldr_hash\":" << e.sw_runtime.ldr_hash << ",";
            out << "\"sw_runtime_error\":\"" << json_escape(e.sw_runtime.error) << "\",";
            out << "\"sw_runtime_warning\":\"" << json_escape(e.sw_runtime.warning) << "\"";
            out << "}\n";
        }

        const double sw_runtime_avg_frame_ms =
            (sw_runtime_frame_ms_count > 0u)
                ? (sw_runtime_frame_ms_sum / static_cast<double>(sw_runtime_frame_ms_count))
                : 0.0;
        out << "{"
            << "\"event\":\"phase_i_end\","
            << "\"composition_count\":" << entries.size() << ","
            << "\"full_parity\":" << full_parity_count << ","
            << "\"vk_only\":" << vk_only_count << ","
            << "\"sw_only\":" << sw_only_count << ","
            << "\"sw_runtime_attempted\":" << sw_runtime_attempted << ","
            << "\"sw_runtime_executed\":" << sw_runtime_executed << ","
            << "\"sw_runtime_avg_frame_ms\":" << sw_runtime_avg_frame_ms
            << "}\n";
        out.flush();

        std::fprintf(
            stderr,
            "[phase-i] Parity report written: %s (full:%zu/%zu, vk-only:%zu, sw-only:%zu, sw-runtime:%zu/%zu avg:%.2fms)\n",
            phase_i_config_.output_path.c_str(),
            full_parity_count,
            entries.size(),
            vk_only_count,
            sw_only_count,
            sw_runtime_executed,
            sw_runtime_attempted,
            sw_runtime_avg_frame_ms);
    }

    void configure_phase_f_from_env()
    {
        const char* phase_f_env = std::getenv("SHS_PHASE_F");
        const char* phase_f_enabled_env = std::getenv("SHS_PHASE_F_ENABLED");
        phase_f_config_.enabled = parse_env_bool(phase_f_enabled_env, parse_env_bool(phase_f_env, false));
        if (!phase_f_config_.enabled) return;

        phase_f_config_.warmup_frames = parse_env_u32(
            std::getenv("SHS_PHASE_F_WARMUP_FRAMES"),
            phase_f_config_.warmup_frames,
            1u);
        phase_f_config_.sample_frames = parse_env_u32(
            std::getenv("SHS_PHASE_F_SAMPLE_FRAMES"),
            phase_f_config_.sample_frames,
            1u);
        phase_f_config_.include_post_variants = parse_env_bool(
            std::getenv("SHS_PHASE_F_INCLUDE_POST_VARIANTS"),
            phase_f_config_.include_post_variants);
        phase_f_config_.include_full_cycle = parse_env_bool(
            std::getenv("SHS_PHASE_F_FULL_CYCLE"),
            phase_f_config_.include_full_cycle);
        phase_f_config_.capture_snapshots = parse_env_bool(
            std::getenv("SHS_PHASE_F_CAPTURE_SNAPSHOTS"),
            phase_f_config_.capture_snapshots);
        phase_f_config_.max_entries = parse_env_u32(
            std::getenv("SHS_PHASE_F_MAX_ENTRIES"),
            phase_f_config_.max_entries,
            0u);

        if (const char* output_env = std::getenv("SHS_PHASE_F_OUTPUT"))
        {
            if (*output_env != '\0') phase_f_config_.output_path = output_env;
        }
        if (const char* snapshot_env = std::getenv("SHS_PHASE_F_SNAPSHOT_DIR"))
        {
            if (*snapshot_env != '\0') phase_f_config_.snapshot_dir = snapshot_env;
        }
    }

    void configure_phase_g_from_env()
    {
        const char* phase_g_env = std::getenv("SHS_PHASE_G");
        const char* phase_g_enabled_env = std::getenv("SHS_PHASE_G_ENABLED");
        phase_g_config_.enabled = parse_env_bool(phase_g_enabled_env, parse_env_bool(phase_g_env, false));
        if (!phase_g_config_.enabled) return;

        phase_g_config_.duration_sec = parse_env_u32(
            std::getenv("SHS_PHASE_G_DURATION_SEC"),
            phase_g_config_.duration_sec,
            1u);
        phase_g_config_.cycle_frames = parse_env_u32(
            std::getenv("SHS_PHASE_G_CYCLE_FRAMES"),
            phase_g_config_.cycle_frames,
            1u);
        phase_g_config_.log_interval_frames = parse_env_u32(
            std::getenv("SHS_PHASE_G_LOG_INTERVAL_FRAMES"),
            phase_g_config_.log_interval_frames,
            1u);
        phase_g_config_.toggle_interval_cycles = parse_env_u32(
            std::getenv("SHS_PHASE_G_TOGGLE_INTERVAL_CYCLES"),
            phase_g_config_.toggle_interval_cycles,
            1u);
        phase_g_config_.accept_max_avg_frame_ms = parse_env_f64(
            std::getenv("SHS_PHASE_G_ACCEPT_MAX_AVG_FRAME_MS"),
            phase_g_config_.accept_max_avg_frame_ms,
            0.1);
        phase_g_config_.accept_max_render_target_rebuild_delta = parse_env_u32(
            std::getenv("SHS_PHASE_G_ACCEPT_MAX_RT_REBUILDS"),
            phase_g_config_.accept_max_render_target_rebuild_delta,
            0u);
        phase_g_config_.accept_max_pipeline_rebuild_delta = parse_env_u32(
            std::getenv("SHS_PHASE_G_ACCEPT_MAX_PIPELINE_REBUILDS"),
            phase_g_config_.accept_max_pipeline_rebuild_delta,
            0u);
        phase_g_config_.accept_max_swapchain_generation_delta = parse_env_u32(
            std::getenv("SHS_PHASE_G_ACCEPT_MAX_SWAPCHAIN_GENERATION"),
            phase_g_config_.accept_max_swapchain_generation_delta,
            0u);
        phase_g_config_.accept_max_cycle_failures = parse_env_u32(
            std::getenv("SHS_PHASE_G_ACCEPT_MAX_CYCLE_FAILURES"),
            phase_g_config_.accept_max_cycle_failures,
            0u);
        if (const char* output_env = std::getenv("SHS_PHASE_G_OUTPUT"))
        {
            if (*output_env != '\0') phase_g_config_.output_path = output_env;
        }
    }

    void phase_g_write_json_line(const std::string& line)
    {
        if (!phase_g_metrics_stream_.is_open()) return;
        phase_g_metrics_stream_ << line << "\n";
        phase_g_metrics_stream_.flush();
    }

    void phase_g_emit_cycle_event(const shs::RenderCompositionRecipe& c, float frame_ms, float ema_ms)
    {
        std::string line = "{";
        line += "\"event\":\"phase_g_cycle\",";
        line += "\"cycle\":" + std::to_string(phase_g_state_.cycles) + ",";
        line += "\"frame\":" + std::to_string(phase_g_state_.frame_counter) + ",";
        line += "\"elapsed_sec\":" + std::to_string(phase_g_state_.elapsed_sec) + ",";
        line += "\"composition\":\"" + c.name + "\",";
        line += "\"path\":\"" + std::string(shs::render_path_preset_name(c.path_preset)) + "\",";
        line += "\"technique\":\"" + std::string(shs::render_technique_preset_name(c.technique_preset)) + "\",";
        line += "\"post_stack\":\"" + std::string(shs::render_composition_post_stack_preset_name(c.post_stack)) + "\",";
        line += "\"frame_ms\":" + std::to_string(frame_ms) + ",";
        line += "\"ema_ms\":" + std::to_string(ema_ms) + ",";
        line += "\"rebuild_target\":" + std::to_string(render_target_rebuild_count_) + ",";
        line += "\"rebuild_pipeline\":" + std::to_string(pipeline_rebuild_count_) + ",";
        line += "\"swapchain_generation\":" + std::to_string(swapchain_generation_change_count_);
        line += "}";
        phase_g_write_json_line(line);
    }

    void phase_g_emit_heartbeat(float frame_ms, float ema_ms)
    {
        const char* composition_name = active_composition_recipe_.name.empty()
            ? "n/a"
            : active_composition_recipe_.name.c_str();
        std::string line = "{";
        line += "\"event\":\"phase_g_heartbeat\",";
        line += "\"frame\":" + std::to_string(phase_g_state_.frame_counter) + ",";
        line += "\"elapsed_sec\":" + std::to_string(phase_g_state_.elapsed_sec) + ",";
        line += "\"composition\":\"" + std::string(composition_name) + "\",";
        line += "\"frame_ms\":" + std::to_string(frame_ms) + ",";
        line += "\"ema_ms\":" + std::to_string(ema_ms) + ",";
        line += "\"visible_lights\":" + std::to_string(visible_light_count_) + ",";
        line += "\"active_lights\":" + std::to_string(active_light_count_) + ",";
        line += "\"rebuild_target\":" + std::to_string(render_target_rebuild_count_) + ",";
        line += "\"rebuild_pipeline\":" + std::to_string(pipeline_rebuild_count_) + ",";
        line += "\"swapchain_generation\":" + std::to_string(swapchain_generation_change_count_) + ",";
        line += "\"gpu_timing_valid\":" + std::string(gpu_pass_timing_valid_ ? "true" : "false");
        line += "}";
        phase_g_write_json_line(line);
    }

    void phase_g_emit_end_event()
    {
        const uint64_t delta_rt_rebuild =
            render_target_rebuild_count_ - phase_g_state_.rebuild_target_start;
        const uint64_t delta_pipeline_rebuild =
            pipeline_rebuild_count_ - phase_g_state_.rebuild_pipeline_start;
        const uint64_t delta_swapchain_gen =
            swapchain_generation_change_count_ - phase_g_state_.swapchain_gen_start;
        const double avg_frame_ms =
            (phase_g_state_.frame_counter > 0u)
                ? (phase_g_state_.frame_ms_sum / static_cast<double>(phase_g_state_.frame_counter))
                : 0.0;
        const double min_frame_ms =
            (phase_g_state_.frame_counter > 0u) ? phase_g_state_.frame_ms_min : 0.0;
        const double max_frame_ms =
            (phase_g_state_.frame_counter > 0u) ? phase_g_state_.frame_ms_max : 0.0;
        const bool accept =
            avg_frame_ms <= phase_g_config_.accept_max_avg_frame_ms &&
            delta_rt_rebuild <= phase_g_config_.accept_max_render_target_rebuild_delta &&
            delta_pipeline_rebuild <= phase_g_config_.accept_max_pipeline_rebuild_delta &&
            delta_swapchain_gen <= phase_g_config_.accept_max_swapchain_generation_delta &&
            phase_g_state_.cycle_apply_failures <= phase_g_config_.accept_max_cycle_failures;

        std::string line = "{";
        line += "\"event\":\"phase_g_end\",";
        line += "\"elapsed_sec\":" + std::to_string(phase_g_state_.elapsed_sec) + ",";
        line += "\"frames\":" + std::to_string(phase_g_state_.frame_counter) + ",";
        line += "\"cycles\":" + std::to_string(phase_g_state_.cycles) + ",";
        line += "\"toggle_events\":" + std::to_string(phase_g_state_.toggle_events) + ",";
        line += "\"avg_frame_ms\":" + std::to_string(avg_frame_ms) + ",";
        line += "\"min_frame_ms\":" + std::to_string(min_frame_ms) + ",";
        line += "\"max_frame_ms\":" + std::to_string(max_frame_ms) + ",";
        line += "\"cycle_apply_failures\":" + std::to_string(phase_g_state_.cycle_apply_failures) + ",";
        line += "\"delta_render_target_rebuild\":" + std::to_string(delta_rt_rebuild) + ",";
        line += "\"delta_pipeline_rebuild\":" + std::to_string(delta_pipeline_rebuild) + ",";
        line += "\"delta_swapchain_generation\":" + std::to_string(delta_swapchain_gen) + ",";
        line += "\"accept\":" + std::string(accept ? "true" : "false");
        line += "}";
        phase_g_write_json_line(line);

        std::fprintf(
            stderr,
            "[phase-g] acceptance: %s (avg:%.2fms <= %.2f, rt:%llu <= %u, pipe:%llu <= %u, swap:%llu <= %u, cycle_fail:%llu <= %u)\n",
            accept ? "PASS" : "FAIL",
            avg_frame_ms,
            phase_g_config_.accept_max_avg_frame_ms,
            static_cast<unsigned long long>(delta_rt_rebuild),
            phase_g_config_.accept_max_render_target_rebuild_delta,
            static_cast<unsigned long long>(delta_pipeline_rebuild),
            phase_g_config_.accept_max_pipeline_rebuild_delta,
            static_cast<unsigned long long>(delta_swapchain_gen),
            phase_g_config_.accept_max_swapchain_generation_delta,
            static_cast<unsigned long long>(phase_g_state_.cycle_apply_failures),
            phase_g_config_.accept_max_cycle_failures);
    }

    void phase_g_apply_toggle_perturbation()
    {
        if (active_taa_pass_enabled())
        {
            temporal_settings_.accumulation_enabled = !temporal_settings_.accumulation_enabled;
            temporal_settings_.jitter_enabled = temporal_settings_.accumulation_enabled;
        }
        cycle_framebuffer_debug_target();
        show_light_volumes_debug_ = !show_light_volumes_debug_;
        ++phase_g_state_.toggle_events;
    }

    void initialize_phase_g_soak()
    {
        phase_g_state_ = PhaseGSoakState{};
        if (!phase_g_config_.enabled) return;
        if (phase_f_config_.enabled)
        {
            std::fprintf(stderr, "[phase-g] Disabled because Phase-F mode is active.\n");
            phase_g_config_.enabled = false;
            return;
        }
        if (composition_cycle_order_.empty())
        {
            std::fprintf(stderr, "[phase-g] No compositions available. Disabling soak mode.\n");
            phase_g_config_.enabled = false;
            return;
        }

        // Add Modern-Extreme coverage variant
        shs::RenderCompositionRecipe extreme{};
        extreme.name = "composition_modern_extreme";
        extreme.path_preset = shs::RenderPathPreset::ClusteredForward;
        extreme.technique_preset = shs::RenderTechniquePreset::PBR;
        extreme.post_stack = shs::RenderCompositionPostStackPreset::Full;
        composition_cycle_order_.push_back(std::move(extreme));

        std::filesystem::path output_path(phase_g_config_.output_path);
        if (output_path.has_parent_path())
        {
            std::error_code ec{};
            std::filesystem::create_directories(output_path.parent_path(), ec);
        }
        phase_g_metrics_stream_.open(phase_g_config_.output_path, std::ios::out | std::ios::trunc);
        if (!phase_g_metrics_stream_.is_open())
        {
            std::fprintf(stderr, "[phase-g] Failed to open output: %s\n", phase_g_config_.output_path.c_str());
            phase_g_config_.enabled = false;
            return;
        }

        phase_g_state_.started = true;
        phase_g_state_.rebuild_target_start = render_target_rebuild_count_;
        phase_g_state_.rebuild_pipeline_start = pipeline_rebuild_count_;
        phase_g_state_.swapchain_gen_start = swapchain_generation_change_count_;

        std::string line = "{";
        line += "\"event\":\"phase_g_begin\",";
        line += "\"duration_sec\":" + std::to_string(phase_g_config_.duration_sec) + ",";
        line += "\"cycle_frames\":" + std::to_string(phase_g_config_.cycle_frames) + ",";
        line += "\"log_interval_frames\":" + std::to_string(phase_g_config_.log_interval_frames) + ",";
        line += "\"toggle_interval_cycles\":" + std::to_string(phase_g_config_.toggle_interval_cycles) + ",";
        line += "\"accept_max_avg_frame_ms\":" + std::to_string(phase_g_config_.accept_max_avg_frame_ms) + ",";
        line += "\"accept_max_rt_rebuilds\":" + std::to_string(phase_g_config_.accept_max_render_target_rebuild_delta) + ",";
        line += "\"accept_max_pipeline_rebuilds\":" + std::to_string(phase_g_config_.accept_max_pipeline_rebuild_delta) + ",";
        line += "\"accept_max_swapchain_generation\":" + std::to_string(phase_g_config_.accept_max_swapchain_generation_delta) + ",";
        line += "\"accept_max_cycle_failures\":" + std::to_string(phase_g_config_.accept_max_cycle_failures) + ",";
        line += "\"composition_count\":" + std::to_string(composition_cycle_order_.size());
        line += "}";
        phase_g_write_json_line(line);

        std::fprintf(
            stderr,
            "[phase-g] Started soak mode (%us, cycle:%u frames, log:%u frames) -> %s\n",
            phase_g_config_.duration_sec,
            phase_g_config_.cycle_frames,
            phase_g_config_.log_interval_frames,
            phase_g_config_.output_path.c_str());
    }

    void phase_g_step_after_frame(float frame_ms, float ema_ms, float dt)
    {
        if (!phase_g_config_.enabled || !phase_g_state_.started || phase_g_state_.finished) return;
        phase_g_state_.elapsed_sec += dt;
        ++phase_g_state_.frame_counter;
        phase_g_state_.frame_ms_sum += frame_ms;
        phase_g_state_.frame_ms_min = std::min(phase_g_state_.frame_ms_min, static_cast<double>(frame_ms));
        phase_g_state_.frame_ms_max = std::max(phase_g_state_.frame_ms_max, static_cast<double>(frame_ms));

        if ((phase_g_state_.frame_counter - phase_g_state_.last_log_frame) >= phase_g_config_.log_interval_frames)
        {
            phase_g_state_.last_log_frame = phase_g_state_.frame_counter;
            phase_g_emit_heartbeat(frame_ms, ema_ms);
        }

        if ((phase_g_state_.frame_counter - phase_g_state_.last_cycle_frame) >= phase_g_config_.cycle_frames)
        {
            phase_g_state_.last_cycle_frame = phase_g_state_.frame_counter;
            if (!composition_cycle_order_.empty())
            {
                const size_t next = (active_composition_index_ + 1u) % composition_cycle_order_.size();
                if (apply_render_composition_by_index(next))
                {
                    ++phase_g_state_.cycles;
                    phase_g_emit_cycle_event(active_composition_recipe_, frame_ms, ema_ms);
                    if (phase_g_config_.toggle_interval_cycles > 0u &&
                        (phase_g_state_.cycles % phase_g_config_.toggle_interval_cycles) == 0u)
                    {
                        phase_g_apply_toggle_perturbation();
                    }
                }
                else
                {
                    ++phase_g_state_.cycle_apply_failures;
                }
            }
        }

        if (phase_g_state_.elapsed_sec >= static_cast<float>(phase_g_config_.duration_sec))
        {
            phase_g_emit_end_event();
            phase_g_state_.finished = true;
            std::fprintf(stderr, "[phase-g] Soak run complete. Results: %s\n", phase_g_config_.output_path.c_str());
            running_ = false;
        }
    }

    void apply_phase_g_camera_tour(float dt, float t)
    {
        (void)dt; (void)t;
        const float elapsed = phase_g_state_.elapsed_sec;
        const float total_duration = static_cast<float>(std::max(1u, phase_g_config_.duration_sec));
        const float t_total = elapsed / total_duration;

        // Add stochastic "randomness" via overlapping sine waves to make it feel natural/dynamic
        const float noise_x = std::sin(elapsed * 1.15f) * 0.6f + std::sin(elapsed * 2.45f) * 0.25f;
        const float noise_y = std::sin(elapsed * 0.85f) * 0.45f + std::sin(elapsed * 1.65f) * 0.35f;
        const float noise_z = std::cos(elapsed * 1.45f) * 0.55f + std::cos(elapsed * 2.15f) * 0.25f;

        if (t_total < 0.35f)
        {
            // Phase 1: Orbit (Outer ring) + Noise
            const float orbit_speed = 0.5f;
            const float radius = 22.0f + noise_x * 2.0f;
            const float height = 7.0f + std::sin(elapsed * 0.3f) * 4.0f + noise_y;
            runtime_state_.camera.pos.x = std::cos(elapsed * orbit_speed) * radius;
            runtime_state_.camera.pos.z = std::sin(elapsed * orbit_speed) * radius;
            runtime_state_.camera.pos.y = height;
            runtime_state_.camera.yaw = elapsed * orbit_speed + glm::pi<float>() + noise_z * 0.05f;
            runtime_state_.camera.pitch = -0.25f + noise_x * 0.02f;
        }
        else if (t_total < 0.70f)
        {
            // Phase 2: High-speed Fly-through (occlusion stress) + Random Jitter
            const float t_phase = (t_total - 0.35f) / 0.35f;
            const float z_pos = glm::mix(-40.0f, 40.0f, t_phase);
            runtime_state_.camera.pos = glm::vec3(std::sin(elapsed * 0.8f) * 12.0f + noise_x * 3.0f, 4.5f + noise_y, z_pos);
            runtime_state_.camera.yaw = glm::half_pi<float>() + noise_z * 0.12f;
            runtime_state_.camera.pitch = std::sin(elapsed * 1.25f) * 0.18f + noise_x * 0.05f;
        }
        else
        {
            // Phase 3: Vertical & Tile Stress + Stochastic Wander
            const float t_phase = (t_total - 0.70f) / 0.30f;
            runtime_state_.camera.pos = glm::vec3(8.0f + noise_x * 2.0f, 2.0f + t_phase * 12.0f + noise_y, -8.0f + noise_z * 2.0f);
            runtime_state_.camera.yaw = glm::radians(225.0f) + noise_x * 0.25f;
            runtime_state_.camera.pitch = glm::sin(elapsed * 0.5f) * 0.5f - 0.2f + noise_z * 0.15f;
        }
    }

    size_t find_composition_index_exact(
        shs::RenderPathPreset path_preset,
        shs::RenderTechniquePreset technique_preset,
        shs::RenderCompositionPostStackPreset post_stack,
        bool* found = nullptr) const
    {
        for (size_t i = 0; i < composition_cycle_order_.size(); ++i)
        {
            const auto& c = composition_cycle_order_[i];
            if (c.path_preset == path_preset &&
                c.technique_preset == technique_preset &&
                c.post_stack == post_stack)
            {
                if (found) *found = true;
                return i;
            }
        }
        if (found) *found = false;
        return 0u;
    }

    void append_phase_f_plan_entry(std::vector<size_t>& out, size_t composition_index)
    {
        if (composition_index >= composition_cycle_order_.size()) return;
        if (std::find(out.begin(), out.end(), composition_index) != out.end()) return;
        out.push_back(composition_index);
    }

    std::vector<size_t> build_phase_f_plan() const
    {
        std::vector<size_t> out{};
        out.reserve(composition_cycle_order_.size());

        const auto& path_order = shs::default_render_path_preset_order();
        const auto& tech_order = shs::default_render_technique_preset_order();
        for (const shs::RenderPathPreset path : path_order)
        {
            for (const shs::RenderTechniquePreset tech : tech_order)
            {
                bool found = false;
                const size_t idx = find_composition_index_exact(
                    path,
                    tech,
                    shs::RenderCompositionPostStackPreset::Default,
                    &found);
                if (found)
                {
                    out.push_back(idx);
                }
            }
        }

        if (phase_f_config_.include_post_variants)
        {
            const auto append_if_present = [&](shs::RenderPathPreset path,
                                               shs::RenderTechniquePreset tech,
                                               shs::RenderCompositionPostStackPreset post) {
                bool found = false;
                const size_t idx = find_composition_index_exact(path, tech, post, &found);
                if (found && std::find(out.begin(), out.end(), idx) == out.end())
                {
                    out.push_back(idx);
                }
            };
            append_if_present(
                shs::RenderPathPreset::ForwardPlus,
                shs::RenderTechniquePreset::PBR,
                shs::RenderCompositionPostStackPreset::Minimal);
            append_if_present(
                shs::RenderPathPreset::Deferred,
                shs::RenderTechniquePreset::PBR,
                shs::RenderCompositionPostStackPreset::Temporal);
            append_if_present(
                shs::RenderPathPreset::Deferred,
                shs::RenderTechniquePreset::PBR,
                shs::RenderCompositionPostStackPreset::Full);
            append_if_present(
                shs::RenderPathPreset::Deferred,
                shs::RenderTechniquePreset::BlinnPhong,
                shs::RenderCompositionPostStackPreset::Full);
            append_if_present(
                shs::RenderPathPreset::TiledDeferred,
                shs::RenderTechniquePreset::PBR,
                shs::RenderCompositionPostStackPreset::Full);
        }

        if (phase_f_config_.include_full_cycle)
        {
            for (size_t i = 0; i < composition_cycle_order_.size(); ++i)
            {
                if (std::find(out.begin(), out.end(), i) == out.end())
                {
                    out.push_back(i);
                }
            }
        }

        if (phase_f_config_.max_entries > 0u && out.size() > phase_f_config_.max_entries)
        {
            out.resize(phase_f_config_.max_entries);
        }
        return out;
    }

    void phase_f_write_json_line(const std::string& line)
    {
        if (!phase_f_metrics_stream_.is_open()) return;
        phase_f_metrics_stream_ << line << "\n";
        phase_f_metrics_stream_.flush();
    }

    std::string phase_f_snapshot_path_for_entry(size_t entry_slot, const shs::RenderCompositionRecipe& composition) const
    {
        const std::string safe_name = sanitize_file_component(composition.name);
        return phase_f_config_.snapshot_dir + "/"
            + std::to_string(entry_slot + 1u) + "_" + safe_name + ".ppm";
    }

    bool phase_f_swapchain_snapshot_supported_format(VkFormat format) const
    {
        switch (format)
        {
            case VK_FORMAT_B8G8R8A8_UNORM:
            case VK_FORMAT_B8G8R8A8_SRGB:
            case VK_FORMAT_R8G8B8A8_UNORM:
            case VK_FORMAT_R8G8B8A8_SRGB:
                return true;
            default:
                break;
        }
        return false;
    }

    void phase_f_begin_entry(size_t entry_slot, size_t composition_index)
    {
        phase_f_active_entry_slot_ = entry_slot;
        phase_f_active_composition_index_ = composition_index;
        phase_f_stage_ = PhaseFBenchmarkStage::Warmup;
        phase_f_stage_frame_counter_ = 0u;
        phase_f_accumulator_.reset();
        phase_f_snapshot_request_armed_ = false;
        phase_f_snapshot_copy_submitted_ = false;
        phase_f_snapshot_completed_ = false;
        phase_f_snapshot_failed_ = false;
        phase_f_snapshot_path_.clear();
        phase_f_rebuild_target_start_ = render_target_rebuild_count_;
        phase_f_rebuild_pipeline_start_ = pipeline_rebuild_count_;
        phase_f_swapchain_gen_start_ = swapchain_generation_change_count_;
    }

    void phase_f_finish_and_exit()
    {
        if (phase_f_finished_) return;
        phase_f_finished_ = true;
        phase_f_stage_ = PhaseFBenchmarkStage::Disabled;
        phase_f_write_json_line(
            "{\"event\":\"phase_f_end\",\"entries_processed\":" +
            std::to_string(phase_f_entries_processed_) + "}");
        std::fprintf(stderr, "[phase-f] Baseline run complete. Results: %s\n", phase_f_config_.output_path.c_str());
        running_ = false;
    }

    void phase_f_advance_entry()
    {
        ++phase_f_entries_processed_;
        const size_t next = phase_f_active_entry_slot_ + 1u;
        if (next >= phase_f_plan_indices_.size())
        {
            phase_f_finish_and_exit();
            return;
        }
        const size_t next_index = phase_f_plan_indices_[next];
        if (!apply_render_composition_by_index(next_index))
        {
            std::fprintf(stderr, "[phase-f] Failed to apply composition index %zu\n", next_index);
            phase_f_finish_and_exit();
            return;
        }
        phase_f_begin_entry(next, next_index);
        const shs::RenderCompositionRecipe& c = composition_cycle_order_[next_index];
        std::fprintf(
            stderr,
            "[phase-f] Entry %zu/%zu warmup:%u sample:%u | %s\n",
            next + 1u,
            phase_f_plan_indices_.size(),
            phase_f_config_.warmup_frames,
            phase_f_config_.sample_frames,
            c.name.c_str());
    }

    void phase_f_emit_sample_result(float ema_ms)
    {
        if (phase_f_active_composition_index_ >= composition_cycle_order_.size()) return;
        const shs::RenderCompositionRecipe& c = composition_cycle_order_[phase_f_active_composition_index_];
        const uint32_t sampled = std::max(1u, phase_f_accumulator_.sampled_frames);
        const double avg_frame_ms = safe_div(phase_f_accumulator_.frame_ms_sum, sampled);
        const double avg_dispatch_ms = safe_div(phase_f_accumulator_.dispatch_cpu_ms_sum, sampled);
        const double avg_gpu_ms = safe_div(phase_f_accumulator_.gpu_ms_sum, phase_f_accumulator_.gpu_valid_frames);
        const double avg_visible_lights = safe_div(static_cast<double>(phase_f_accumulator_.visible_lights_sum), sampled);
        const double avg_active_lights = safe_div(static_cast<double>(phase_f_accumulator_.active_lights_sum), sampled);
        const uint64_t delta_target_rebuild = render_target_rebuild_count_ - phase_f_rebuild_target_start_;
        const uint64_t delta_pipeline_rebuild = pipeline_rebuild_count_ - phase_f_rebuild_pipeline_start_;
        const uint64_t delta_swapchain_gen = swapchain_generation_change_count_ - phase_f_swapchain_gen_start_;

        std::string line = "{";
        line += "\"event\":\"composition_sample\",";
        line += "\"entry\":" + std::to_string(phase_f_active_entry_slot_ + 1u) + ",";
        line += "\"composition\":\"" + c.name + "\",";
        line += "\"path\":\"" + std::string(shs::render_path_preset_name(c.path_preset)) + "\",";
        line += "\"technique\":\"" + std::string(shs::render_technique_preset_name(c.technique_preset)) + "\",";
        line += "\"post_stack\":\"" + std::string(shs::render_composition_post_stack_preset_name(c.post_stack)) + "\",";
        line += "\"sampled_frames\":" + std::to_string(sampled) + ",";
        line += "\"ema_frame_ms\":" + std::to_string(ema_ms) + ",";
        line += "\"avg_frame_ms\":" + std::to_string(avg_frame_ms) + ",";
        line += "\"min_frame_ms\":" + std::to_string(phase_f_accumulator_.frame_ms_min) + ",";
        line += "\"max_frame_ms\":" + std::to_string(phase_f_accumulator_.frame_ms_max) + ",";
        line += "\"avg_dispatch_cpu_ms\":" + std::to_string(avg_dispatch_ms) + ",";
        line += "\"avg_gpu_ms\":" + std::to_string(avg_gpu_ms) + ",";
        line += "\"gpu_valid_frames\":" + std::to_string(phase_f_accumulator_.gpu_valid_frames) + ",";
        line += "\"gpu_zero_sample_frames\":" + std::to_string(phase_f_accumulator_.gpu_zero_sample_frames) + ",";
        line += "\"gpu_sample_count_sum\":" + std::to_string(phase_f_accumulator_.gpu_sample_count_sum) + ",";
        line += "\"gpu_rejected_sample_count_sum\":" + std::to_string(phase_f_accumulator_.gpu_rejected_sample_count_sum) + ",";
        line += "\"avg_visible_lights\":" + std::to_string(avg_visible_lights) + ",";
        line += "\"avg_active_lights\":" + std::to_string(avg_active_lights) + ",";
        line += "\"gbuffer_ratio\":" + std::to_string(safe_div(phase_f_accumulator_.gbuffer_frames, sampled)) + ",";
        line += "\"ssao_ratio\":" + std::to_string(safe_div(phase_f_accumulator_.ssao_frames, sampled)) + ",";
        line += "\"deferred_ratio\":" + std::to_string(safe_div(phase_f_accumulator_.deferred_frames, sampled)) + ",";
        line += "\"taa_ratio\":" + std::to_string(safe_div(phase_f_accumulator_.taa_frames, sampled)) + ",";
        line += "\"motion_ratio\":" + std::to_string(safe_div(phase_f_accumulator_.motion_frames, sampled)) + ",";
        line += "\"dof_ratio\":" + std::to_string(safe_div(phase_f_accumulator_.dof_frames, sampled)) + ",";
        line += "\"delta_render_target_rebuild\":" + std::to_string(delta_target_rebuild) + ",";
        line += "\"delta_pipeline_rebuild\":" + std::to_string(delta_pipeline_rebuild) + ",";
        line += "\"delta_swapchain_generation\":" + std::to_string(delta_swapchain_gen) + ",";
        line += "\"snapshot\":\"" + (phase_f_snapshot_path_.empty() ? std::string("") : phase_f_snapshot_path_) + "\"";
        line += "}";
        phase_f_write_json_line(line);
    }

    void phase_f_step_after_frame(float frame_ms, float ema_ms)
    {
        if (!phase_f_config_.enabled || phase_f_finished_) return;
        if (phase_f_stage_ == PhaseFBenchmarkStage::Disabled) return;

        if (phase_f_stage_ == PhaseFBenchmarkStage::Warmup)
        {
            ++phase_f_stage_frame_counter_;
            if (phase_f_stage_frame_counter_ >= phase_f_config_.warmup_frames)
            {
                phase_f_stage_ = PhaseFBenchmarkStage::Sample;
                phase_f_stage_frame_counter_ = 0u;
                phase_f_accumulator_.reset();
            }
            return;
        }

        if (phase_f_stage_ == PhaseFBenchmarkStage::Sample)
        {
            phase_f_accumulator_.sampled_frames++;
            phase_f_accumulator_.frame_ms_sum += frame_ms;
            phase_f_accumulator_.frame_ms_min = std::min(phase_f_accumulator_.frame_ms_min, static_cast<double>(frame_ms));
            phase_f_accumulator_.frame_ms_max = std::max(phase_f_accumulator_.frame_ms_max, static_cast<double>(frame_ms));
            phase_f_accumulator_.dispatch_cpu_ms_sum += dispatch_total_cpu_ms_;
            phase_f_accumulator_.visible_lights_sum += visible_light_count_;
            phase_f_accumulator_.active_lights_sum += active_light_count_;
            phase_f_accumulator_.gbuffer_frames += frame_gbuffer_pass_executed_ ? 1u : 0u;
            phase_f_accumulator_.ssao_frames += frame_ssao_pass_executed_ ? 1u : 0u;
            phase_f_accumulator_.deferred_frames += frame_deferred_lighting_pass_executed_ ? 1u : 0u;
            phase_f_accumulator_.taa_frames += frame_taa_pass_executed_ ? 1u : 0u;
            phase_f_accumulator_.motion_frames += frame_motion_blur_pass_executed_ ? 1u : 0u;
            phase_f_accumulator_.dof_frames += frame_depth_of_field_pass_executed_ ? 1u : 0u;
            phase_f_accumulator_.gpu_sample_count_sum += gpu_pass_sample_count_;
            phase_f_accumulator_.gpu_rejected_sample_count_sum += gpu_pass_rejected_sample_count_;
            if (gpu_pass_timing_valid_)
            {
                phase_f_accumulator_.gpu_ms_sum += gpu_pass_total_ms_;
                phase_f_accumulator_.gpu_valid_frames++;
            }
            if (gpu_pass_sample_count_ == 0u)
            {
                phase_f_accumulator_.gpu_zero_sample_frames++;
            }

            ++phase_f_stage_frame_counter_;
            if (phase_f_stage_frame_counter_ >= phase_f_config_.sample_frames)
            {
                phase_f_emit_sample_result(ema_ms);
                phase_f_stage_frame_counter_ = 0u;

                if (phase_f_config_.capture_snapshots)
                {
                    const shs::RenderCompositionRecipe& c = composition_cycle_order_[phase_f_active_composition_index_];
                    phase_f_snapshot_path_ = phase_f_snapshot_path_for_entry(phase_f_active_entry_slot_, c);
                    phase_f_snapshot_request_armed_ = true;
                    phase_f_snapshot_completed_ = false;
                    phase_f_snapshot_failed_ = false;
                    phase_f_stage_ = PhaseFBenchmarkStage::AwaitSnapshot;
                    return;
                }

                phase_f_advance_entry();
            }
            return;
        }

        if (phase_f_stage_ == PhaseFBenchmarkStage::AwaitSnapshot)
        {
            if (phase_f_snapshot_completed_ || phase_f_snapshot_failed_ || !phase_f_snapshot_request_armed_)
            {
                phase_f_advance_entry();
            }
        }
    }

    void initialize_phase_f_benchmark()
    {
        phase_f_finished_ = false;
        phase_f_stage_ = PhaseFBenchmarkStage::Disabled;
        phase_f_plan_indices_.clear();
        if (!phase_f_config_.enabled) return;

        phase_f_plan_indices_ = build_phase_f_plan();
        if (phase_f_plan_indices_.empty())
        {
            std::fprintf(stderr, "[phase-f] No compositions available. Disabling benchmark mode.\n");
            phase_f_config_.enabled = false;
            return;
        }

        std::filesystem::path output_path(phase_f_config_.output_path);
        if (output_path.has_parent_path())
        {
            std::error_code ec{};
            std::filesystem::create_directories(output_path.parent_path(), ec);
        }
        if (phase_f_config_.capture_snapshots && !phase_f_config_.snapshot_dir.empty())
        {
            std::error_code ec{};
            std::filesystem::create_directories(std::filesystem::path(phase_f_config_.snapshot_dir), ec);
        }

        phase_f_metrics_stream_.open(phase_f_config_.output_path, std::ios::out | std::ios::trunc);
        if (!phase_f_metrics_stream_.is_open())
        {
            std::fprintf(stderr, "[phase-f] Failed to open output: %s\n", phase_f_config_.output_path.c_str());
            phase_f_config_.enabled = false;
            return;
        }

        phase_f_write_json_line(
            "{\"event\":\"phase_f_begin\",\"entries\":" + std::to_string(phase_f_plan_indices_.size()) +
            ",\"warmup_frames\":" + std::to_string(phase_f_config_.warmup_frames) +
            ",\"sample_frames\":" + std::to_string(phase_f_config_.sample_frames) +
            ",\"capture_snapshots\":" + std::string(phase_f_config_.capture_snapshots ? "true" : "false") + "}");

        phase_f_entries_processed_ = 0u;
        const size_t first_index = phase_f_plan_indices_.front();
        if (!apply_render_composition_by_index(first_index))
        {
            std::fprintf(stderr, "[phase-f] Failed to apply first composition index %zu\n", first_index);
            phase_f_config_.enabled = false;
            return;
        }
        phase_f_begin_entry(0u, first_index);
        const shs::RenderCompositionRecipe& c = composition_cycle_order_[first_index];
        std::fprintf(
            stderr,
            "[phase-f] Started benchmark (%zu entries) -> %s | warmup:%u sample:%u\n",
            phase_f_plan_indices_.size(),
            c.name.c_str(),
            phase_f_config_.warmup_frames,
            phase_f_config_.sample_frames);
    }

    void init_sdl()
    {
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
        {
            throw std::runtime_error(std::string("SDL_Init failed: ") + SDL_GetError());
        }
        sdl_ready_ = true;

        win_ = SDL_CreateWindow(
            kAppName,
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            kDefaultW,
            kDefaultH,
            SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE
        );
        if (!win_)
        {
            throw std::runtime_error(std::string("SDL_CreateWindow failed: ") + SDL_GetError());
        }
    }

    void init_backend()
    {
        shs::RenderBackendCreateResult created = shs::create_render_backend(shs::RenderBackendType::Vulkan);
        if (!created.note.empty()) std::fprintf(stderr, "[shs] %s\n", created.note.c_str());
        if (!created.backend) throw std::runtime_error("Backend factory did not return a backend");

        keep_.push_back(std::move(created.backend));
        for (auto& aux : created.auxiliary_backends)
        {
            if (aux) keep_.push_back(std::move(aux));
        }
        for (const auto& b : keep_)
        {
            ctx_.register_backend(b.get());
        }

        if (created.active != shs::RenderBackendType::Vulkan)
        {
            throw std::runtime_error("Vulkan backend is not active");
        }

        vk_ = dynamic_cast<shs::VulkanRenderBackend*>(ctx_.backend(shs::RenderBackendType::Vulkan));
        if (!vk_)
        {
            throw std::runtime_error("Failed to acquire Vulkan backend instance");
        }

        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            dw = kDefaultW;
            dh = kDefaultH;
        }

        shs::VulkanRenderBackend::InitDesc init{};
        init.window = win_;
        init.width = dw;
        init.height = dh;
        init.enable_validation = true;
        init.app_name = kAppName;
        if (!vk_->init(init))
        {
            throw std::runtime_error("Vulkan backend init_sdl failed");
        }

        ctx_.set_primary_backend(vk_);
        std::fprintf(stderr, "[shs] active backend: %s\n", ctx_.active_backend_name());
    }

    void init_jobs()
    {
        const unsigned hc = std::max(1u, std::thread::hardware_concurrency());
        worker_count_ = std::clamp<uint32_t>(hc, 1u, 8u);
        jobs_ = std::make_unique<shs::ThreadPoolJobSystem>(worker_count_);
    }

    static shs::AABB compute_local_aabb_from_positions(const std::vector<glm::vec3>& positions)
    {
        shs::AABB out{};
        if (positions.empty())
        {
            out.minv = glm::vec3(-0.5f);
            out.maxv = glm::vec3(0.5f);
            return out;
        }
        for (const glm::vec3& p : positions) out.expand(p);
        return out;
    }

    static shs::AABB compute_local_aabb_from_vertices(const std::vector<Vertex>& vertices)
    {
        shs::AABB out{};
        if (vertices.empty())
        {
            out.minv = glm::vec3(-0.5f);
            out.maxv = glm::vec3(0.5f);
            return out;
        }
        for (const Vertex& v : vertices) out.expand(v.pos);
        return out;
    }

    static void make_tessellated_floor_geometry(
        float half_extent,
        int subdivisions,
        std::vector<Vertex>& out_vertices,
        std::vector<uint32_t>& out_indices)
    {
        const int div = std::max(1, subdivisions);
        const int verts_per_row = div + 1;
        const float full = std::max(half_extent, 1.0f) * 2.0f;
        const float step = full / static_cast<float>(div);

        out_vertices.clear();
        out_indices.clear();
        out_vertices.reserve(static_cast<size_t>(verts_per_row) * static_cast<size_t>(verts_per_row));
        out_indices.reserve(static_cast<size_t>(div) * static_cast<size_t>(div) * 6u);

        for (int z = 0; z <= div; ++z)
        {
            for (int x = 0; x <= div; ++x)
            {
                const float px = -half_extent + static_cast<float>(x) * step;
                const float pz = -half_extent + static_cast<float>(z) * step;
                Vertex v{};
                v.pos = glm::vec3(px, 0.0f, pz);
                v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
                out_vertices.push_back(v);
            }
        }

        const auto idx_of = [verts_per_row](int x, int z) -> uint32_t {
            return static_cast<uint32_t>(z * verts_per_row + x);
        };

        for (int z = 0; z < div; ++z)
        {
            for (int x = 0; x < div; ++x)
            {
                const uint32_t i00 = idx_of(x + 0, z + 0);
                const uint32_t i10 = idx_of(x + 1, z + 0);
                const uint32_t i01 = idx_of(x + 0, z + 1);
                const uint32_t i11 = idx_of(x + 1, z + 1);

                out_indices.push_back(i00);
                out_indices.push_back(i10);
                out_indices.push_back(i11);

                out_indices.push_back(i00);
                out_indices.push_back(i11);
                out_indices.push_back(i01);
            }
        }
    }

    static shs::DebugMesh make_debug_mesh_from_vertex_index_data(
        const std::vector<Vertex>& verts,
        const std::vector<uint32_t>& indices)
    {
        shs::DebugMesh mesh{};
        mesh.vertices.reserve(verts.size());
        for (const Vertex& v : verts)
        {
            mesh.vertices.push_back(v.pos);
        }
        mesh.indices = indices;
        return mesh;
    }

    static std::vector<uint32_t> make_line_indices_from_triangles(const std::vector<uint32_t>& tri_indices)
    {
        std::vector<uint32_t> out{};
        out.reserve((tri_indices.size() / 3u) * 6u);
        for (size_t i = 0; i + 2 < tri_indices.size(); i += 3)
        {
            const uint32_t a = tri_indices[i + 0];
            const uint32_t b = tri_indices[i + 1];
            const uint32_t c = tri_indices[i + 2];
            out.push_back(a); out.push_back(b);
            out.push_back(b); out.push_back(c);
            out.push_back(c); out.push_back(a);
        }
        return out;
    }

    static std::vector<Vertex> make_vertices_with_normals_from_debug_mesh(const shs::DebugMesh& mesh)
    {
        std::vector<Vertex> verts(mesh.vertices.size());
        for (size_t i = 0; i < mesh.vertices.size(); ++i)
        {
            verts[i].pos = mesh.vertices[i];
            verts[i].normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }

        for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3)
        {
            const uint32_t i0 = mesh.indices[i + 0];
            const uint32_t i1 = mesh.indices[i + 1];
            const uint32_t i2 = mesh.indices[i + 2];
            if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;

            const glm::vec3 p0 = verts[i0].pos;
            const glm::vec3 p1 = verts[i1].pos;
            const glm::vec3 p2 = verts[i2].pos;
            glm::vec3 n = glm::cross(p2 - p0, p1 - p0);
            const float n2 = glm::dot(n, n);
            if (n2 <= 1e-12f) n = glm::vec3(0.0f, 1.0f, 0.0f);
            else n *= (1.0f / std::sqrt(n2));

            verts[i0].normal += n;
            verts[i1].normal += n;
            verts[i2].normal += n;
        }

        for (auto& v : verts)
        {
            const float n2 = glm::dot(v.normal, v.normal);
            if (n2 <= 1e-12f) v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            else v.normal *= (1.0f / std::sqrt(n2));
        }
        return verts;
    }

    const shs::AABB& local_aabb_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_local_aabb_;
            case Instance::MeshKind::Cone: return cone_local_aabb_;
            case Instance::MeshKind::Capsule: return capsule_local_aabb_;
            case Instance::MeshKind::Cylinder: return cylinder_local_aabb_;
            case Instance::MeshKind::Sphere:
            default: return sphere_local_aabb_;
        }
    }

    const shs::Sphere& local_bound_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_local_bound_;
            case Instance::MeshKind::Cone: return cone_local_bound_;
            case Instance::MeshKind::Capsule: return capsule_local_bound_;
            case Instance::MeshKind::Cylinder: return cylinder_local_bound_;
            case Instance::MeshKind::Sphere:
            default: return sphere_local_bound_;
        }
    }

    const JPH::ShapeRefC& cull_shape_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_shape_jolt_;
            case Instance::MeshKind::Cone: return cone_shape_jolt_;
            case Instance::MeshKind::Capsule: return capsule_shape_jolt_;
            case Instance::MeshKind::Cylinder: return cylinder_shape_jolt_;
            case Instance::MeshKind::Sphere:
            default: return sphere_shape_jolt_;
        }
    }

    const shs::DebugMesh& occluder_mesh_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_occluder_mesh_;
            case Instance::MeshKind::Cone: return cone_occluder_mesh_;
            case Instance::MeshKind::Capsule: return capsule_occluder_mesh_;
            case Instance::MeshKind::Cylinder: return cylinder_occluder_mesh_;
            case Instance::MeshKind::Sphere:
            default: return sphere_occluder_mesh_;
        }
    }

    const GpuBuffer& vertex_buffer_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_vertex_buffer_;
            case Instance::MeshKind::Cone: return cone_vertex_buffer_;
            case Instance::MeshKind::Capsule: return capsule_vertex_buffer_;
            case Instance::MeshKind::Cylinder: return cylinder_vertex_buffer_;
            case Instance::MeshKind::Sphere:
            default: return vertex_buffer_;
        }
    }

    const GpuBuffer& index_buffer_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return box_index_buffer_;
            case Instance::MeshKind::Cone: return cone_index_buffer_;
            case Instance::MeshKind::Capsule: return capsule_index_buffer_;
            case Instance::MeshKind::Cylinder: return cylinder_index_buffer_;
            case Instance::MeshKind::Sphere:
            default: return index_buffer_;
        }
    }

    uint32_t index_count_for_mesh(Instance::MeshKind kind) const
    {
        switch (kind)
        {
            case Instance::MeshKind::Box: return static_cast<uint32_t>(box_indices_.size());
            case Instance::MeshKind::Cone: return static_cast<uint32_t>(cone_indices_.size());
            case Instance::MeshKind::Capsule: return static_cast<uint32_t>(capsule_indices_.size());
            case Instance::MeshKind::Cylinder: return static_cast<uint32_t>(cylinder_indices_.size());
            case Instance::MeshKind::Sphere:
            default: return static_cast<uint32_t>(indices_.size());
        }
    }

    void init_scene_data()
    {
        shs::ResourceRegistry resources{};
        const shs::MeshAssetHandle sphere_h = shs::import_sphere_primitive(resources, shs::SphereDesc{0.5f, 18, 12}, "fplus_sphere");
        const shs::MeshAssetHandle cone_h = shs::import_cone_primitive(resources, shs::ConeDesc{1.0f, 1.0f, 20, 1, false}, "fplus_light_cone");
        const shs::MeshAssetHandle box_h = shs::import_box_primitive(resources, shs::BoxDesc{glm::vec3(1.0f), 1, 1, 1}, "fplus_light_box");

        const shs::MeshData* sphere_mesh = resources.get_mesh(sphere_h);
        if (!sphere_mesh || sphere_mesh->empty())
        {
            throw std::runtime_error("Failed to generate sphere primitive mesh");
        }
        const shs::MeshData* cone_mesh = resources.get_mesh(cone_h);
        if (!cone_mesh || cone_mesh->empty())
        {
            throw std::runtime_error("Failed to generate cone primitive mesh");
        }
        const shs::MeshData* box_mesh = resources.get_mesh(box_h);
        if (!box_mesh || box_mesh->empty())
        {
            throw std::runtime_error("Failed to generate box primitive mesh");
        }

        const JPH::ShapeRefC capsule_debug_shape = shs::jolt::make_capsule(0.92f, 0.42f);
        const JPH::ShapeRefC cylinder_debug_shape = shs::jolt::make_cylinder(0.90f, 0.46f);
        const shs::DebugMesh capsule_debug_mesh = shs::debug_mesh_from_shape(*capsule_debug_shape, JPH::Mat44::sIdentity());
        const shs::DebugMesh cylinder_debug_mesh = shs::debug_mesh_from_shape(*cylinder_debug_shape, JPH::Mat44::sIdentity());
        if (capsule_debug_mesh.vertices.empty() || capsule_debug_mesh.indices.empty())
        {
            throw std::runtime_error("Failed to build capsule debug mesh");
        }
        if (cylinder_debug_mesh.vertices.empty() || cylinder_debug_mesh.indices.empty())
        {
            throw std::runtime_error("Failed to build cylinder debug mesh");
        }

        sphere_local_aabb_ = compute_local_aabb_from_positions(sphere_mesh->positions);
        make_tessellated_floor_geometry(kDemoFloorSizeM, 72, floor_vertices_, floor_indices_);
        floor_local_aabb_ = compute_local_aabb_from_vertices(floor_vertices_);
        cone_local_aabb_ = compute_local_aabb_from_positions(cone_mesh->positions);
        box_local_aabb_ = compute_local_aabb_from_positions(box_mesh->positions);
        capsule_local_aabb_ = compute_local_aabb_from_positions(capsule_debug_mesh.vertices);
        cylinder_local_aabb_ = compute_local_aabb_from_positions(cylinder_debug_mesh.vertices);
        sphere_local_bound_ = shs::sphere_from_aabb(sphere_local_aabb_);
        cone_local_bound_ = shs::sphere_from_aabb(cone_local_aabb_);
        box_local_bound_ = shs::sphere_from_aabb(box_local_aabb_);
        capsule_local_bound_ = shs::sphere_from_aabb(capsule_local_aabb_);
        cylinder_local_bound_ = shs::sphere_from_aabb(cylinder_local_aabb_);
        sphere_shape_jolt_ = shs::jolt::make_sphere(sphere_local_bound_.radius);
        box_shape_jolt_ = shs::jolt::make_box(box_local_aabb_.extent());
        cone_shape_jolt_ = shs::jolt::make_convex_hull(cone_mesh->positions);
        capsule_shape_jolt_ = capsule_debug_shape;
        cylinder_shape_jolt_ = cylinder_debug_shape;

        vertices_.clear();
        vertices_.reserve(sphere_mesh->positions.size());
        for (size_t i = 0; i < sphere_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = sphere_mesh->positions[i];
            if (i < sphere_mesh->normals.size()) v.normal = sphere_mesh->normals[i];
            vertices_.push_back(v);
        }
        indices_ = sphere_mesh->indices;

        floor_model_ = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, -0.15f, 0.0f));
        floor_material_color_ = glm::vec4(120.0f / 255.0f, 122.0f / 255.0f, 128.0f / 255.0f, 1.0f);
        // PBR plastic floor material.
        floor_material_params_ = glm::vec4(0.0f, 0.62f, 1.0f, 0.0f);

        cone_vertices_.clear();
        cone_vertices_.reserve(cone_mesh->positions.size());
        for (size_t i = 0; i < cone_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = cone_mesh->positions[i];
            if (i < cone_mesh->normals.size()) v.normal = cone_mesh->normals[i];
            cone_vertices_.push_back(v);
        }
        cone_indices_ = cone_mesh->indices;
        cone_line_indices_ = make_line_indices_from_triangles(cone_indices_);
        cone_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(cone_vertices_, cone_indices_);

        box_vertices_.clear();
        box_vertices_.reserve(box_mesh->positions.size());
        for (size_t i = 0; i < box_mesh->positions.size(); ++i)
        {
            Vertex v{};
            v.pos = box_mesh->positions[i];
            if (i < box_mesh->normals.size()) v.normal = box_mesh->normals[i];
            box_vertices_.push_back(v);
        }
        box_indices_ = box_mesh->indices;
        box_line_indices_ = make_line_indices_from_triangles(box_indices_);

        capsule_vertices_ = make_vertices_with_normals_from_debug_mesh(capsule_debug_mesh);
        capsule_indices_ = capsule_debug_mesh.indices;
        cylinder_vertices_ = make_vertices_with_normals_from_debug_mesh(cylinder_debug_mesh);
        cylinder_indices_ = cylinder_debug_mesh.indices;

        sphere_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(vertices_, indices_);
        sphere_line_indices_ = make_line_indices_from_triangles(indices_);
        box_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(box_vertices_, box_indices_);
        capsule_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(capsule_vertices_, capsule_indices_);
        cylinder_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(cylinder_vertices_, cylinder_indices_);
        floor_occluder_mesh_ = make_debug_mesh_from_vertex_index_data(floor_vertices_, floor_indices_);

        instances_.clear();
        instance_models_.clear();
        const int layer_count = 5;
        const int rows_per_layer = 8;
        const int cols_per_row = 12;
        const float col_spacing_x = 2.35f * shs::units::meter;
        const float row_spacing_z = 2.15f * shs::units::meter;
        const float layer_spacing_z = 8.5f * shs::units::meter;
        const float base_y = 0.95f * shs::units::meter;
        const float layer_y_step = 0.55f * shs::units::meter;
        std::mt19937 rng(1337u);
        std::uniform_real_distribution<float> jitter(-0.12f * shs::units::meter, 0.12f * shs::units::meter);
        std::uniform_real_distribution<float> hue(0.0f, 1.0f);
        std::uniform_real_distribution<float> scale_rand(0.40f, 0.90f);
        std::uniform_real_distribution<float> rot_rand(-0.28f, 0.28f);
        std::uniform_real_distribution<float> spin_rand(0.06f, 0.26f);
        for (int layer = 0; layer < layer_count; ++layer)
        {
            const float layer_z = (-0.5f * static_cast<float>(layer_count - 1) + static_cast<float>(layer)) * layer_spacing_z;
            for (int row = 0; row < rows_per_layer; ++row)
            {
                const float row_z = layer_z + (-0.5f * static_cast<float>(rows_per_layer - 1) + static_cast<float>(row)) * row_spacing_z;
                const float zig = (((row + layer) & 1) != 0) ? (0.45f * col_spacing_x) : 0.0f;
                for (int col = 0; col < cols_per_row; ++col)
                {
                    const uint32_t logical_idx =
                        static_cast<uint32_t>(layer * rows_per_layer * cols_per_row + row * cols_per_row + col);
                    Instance inst{};
                    switch (logical_idx % 5u)
                    {
                        case 1u:
                            inst.mesh_kind = Instance::MeshKind::Box;
                            break;
                        case 2u:
                            inst.mesh_kind = Instance::MeshKind::Cone;
                            break;
                        case 3u:
                            inst.mesh_kind = Instance::MeshKind::Capsule;
                            break;
                        case 4u:
                            inst.mesh_kind = Instance::MeshKind::Cylinder;
                            break;
                        case 0u:
                        default:
                            inst.mesh_kind = Instance::MeshKind::Sphere;
                            break;
                    }
                    inst.base_pos = glm::vec3(
                        (-0.5f * static_cast<float>(cols_per_row - 1) + static_cast<float>(col)) * col_spacing_x + zig + jitter(rng),
                        base_y + layer_y_step * static_cast<float>(layer) + (0.18f * shs::units::meter) * static_cast<float>(col % 3),
                        row_z + jitter(rng));
                    const float h = hue(rng);
                    inst.base_color = glm::vec4(
                        0.45f + 0.55f * std::sin(6.28318f * (h + 0.00f)),
                        0.45f + 0.55f * std::sin(6.28318f * (h + 0.33f)),
                        0.45f + 0.55f * std::sin(6.28318f * (h + 0.66f)),
                        1.0f);
                    inst.scale = scale_rand(rng);
                    inst.phase = hue(rng) * 10.0f;
                    inst.base_rot = glm::vec3(rot_rand(rng), rot_rand(rng), rot_rand(rng));
                    inst.rot_speed = glm::vec3(spin_rand(rng), spin_rand(rng), spin_rand(rng));
                    inst.metallic = 0.04f + 0.22f * hue(rng);
                    inst.roughness = 0.24f + 0.42f * hue(rng);
                    inst.ao = 1.0f;
                    instances_.push_back(inst);
                }
            }
        }
        instance_models_.resize(instances_.size(), glm::mat4(1.0f));
        instance_visible_mask_.assign(instances_.size(), 1u);
        visible_instance_count_ = static_cast<uint32_t>(instances_.size());
        floor_visible_ = true;

        // Build a stable world-space caster bounds for sun shadow fitting.
        // This avoids per-frame shadow frustum jitter from animation/camera culling.
        shadow_scene_static_aabb_ = shs::transform_aabb(floor_local_aabb_, floor_model_);
        constexpr float kMaxBobAmplitude = 0.18f;
        for (const Instance& inst : instances_)
        {
            const float r = std::max(0.001f, local_bound_for_mesh(inst.mesh_kind).radius * inst.scale * 1.20f);
            const glm::vec3 minv = inst.base_pos + glm::vec3(-r, -r - kMaxBobAmplitude, -r);
            const glm::vec3 maxv = inst.base_pos + glm::vec3( r,  r + kMaxBobAmplitude,  r);
            shadow_scene_static_aabb_.expand(minv);
            shadow_scene_static_aabb_.expand(maxv);
        }
        shadow_scene_static_bounds_ready_ = true;

        light_anim_.clear();
        light_anim_.reserve(kMaxLights);
        gpu_lights_.resize(kMaxLights);
        shadow_lights_gpu_.assign(kMaxLights, ShadowLightGPU{});
        std::uniform_real_distribution<float> angle0(0.0f, 6.28318f);
        std::uniform_real_distribution<float> rad(3.0f * shs::units::meter, 14.0f * shs::units::meter);
        std::uniform_real_distribution<float> hgt(1.6f * shs::units::meter, 4.8f * shs::units::meter);
        std::uniform_real_distribution<float> spd(0.18f, 0.85f);
        std::uniform_real_distribution<float> radius(3.0f * shs::units::meter, 6.8f * shs::units::meter);
        std::uniform_real_distribution<float> inner_deg(12.0f, 20.0f);
        std::uniform_real_distribution<float> outer_extra_deg(6.0f, 14.0f);
        std::uniform_real_distribution<float> area_extent(0.45f * shs::units::meter, 1.25f * shs::units::meter);
        std::uniform_real_distribution<float> tube_half_len(0.45f * shs::units::meter, 1.40f * shs::units::meter);
        std::uniform_real_distribution<float> tube_rad(0.10f * shs::units::meter, 0.28f * shs::units::meter);
        std::uniform_real_distribution<float> axis_rand(-1.0f, 1.0f);
        std::uniform_real_distribution<float> att_pow(0.85f, 1.55f);
        std::uniform_real_distribution<float> att_bias(0.01f, 0.12f);
        std::uniform_real_distribution<float> right_rand(-1.0f, 1.0f);
        for (uint32_t i = 0; i < kMaxLights; ++i)
        {
            LightAnim l{};
            l.angle0 = angle0(rng);
            l.orbit_radius = rad(rng);
            l.height = hgt(rng);
            l.speed = spd(rng) * ((i & 1u) ? 1.0f : -1.0f);
            l.range = radius(rng);
            l.phase = hue(rng) * 10.0f;
            const float t = static_cast<float>(i) / static_cast<float>(kMaxLights);
            l.color = glm::vec3(
                0.35f + 0.65f * std::sin(6.28318f * (t + 0.00f)) * 0.5f + 0.5f,
                0.35f + 0.65f * std::sin(6.28318f * (t + 0.33f)) * 0.5f + 0.5f,
                0.35f + 0.65f * std::sin(6.28318f * (t + 0.66f)) * 0.5f + 0.5f);
            l.intensity = 4.5f + 5.0f * std::fmod(0.6180339f * static_cast<float>(i), 1.0f);
            l.attenuation_power = att_pow(rng);
            l.attenuation_bias = att_bias(rng);
            l.attenuation_cutoff = 0.0f;

            switch (i % 4u)
            {
                case 0u:
                    l.type = shs::LightType::Point;
                    l.attenuation_model = shs::LightAttenuationModel::InverseSquare;
                    l.intensity *= 0.95f;
                    l.color = glm::mix(l.color, glm::vec3(1.0f, 0.66f, 0.30f), 0.58f);
                    break;
                case 1u:
                {
                    l.type = shs::LightType::Spot;
                    l.attenuation_model = shs::LightAttenuationModel::InverseSquare;
                    const float inner = glm::radians(inner_deg(rng));
                    l.spot_inner_outer.x = inner;
                    l.spot_inner_outer.y = inner + glm::radians(outer_extra_deg(rng));
                    l.intensity *= 1.10f;
                    l.color = glm::mix(l.color, glm::vec3(0.34f, 0.84f, 1.0f), 0.63f);
                    break;
                }
                case 2u:
                    l.type = shs::LightType::RectArea;
                    l.attenuation_model = shs::LightAttenuationModel::Smooth;
                    l.shape_params = glm::vec4(area_extent(rng), area_extent(rng), 0.0f, 0.0f);
                    l.rect_right_ws = shs::normalize_or(glm::vec3(right_rand(rng), 0.0f, right_rand(rng)), glm::vec3(1.0f, 0.0f, 0.0f));
                    l.intensity *= 0.85f;
                    l.color = glm::mix(l.color, glm::vec3(0.98f, 0.44f, 0.80f), 0.64f);
                    break;
                case 3u:
                default:
                    l.type = shs::LightType::TubeArea;
                    l.attenuation_model = shs::LightAttenuationModel::Linear;
                    l.shape_params = glm::vec4(tube_half_len(rng), tube_rad(rng), 0.0f, 0.0f);
                    l.intensity *= 0.90f;
                    l.color = glm::mix(l.color, glm::vec3(0.36f, 1.0f, 0.58f), 0.60f);
                    break;
            }
            l.direction_ws = shs::normalize_or(glm::vec3(axis_rand(rng), -0.85f, axis_rand(rng)), glm::vec3(0.0f, -1.0f, 0.0f));
            light_anim_.push_back(l);
        }
        light_set_.points.reserve(kMaxLights);
        light_set_.spots.reserve(kMaxLights);
        light_set_.rect_areas.reserve(kMaxLights / 2u);
        light_set_.tube_areas.reserve(kMaxLights / 2u);

        shadow_settings_ = shs::make_default_shadow_composition_settings();
        shadow_settings_.quality.directional_resolution = kSunShadowMapSize;
        shadow_settings_.quality.local_resolution = kLocalShadowMapSize;
        shadow_settings_.quality.point_resolution = kLocalShadowMapSize;
        shadow_settings_.quality.filter = shs::ShadowFilter::PCF5x5;
        shadow_settings_.quality.pcf_step = 1.0f;
        shadow_settings_.budget.max_spot = std::min<uint32_t>(4u, kMaxSpotShadowMaps);
        shadow_settings_.budget.max_point = std::min<uint32_t>(2u, kMaxPointShadowLights);
        shadow_settings_.rect_area_proxy = false;
        shadow_settings_.tube_area_proxy = false;
        shadow_settings_.budget.max_rect_area = 0u;
        shadow_settings_.budget.max_tube_area = 0u;

        configure_render_path_defaults();
    }

    void configure_vulkan_culler_backend_from_env()
    {
        const char* env = std::getenv("SHS_VK_CULLER_BACKEND");
        if (!env || *env == '\0')
        {
            vulkan_culler_backend_ = VulkanCullerBackend::GpuCompute;
            return;
        }

        std::string v(env);
        std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (v == "off" || v == "0" || v == "disabled" || v == "none")
        {
            vulkan_culler_backend_ = VulkanCullerBackend::Disabled;
            return;
        }
        vulkan_culler_backend_ = VulkanCullerBackend::GpuCompute;
    }

    void init_gpu_resources()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device unavailable");

        create_worker_pools();
        create_gpu_pass_timestamp_resources();
        create_descriptor_resources();
        create_geometry_buffers();
        create_dynamic_buffers();
        const VkExtent2D extent = vk_->swapchain_extent();
        ensure_render_targets(extent.width, extent.height);
        create_pipelines(true, "init");
        observed_swapchain_generation_ = vk_->swapchain_generation();
    }

    void create_worker_pools()
    {
        destroy_worker_pools();
        worker_pools_.resize(worker_count_);
        VkCommandPoolCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = vk_->graphics_queue_family_index();

        for (uint32_t i = 0; i < worker_count_; ++i)
        {
            for (uint32_t f = 0; f < kWorkerPoolRingSize; ++f)
            {
                if (vkCreateCommandPool(vk_->device(), &ci, nullptr, &worker_pools_[i].pools[f]) != VK_SUCCESS)
                {
                    throw std::runtime_error("vkCreateCommandPool failed for worker");
                }
            }
        }
    }

    void destroy_worker_pools()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        for (auto& w : worker_pools_)
        {
            for (auto& pool : w.pools)
            {
                if (pool == VK_NULL_HANDLE) continue;
                vkDestroyCommandPool(vk_->device(), pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
        worker_pools_.clear();
    }

    void destroy_gpu_pass_timestamp_resources()
    {
        gpu_pass_timestamps_supported_ = false;
        gpu_timestamp_period_ns_ = 0.0f;
        gpu_pass_timestamp_recording_active_ = false;
        gpu_pass_query_cursor_ = 0u;
        gpu_pass_timestamp_record_frame_slot_ = 0u;
        gpu_pass_total_ms_ = 0.0;
        gpu_pass_slowest_ms_ = 0.0;
        gpu_pass_slowest_id_.clear();
        gpu_pass_timing_valid_ = false;
        gpu_pass_sample_count_ = 0u;
        gpu_pass_rejected_sample_count_ = 0u;
        gpu_pass_timing_state_ = "disabled";

        for (auto& state : gpu_pass_timestamp_frames_)
        {
            state.samples.clear();
            state.query_count = 0u;
            state.pending = false;
        }

        if (!vk_ || vk_->device() == VK_NULL_HANDLE)
        {
            for (auto& pool : gpu_pass_query_pools_)
            {
                pool = VK_NULL_HANDLE;
            }
            return;
        }

        for (auto& pool : gpu_pass_query_pools_)
        {
            if (pool != VK_NULL_HANDLE)
            {
                vkDestroyQueryPool(vk_->device(), pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
    }

    void create_gpu_pass_timestamp_resources()
    {
        destroy_gpu_pass_timestamp_resources();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE || vk_->physical_device() == VK_NULL_HANDLE) return;

        uint32_t family_count = 0u;
        vkGetPhysicalDeviceQueueFamilyProperties(vk_->physical_device(), &family_count, nullptr);
        if (family_count == 0u)
        {
            gpu_pass_timing_state_ = "no-queue-family";
            return;
        }

        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(vk_->physical_device(), &family_count, families.data());
        const uint32_t graphics_family = vk_->graphics_queue_family_index();
        if (graphics_family >= family_count)
        {
            gpu_pass_timing_state_ = "no-graphics-family";
            return;
        }
        if (families[graphics_family].timestampValidBits == 0u)
        {
            gpu_pass_timing_state_ = "unsupported";
            return;
        }

        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(vk_->physical_device(), &props);
        const uint32_t api_major = VK_VERSION_MAJOR(props.apiVersion);
        const uint32_t api_minor = VK_VERSION_MINOR(props.apiVersion);
        if (api_major < 1u || (api_major == 1u && api_minor < 2u))
        {
            gpu_pass_timing_state_ = "vk<1.2";
            return;
        }
        if (props.limits.timestampPeriod <= 0.0f)
        {
            gpu_pass_timing_state_ = "bad-period";
            return;
        }

        VkQueryPoolCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = kMaxGpuPassTimestampQueries;

        for (auto& pool : gpu_pass_query_pools_)
        {
            if (vkCreateQueryPool(vk_->device(), &qci, nullptr, &pool) != VK_SUCCESS)
            {
                destroy_gpu_pass_timestamp_resources();
                gpu_pass_timing_state_ = "pool-create-failed";
                return;
            }
        }

        gpu_timestamp_period_ns_ = props.limits.timestampPeriod;
        gpu_pass_timestamps_supported_ = true;
        gpu_pass_timing_state_ = "ready";
    }

    void collect_gpu_pass_timing_results(uint32_t frame_slot)
    {
        gpu_pass_total_ms_ = 0.0;
        gpu_pass_slowest_ms_ = 0.0;
        gpu_pass_slowest_id_.clear();
        gpu_pass_timing_valid_ = false;
        gpu_pass_sample_count_ = 0u;
        gpu_pass_rejected_sample_count_ = 0u;

        if (!gpu_pass_timestamps_supported_)
        {
            gpu_pass_timing_state_ = "disabled";
            return;
        }
        if (!vk_ || vk_->device() == VK_NULL_HANDLE)
        {
            gpu_pass_timing_state_ = "no-device";
            return;
        }
        if (frame_slot >= kWorkerPoolRingSize)
        {
            gpu_pass_timing_state_ = "bad-slot";
            return;
        }
        if (gpu_pass_query_pools_[frame_slot] == VK_NULL_HANDLE)
        {
            gpu_pass_timing_state_ = "no-query-pool";
            return;
        }

        GpuPassTimestampFrameState& frame_state = gpu_pass_timestamp_frames_[frame_slot];
        if (!frame_state.pending)
        {
            gpu_pass_timing_state_ = "idle";
            return;
        }
        if (frame_state.query_count < 2u || frame_state.samples.empty())
        {
            frame_state.pending = false;
            gpu_pass_timing_state_ = "no-samples";
            return;
        }

        std::vector<uint64_t> ticks(frame_state.query_count, 0ull);
        const VkResult qr = vkGetQueryPoolResults(
            vk_->device(),
            gpu_pass_query_pools_[frame_slot],
            0u,
            frame_state.query_count,
            ticks.size() * sizeof(uint64_t),
            ticks.data(),
            sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT);
        if (qr == VK_NOT_READY)
        {
            gpu_pass_timing_state_ = "query-pending";
            return;
        }
        frame_state.pending = false;
        if (qr != VK_SUCCESS)
        {
            gpu_pass_timing_state_ = "query-failed";
            return;
        }

        for (const GpuPassTimestampSample& sample : frame_state.samples)
        {
            if (!sample.success)
            {
                ++gpu_pass_rejected_sample_count_;
                continue;
            }
            if (sample.begin_query == UINT32_MAX || sample.end_query == UINT32_MAX)
            {
                ++gpu_pass_rejected_sample_count_;
                continue;
            }
            if (sample.begin_query >= frame_state.query_count || sample.end_query >= frame_state.query_count)
            {
                ++gpu_pass_rejected_sample_count_;
                continue;
            }
            const uint64_t begin_tick = ticks[sample.begin_query];
            const uint64_t end_tick = ticks[sample.end_query];
            if (end_tick < begin_tick)
            {
                ++gpu_pass_rejected_sample_count_;
                continue;
            }

            const double ms =
                static_cast<double>(end_tick - begin_tick) *
                static_cast<double>(gpu_timestamp_period_ns_) *
                1e-6;
            gpu_pass_total_ms_ += ms;
            if (ms >= gpu_pass_slowest_ms_)
            {
                gpu_pass_slowest_ms_ = ms;
                gpu_pass_slowest_id_ = sample.pass_id;
            }
            ++gpu_pass_sample_count_;
        }

        if (gpu_pass_sample_count_ == 0u)
        {
            gpu_pass_timing_state_ = "zero-sample";
            return;
        }

        gpu_pass_timing_state_ = "ready";
        gpu_pass_timing_valid_ = true;
    }

    void begin_gpu_pass_timing_recording(VkCommandBuffer cmd, uint32_t frame_slot)
    {
        gpu_pass_timestamp_recording_active_ = false;
        gpu_pass_query_cursor_ = 0u;
        if (!gpu_pass_timestamps_supported_)
        {
            gpu_pass_timing_state_ = "disabled";
            return;
        }
        if (frame_slot >= kWorkerPoolRingSize)
        {
            gpu_pass_timing_state_ = "bad-slot";
            return;
        }
        if (gpu_pass_query_pools_[frame_slot] == VK_NULL_HANDLE)
        {
            gpu_pass_timing_state_ = "no-query-pool";
            return;
        }

        GpuPassTimestampFrameState& frame_state = gpu_pass_timestamp_frames_[frame_slot];
        frame_state.samples.clear();
        frame_state.query_count = 0u;
        frame_state.pending = false;

        vkCmdResetQueryPool(cmd, gpu_pass_query_pools_[frame_slot], 0u, kMaxGpuPassTimestampQueries);
        gpu_pass_timestamp_record_frame_slot_ = frame_slot;
        gpu_pass_timestamp_recording_active_ = true;
        gpu_pass_timing_state_ = "recording";
    }

    uint32_t begin_gpu_pass_timestamp(
        shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>& ctx,
        const shs::RenderPathCompiledPass& pass)
    {
        if (!gpu_pass_timestamp_recording_active_) return UINT32_MAX;
        if (!ctx.fi) return UINT32_MAX;
        if (ctx.frame_slot >= kWorkerPoolRingSize) return UINT32_MAX;
        if (ctx.frame_slot != gpu_pass_timestamp_record_frame_slot_) return UINT32_MAX;
        if (gpu_pass_query_cursor_ + 2u > kMaxGpuPassTimestampQueries) return UINT32_MAX;
        if (gpu_pass_query_pools_[ctx.frame_slot] == VK_NULL_HANDLE) return UINT32_MAX;

        GpuPassTimestampFrameState& frame_state = gpu_pass_timestamp_frames_[ctx.frame_slot];
        GpuPassTimestampSample sample{};
        sample.pass_id = pass.id;
        sample.pass_kind = shs::pass_id_is_standard(pass.pass_id) ? pass.pass_id : shs::parse_pass_id(pass.id);
        sample.begin_query = gpu_pass_query_cursor_++;
        vkCmdWriteTimestamp(
            ctx.fi->cmd,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            gpu_pass_query_pools_[ctx.frame_slot],
            sample.begin_query);
        frame_state.samples.push_back(std::move(sample));
        return static_cast<uint32_t>(frame_state.samples.size() - 1u);
    }

    void end_gpu_pass_timestamp(
        shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>& ctx,
        uint32_t sample_index,
        bool success)
    {
        if (!gpu_pass_timestamp_recording_active_) return;
        if (!ctx.fi) return;
        if (sample_index == UINT32_MAX) return;
        if (ctx.frame_slot >= kWorkerPoolRingSize) return;
        if (ctx.frame_slot != gpu_pass_timestamp_record_frame_slot_) return;
        if (gpu_pass_query_pools_[ctx.frame_slot] == VK_NULL_HANDLE) return;
        if (gpu_pass_query_cursor_ >= kMaxGpuPassTimestampQueries) return;

        GpuPassTimestampFrameState& frame_state = gpu_pass_timestamp_frames_[ctx.frame_slot];
        if (sample_index >= frame_state.samples.size()) return;
        GpuPassTimestampSample& sample = frame_state.samples[sample_index];
        sample.end_query = gpu_pass_query_cursor_++;
        sample.success = success;
        vkCmdWriteTimestamp(
            ctx.fi->cmd,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            gpu_pass_query_pools_[ctx.frame_slot],
            sample.end_query);
    }

    void finalize_gpu_pass_timing_recording(uint32_t frame_slot)
    {
        if (!gpu_pass_timestamp_recording_active_) return;
        if (frame_slot >= kWorkerPoolRingSize) return;
        if (frame_slot != gpu_pass_timestamp_record_frame_slot_) return;

        GpuPassTimestampFrameState& frame_state = gpu_pass_timestamp_frames_[frame_slot];
        frame_state.query_count = std::min<uint32_t>(gpu_pass_query_cursor_, kMaxGpuPassTimestampQueries);
        frame_state.pending = frame_state.query_count >= 2u && !frame_state.samples.empty();
        if (frame_state.pending)
        {
            gpu_pass_timing_state_ = "submitted";
        }
        else
        {
            gpu_pass_timing_state_ = "no-samples";
        }

        gpu_pass_timestamp_recording_active_ = false;
        gpu_pass_query_cursor_ = 0u;
    }

    template <typename THandler>
    bool execute_profiled_pass_handler(
        shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>& ctx,
        const shs::RenderPathCompiledPass& pass,
        THandler&& handler)
    {
        const uint32_t token = begin_gpu_pass_timestamp(ctx, pass);
        const bool ok = handler(ctx, pass);
        end_gpu_pass_timestamp(ctx, token, ok);
        return ok;
    }

    void create_buffer(
        VkDeviceSize size,
        VkBufferUsageFlags usage,
        VkMemoryPropertyFlags mem_flags,
        GpuBuffer& out,
        bool map_memory)
    {
        destroy_buffer(out);
        if (!shs::vk_create_buffer(
                vk_->device(),
                vk_->physical_device(),
                size,
                usage,
                mem_flags,
                out.buffer,
                out.memory))
        {
            throw std::runtime_error("vk_create_buffer failed");
        }

        out.size = size;
        if (map_memory)
        {
            if (vkMapMemory(vk_->device(), out.memory, 0, size, 0, &out.mapped) != VK_SUCCESS)
            {
                shs::vk_destroy_buffer(vk_->device(), out.buffer, out.memory);
                throw std::runtime_error("vkMapMemory failed");
            }
        }
    }

    void destroy_buffer(GpuBuffer& b)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (b.mapped)
        {
            vkUnmapMemory(vk_->device(), b.memory);
            b.mapped = nullptr;
        }
        shs::vk_destroy_buffer(vk_->device(), b.buffer, b.memory);
        b.size = 0;
    }

    void create_geometry_buffers()
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        create_buffer(
            static_cast<VkDeviceSize>(vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            vertex_buffer_,
            true);
        std::memcpy(vertex_buffer_.mapped, vertices_.data(), vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            index_buffer_,
            true);
        std::memcpy(index_buffer_.mapped, indices_.data(), indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(floor_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            floor_vertex_buffer_,
            true);
        std::memcpy(floor_vertex_buffer_.mapped, floor_vertices_.data(), floor_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(floor_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            floor_index_buffer_,
            true);
        std::memcpy(floor_index_buffer_.mapped, floor_indices_.data(), floor_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(cone_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            cone_vertex_buffer_,
            true);
        std::memcpy(cone_vertex_buffer_.mapped, cone_vertices_.data(), cone_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(cone_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            cone_index_buffer_,
            true);
        std::memcpy(cone_index_buffer_.mapped, cone_indices_.data(), cone_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(box_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            box_vertex_buffer_,
            true);
        std::memcpy(box_vertex_buffer_.mapped, box_vertices_.data(), box_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(box_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            box_index_buffer_,
            true);
        std::memcpy(box_index_buffer_.mapped, box_indices_.data(), box_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(sphere_line_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            sphere_line_index_buffer_,
            true);
        std::memcpy(
            sphere_line_index_buffer_.mapped,
            sphere_line_indices_.data(),
            sphere_line_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(cone_line_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            cone_line_index_buffer_,
            true);
        std::memcpy(
            cone_line_index_buffer_.mapped,
            cone_line_indices_.data(),
            cone_line_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(box_line_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            box_line_index_buffer_,
            true);
        std::memcpy(
            box_line_index_buffer_.mapped,
            box_line_indices_.data(),
            box_line_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(capsule_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            capsule_vertex_buffer_,
            true);
        std::memcpy(capsule_vertex_buffer_.mapped, capsule_vertices_.data(), capsule_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(capsule_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            capsule_index_buffer_,
            true);
        std::memcpy(capsule_index_buffer_.mapped, capsule_indices_.data(), capsule_indices_.size() * sizeof(uint32_t));

        create_buffer(
            static_cast<VkDeviceSize>(cylinder_vertices_.size() * sizeof(Vertex)),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host_flags,
            cylinder_vertex_buffer_,
            true);
        std::memcpy(cylinder_vertex_buffer_.mapped, cylinder_vertices_.data(), cylinder_vertices_.size() * sizeof(Vertex));

        create_buffer(
            static_cast<VkDeviceSize>(cylinder_indices_.size() * sizeof(uint32_t)),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host_flags,
            cylinder_index_buffer_,
            true);
        std::memcpy(cylinder_index_buffer_.mapped, cylinder_indices_.data(), cylinder_indices_.size() * sizeof(uint32_t));
    }

    void create_dynamic_buffers()
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        for (FrameResources& fr : frame_resources_)
        {
            create_buffer(
                sizeof(CameraUBO),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                host_flags,
                fr.camera_buffer,
                true);

            create_buffer(
                static_cast<VkDeviceSize>(kMaxLights) * sizeof(shs::CullingLightGPU),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_flags,
                fr.light_buffer,
                true);

            create_buffer(
                static_cast<VkDeviceSize>(kMaxLights) * sizeof(ShadowLightGPU),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_flags,
                fr.shadow_light_buffer,
                true);

            std::memset(fr.shadow_light_buffer.mapped, 0, static_cast<size_t>(fr.shadow_light_buffer.size));
        }
    }

    VkFormat choose_depth_format() const
    {
        const std::array<VkFormat, 3> candidates{
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
        };

        for (VkFormat fmt : candidates)
        {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(vk_->physical_device(), fmt, &props);
            const VkFormatFeatureFlags need =
                VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((props.optimalTilingFeatures & need) == need)
            {
                return fmt;
            }
        }

        return VK_FORMAT_D32_SFLOAT;
    }

    void destroy_depth_target()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        if (depth_target_.framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(dev, depth_target_.framebuffer, nullptr);
            depth_target_.framebuffer = VK_NULL_HANDLE;
        }
        if (depth_target_.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, depth_target_.render_pass, nullptr);
            depth_target_.render_pass = VK_NULL_HANDLE;
        }
        if (depth_target_.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(dev, depth_target_.view, nullptr);
            depth_target_.view = VK_NULL_HANDLE;
        }
        if (depth_target_.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(dev, depth_target_.image, nullptr);
            depth_target_.image = VK_NULL_HANDLE;
        }
        if (depth_target_.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(dev, depth_target_.memory, nullptr);
            depth_target_.memory = VK_NULL_HANDLE;
        }
        depth_target_.w = 0;
        depth_target_.h = 0;
        depth_target_.format = VK_FORMAT_UNDEFINED;
    }

    void destroy_gbuffer_target()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();

        if (gbuffer_target_.framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(dev, gbuffer_target_.framebuffer, nullptr);
            gbuffer_target_.framebuffer = VK_NULL_HANDLE;
        }
        if (gbuffer_target_.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, gbuffer_target_.render_pass, nullptr);
            gbuffer_target_.render_pass = VK_NULL_HANDLE;
        }

        for (auto& att : gbuffer_target_.colors)
        {
            if (att.view != VK_NULL_HANDLE)
            {
                vkDestroyImageView(dev, att.view, nullptr);
                att.view = VK_NULL_HANDLE;
            }
            if (att.image != VK_NULL_HANDLE)
            {
                vkDestroyImage(dev, att.image, nullptr);
                att.image = VK_NULL_HANDLE;
            }
            if (att.memory != VK_NULL_HANDLE)
            {
                vkFreeMemory(dev, att.memory, nullptr);
                att.memory = VK_NULL_HANDLE;
            }
            att.format = VK_FORMAT_UNDEFINED;
        }

        gbuffer_target_.w = 0;
        gbuffer_target_.h = 0;
    }

    VkFormat choose_ao_format() const
    {
        const std::array<VkFormat, 3> candidates{
            VK_FORMAT_R8_UNORM,
            VK_FORMAT_R16_SFLOAT,
            VK_FORMAT_R8G8B8A8_UNORM
        };
        for (VkFormat fmt : candidates)
        {
            VkFormatProperties props{};
            vkGetPhysicalDeviceFormatProperties(vk_->physical_device(), fmt, &props);
            const VkFormatFeatureFlags need =
                VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((props.optimalTilingFeatures & need) == need)
            {
                return fmt;
            }
        }
        return VK_FORMAT_R8G8B8A8_UNORM;
    }

    void destroy_ao_target()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();

        if (ao_target_.framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(dev, ao_target_.framebuffer, nullptr);
            ao_target_.framebuffer = VK_NULL_HANDLE;
        }
        if (ao_target_.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, ao_target_.render_pass, nullptr);
            ao_target_.render_pass = VK_NULL_HANDLE;
        }
        if (ao_target_.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(dev, ao_target_.view, nullptr);
            ao_target_.view = VK_NULL_HANDLE;
        }
        if (ao_target_.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(dev, ao_target_.image, nullptr);
            ao_target_.image = VK_NULL_HANDLE;
        }
        if (ao_target_.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(dev, ao_target_.memory, nullptr);
            ao_target_.memory = VK_NULL_HANDLE;
        }

        ao_target_.w = 0;
        ao_target_.h = 0;
        ao_target_.format = VK_FORMAT_UNDEFINED;
    }

    VkFormat choose_gbuffer_format() const
    {
        return VK_FORMAT_R16G16B16A16_SFLOAT;
    }

    void create_ao_target(uint32_t w, uint32_t h)
    {
        destroy_ao_target();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device unavailable");
        if (w == 0 || h == 0) return;

        ao_target_.w = w;
        ao_target_.h = h;
        ao_target_.format = choose_ao_format();

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.format = ao_target_.format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk_->device(), &ici, nullptr, &ao_target_.image) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImage failed for AO target");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(vk_->device(), ao_target_.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = shs::vk_find_memory_type(
            vk_->physical_device(),
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            throw std::runtime_error("No compatible memory type for AO target");
        }
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &ao_target_.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for AO target");
        }
        if (vkBindImageMemory(vk_->device(), ao_target_.image, ao_target_.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindImageMemory failed for AO target");
        }

        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = ao_target_.image;
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = ao_target_.format;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.baseMipLevel = 0;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk_->device(), &iv, nullptr, &ao_target_.view) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for AO target");
        }

        VkAttachmentDescription ao_att{};
        ao_att.format = ao_target_.format;
        ao_att.samples = VK_SAMPLE_COUNT_1_BIT;
        ao_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ao_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ao_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ao_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ao_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ao_att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference ao_ref{};
        ao_ref.attachment = 0;
        ao_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &ao_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &ao_att;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &ao_target_.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for AO target");
        }

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = ao_target_.render_pass;
        fb.attachmentCount = 1;
        fb.pAttachments = &ao_target_.view;
        fb.width = w;
        fb.height = h;
        fb.layers = 1;
        if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &ao_target_.framebuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer failed for AO target");
        }
    }

    void destroy_post_color_target(PostColorTarget& t)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        if (t.framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(dev, t.framebuffer, nullptr);
            t.framebuffer = VK_NULL_HANDLE;
        }
        if (t.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, t.render_pass, nullptr);
            t.render_pass = VK_NULL_HANDLE;
        }
        if (t.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(dev, t.view, nullptr);
            t.view = VK_NULL_HANDLE;
        }
        if (t.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(dev, t.image, nullptr);
            t.image = VK_NULL_HANDLE;
        }
        if (t.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(dev, t.memory, nullptr);
            t.memory = VK_NULL_HANDLE;
        }
        t.w = 0;
        t.h = 0;
        t.format = VK_FORMAT_UNDEFINED;
        if (&t == &post_target_a_) post_target_a_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        if (&t == &post_target_b_) post_target_b_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void create_post_color_target(PostColorTarget& t, uint32_t w, uint32_t h, VkFormat format)
    {
        destroy_post_color_target(t);
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device unavailable");
        if (w == 0 || h == 0 || format == VK_FORMAT_UNDEFINED) return;

        t.w = w;
        t.h = h;
        t.format = format;

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.format = format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk_->device(), &ici, nullptr, &t.image) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImage failed for post color target");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(vk_->device(), t.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = shs::vk_find_memory_type(
            vk_->physical_device(),
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            throw std::runtime_error("No compatible memory type for post color target");
        }
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &t.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for post color target");
        }
        if (vkBindImageMemory(vk_->device(), t.image, t.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindImageMemory failed for post color target");
        }

        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = t.image;
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = format;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        iv.subresourceRange.baseMipLevel = 0;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk_->device(), &iv, nullptr, &t.view) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for post color target");
        }

        VkAttachmentDescription color_att{};
        color_att.format = format;
        color_att.samples = VK_SAMPLE_COUNT_1_BIT;
        color_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        color_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        color_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        color_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        color_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        color_att.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference color_ref{};
        color_ref.attachment = 0;
        color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = 1;
        sub.pColorAttachments = &color_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &color_att;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &t.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for post color target");
        }

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = t.render_pass;
        fb.attachmentCount = 1;
        fb.pAttachments = &t.view;
        fb.width = w;
        fb.height = h;
        fb.layers = 1;
        if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &t.framebuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer failed for post color target");
        }
        if (&t == &post_target_a_) post_target_a_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
        if (&t == &post_target_b_) post_target_b_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    }

    void create_gbuffer_target(uint32_t w, uint32_t h)
    {
        destroy_gbuffer_target();
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) throw std::runtime_error("Vulkan device unavailable");
        if (depth_target_.view == VK_NULL_HANDLE || depth_target_.format == VK_FORMAT_UNDEFINED)
        {
            throw std::runtime_error("Depth target must be created before gbuffer target");
        }

        gbuffer_target_.w = w;
        gbuffer_target_.h = h;
        const VkFormat color_fmt = choose_gbuffer_format();

        for (auto& att : gbuffer_target_.colors)
        {
            att.format = color_fmt;

            VkImageCreateInfo ici{};
            ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ici.imageType = VK_IMAGE_TYPE_2D;
            ici.extent.width = w;
            ici.extent.height = h;
            ici.extent.depth = 1;
            ici.mipLevels = 1;
            ici.arrayLayers = 1;
            ici.format = att.format;
            ici.tiling = VK_IMAGE_TILING_OPTIMAL;
            ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            ici.samples = VK_SAMPLE_COUNT_1_BIT;
            ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateImage(vk_->device(), &ici, nullptr, &att.image) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateImage failed for gbuffer attachment");
            }

            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(vk_->device(), att.image, &req);

            VkMemoryAllocateInfo mai{};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = shs::vk_find_memory_type(
                vk_->physical_device(),
                req.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (mai.memoryTypeIndex == UINT32_MAX)
            {
                throw std::runtime_error("No compatible memory type for gbuffer attachment");
            }
            if (vkAllocateMemory(vk_->device(), &mai, nullptr, &att.memory) != VK_SUCCESS)
            {
                throw std::runtime_error("vkAllocateMemory failed for gbuffer attachment");
            }
            if (vkBindImageMemory(vk_->device(), att.image, att.memory, 0) != VK_SUCCESS)
            {
                throw std::runtime_error("vkBindImageMemory failed for gbuffer attachment");
            }

            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = att.image;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = att.format;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = 0;
            iv.subresourceRange.layerCount = 1;
            if (vkCreateImageView(vk_->device(), &iv, nullptr, &att.view) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateImageView failed for gbuffer attachment");
            }
        }

        std::array<VkAttachmentDescription, 5> attachments{};
        for (uint32_t i = 0; i < 4; ++i)
        {
            attachments[i].format = gbuffer_target_.colors[i].format;
            attachments[i].samples = VK_SAMPLE_COUNT_1_BIT;
            attachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            attachments[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            attachments[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            attachments[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            attachments[i].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        attachments[4].format = depth_target_.format;
        attachments[4].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[4].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[4].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[4].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[4].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[4].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[4].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        std::array<VkAttachmentReference, 4> color_refs{};
        for (uint32_t i = 0; i < 4; ++i)
        {
            color_refs[i].attachment = i;
            color_refs[i].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 4;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.colorAttachmentCount = static_cast<uint32_t>(color_refs.size());
        sub.pColorAttachments = color_refs.data();
        sub.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask =
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask =
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask =
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = static_cast<uint32_t>(attachments.size());
        rp.pAttachments = attachments.data();
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &gbuffer_target_.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for gbuffer target");
        }

        std::array<VkImageView, 5> views = {
            gbuffer_target_.colors[0].view,
            gbuffer_target_.colors[1].view,
            gbuffer_target_.colors[2].view,
            gbuffer_target_.colors[3].view,
            depth_target_.view
        };

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = gbuffer_target_.render_pass;
        fb.attachmentCount = static_cast<uint32_t>(views.size());
        fb.pAttachments = views.data();
        fb.width = w;
        fb.height = h;
        fb.layers = 1;
        if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &gbuffer_target_.framebuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer failed for gbuffer target");
        }
    }

    void create_depth_target(uint32_t w, uint32_t h)
    {
        destroy_depth_target();
        depth_target_.w = w;
        depth_target_.h = h;
        depth_target_.format = choose_depth_format();

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.format = depth_target_.format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk_->device(), &ici, nullptr, &depth_target_.image) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImage failed for depth target");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(vk_->device(), depth_target_.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = shs::vk_find_memory_type(
            vk_->physical_device(),
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            throw std::runtime_error("No compatible memory type for depth target");
        }
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &depth_target_.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for depth target");
        }
        if (vkBindImageMemory(vk_->device(), depth_target_.image, depth_target_.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindImageMemory failed for depth target");
        }

        VkImageViewCreateInfo iv{};
        iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        iv.image = depth_target_.image;
        iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
        iv.format = depth_target_.format;
        iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        iv.subresourceRange.baseMipLevel = 0;
        iv.subresourceRange.levelCount = 1;
        iv.subresourceRange.baseArrayLayer = 0;
        iv.subresourceRange.layerCount = 1;
        if (vkCreateImageView(vk_->device(), &iv, nullptr, &depth_target_.view) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for depth target");
        }

        VkAttachmentDescription depth_att{};
        depth_att.format = depth_target_.format;
        depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &depth_att;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &depth_target_.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for depth prepass");
        }

        VkFramebufferCreateInfo fb{};
        fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fb.renderPass = depth_target_.render_pass;
        fb.attachmentCount = 1;
        fb.pAttachments = &depth_target_.view;
        fb.width = w;
        fb.height = h;
        fb.layers = 1;
        if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &depth_target_.framebuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer failed for depth prepass");
        }
    }

    void destroy_layered_depth_target(LayeredDepthTarget& t)
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        VkDevice dev = vk_->device();
        for (VkFramebuffer fb : t.framebuffers)
        {
            if (fb != VK_NULL_HANDLE) vkDestroyFramebuffer(dev, fb, nullptr);
        }
        t.framebuffers.clear();
        for (VkImageView v : t.layer_views)
        {
            if (v != VK_NULL_HANDLE) vkDestroyImageView(dev, v, nullptr);
        }
        t.layer_views.clear();
        if (t.render_pass != VK_NULL_HANDLE)
        {
            vkDestroyRenderPass(dev, t.render_pass, nullptr);
            t.render_pass = VK_NULL_HANDLE;
        }
        if (t.sampled_view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(dev, t.sampled_view, nullptr);
            t.sampled_view = VK_NULL_HANDLE;
        }
        if (t.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(dev, t.image, nullptr);
            t.image = VK_NULL_HANDLE;
        }
        if (t.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(dev, t.memory, nullptr);
            t.memory = VK_NULL_HANDLE;
        }
        t.w = 0;
        t.h = 0;
        t.layers = 0;
        t.format = VK_FORMAT_UNDEFINED;
    }

    void create_layered_depth_target(
        LayeredDepthTarget& out,
        uint32_t w,
        uint32_t h,
        uint32_t layers,
        VkImageViewType sampled_view_type)
    {
        destroy_layered_depth_target(out);
        out.w = w;
        out.h = h;
        out.layers = layers;
        out.format = choose_depth_format();

        VkImageCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.extent.width = w;
        ici.extent.height = h;
        ici.extent.depth = 1;
        ici.mipLevels = 1;
        ici.arrayLayers = layers;
        ici.format = out.format;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateImage(vk_->device(), &ici, nullptr, &out.image) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImage failed for layered depth target");
        }

        VkMemoryRequirements req{};
        vkGetImageMemoryRequirements(vk_->device(), out.image, &req);

        VkMemoryAllocateInfo mai{};
        mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = shs::vk_find_memory_type(
            vk_->physical_device(),
            req.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == UINT32_MAX)
        {
            throw std::runtime_error("No compatible memory type for layered depth target");
        }
        if (vkAllocateMemory(vk_->device(), &mai, nullptr, &out.memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory failed for layered depth target");
        }
        if (vkBindImageMemory(vk_->device(), out.image, out.memory, 0) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBindImageMemory failed for layered depth target");
        }

        VkImageViewCreateInfo sv{};
        sv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        sv.image = out.image;
        sv.viewType = sampled_view_type;
        sv.format = out.format;
        sv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        sv.subresourceRange.baseMipLevel = 0;
        sv.subresourceRange.levelCount = 1;
        sv.subresourceRange.baseArrayLayer = 0;
        sv.subresourceRange.layerCount = layers;
        if (vkCreateImageView(vk_->device(), &sv, nullptr, &out.sampled_view) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for layered depth sampled view");
        }

        VkAttachmentDescription depth_att{};
        depth_att.format = out.format;
        depth_att.samples = VK_SAMPLE_COUNT_1_BIT;
        depth_att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depth_att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depth_att.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_att.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

        VkAttachmentReference depth_ref{};
        depth_ref.attachment = 0;
        depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription sub{};
        sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sub.pDepthStencilAttachment = &depth_ref;

        VkSubpassDependency deps[2]{};
        deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        deps[0].dstSubpass = 0;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        deps[1].srcSubpass = 0;
        deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        deps[1].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        VkRenderPassCreateInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        rp.attachmentCount = 1;
        rp.pAttachments = &depth_att;
        rp.subpassCount = 1;
        rp.pSubpasses = &sub;
        rp.dependencyCount = 2;
        rp.pDependencies = deps;
        if (vkCreateRenderPass(vk_->device(), &rp, nullptr, &out.render_pass) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateRenderPass failed for layered depth target");
        }

        out.layer_views.resize(layers, VK_NULL_HANDLE);
        out.framebuffers.resize(layers, VK_NULL_HANDLE);
        for (uint32_t i = 0; i < layers; ++i)
        {
            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = out.image;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = out.format;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = i;
            iv.subresourceRange.layerCount = 1;
            if (vkCreateImageView(vk_->device(), &iv, nullptr, &out.layer_views[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateImageView failed for layered depth view");
            }

            VkFramebufferCreateInfo fb{};
            fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            fb.renderPass = out.render_pass;
            fb.attachmentCount = 1;
            fb.pAttachments = &out.layer_views[i];
            fb.width = w;
            fb.height = h;
            fb.layers = 1;
            if (vkCreateFramebuffer(vk_->device(), &fb, nullptr, &out.framebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateFramebuffer failed for layered depth target");
            }
        }
    }

    void ensure_shadow_targets()
    {
        const bool sun_ok =
            sun_shadow_target_.w == kSunShadowMapSize &&
            sun_shadow_target_.h == kSunShadowMapSize &&
            sun_shadow_target_.layers == 1u &&
            sun_shadow_target_.sampled_view != VK_NULL_HANDLE;
        const bool local_ok =
            local_shadow_target_.w == kLocalShadowMapSize &&
            local_shadow_target_.h == kLocalShadowMapSize &&
            local_shadow_target_.layers == kMaxLocalShadowLayers &&
            local_shadow_target_.sampled_view != VK_NULL_HANDLE;
        if (sun_ok && local_ok) return;

        create_layered_depth_target(
            sun_shadow_target_,
            kSunShadowMapSize,
            kSunShadowMapSize,
            1u,
            VK_IMAGE_VIEW_TYPE_2D);
        create_layered_depth_target(
            local_shadow_target_,
            kLocalShadowMapSize,
            kLocalShadowMapSize,
            kMaxLocalShadowLayers,
            VK_IMAGE_VIEW_TYPE_2D_ARRAY);
    }

    void create_or_resize_tile_buffers(const shs::RenderPathLightGridRuntimeLayout& layout)
    {
        const VkMemoryPropertyFlags host_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        const shs::RenderPathLightGridBufferSizes sizes =
            shs::make_render_path_light_grid_buffer_sizes(layout, kMaxLightsPerTile);
        const VkDeviceSize counts_size = static_cast<VkDeviceSize>(sizes.counts_bytes);
        const VkDeviceSize indices_size = static_cast<VkDeviceSize>(sizes.indices_bytes);
        const VkDeviceSize depth_ranges_size = static_cast<VkDeviceSize>(sizes.depth_ranges_bytes);

        for (FrameResources& fr : frame_resources_)
        {
            create_buffer(counts_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, fr.tile_counts_buffer, true);
            create_buffer(indices_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, fr.tile_indices_buffer, true);
            create_buffer(depth_ranges_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_flags, fr.tile_depth_ranges_buffer, true);

            std::memset(fr.tile_counts_buffer.mapped, 0, static_cast<size_t>(counts_size));
            std::memset(fr.tile_indices_buffer.mapped, 0, static_cast<size_t>(indices_size));
            std::memset(fr.tile_depth_ranges_buffer.mapped, 0, static_cast<size_t>(depth_ranges_size));
        }
    }

    void create_descriptor_resources()
    {
        if (depth_sampler_ == VK_NULL_HANDLE)
        {
            VkSamplerCreateInfo sci{};
            sci.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            sci.magFilter = VK_FILTER_NEAREST;
            sci.minFilter = VK_FILTER_NEAREST;
            sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            sci.minLod = 0.0f;
            sci.maxLod = 0.0f;
            sci.maxAnisotropy = 1.0f;
            if (vkCreateSampler(vk_->device(), &sci, nullptr, &depth_sampler_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateSampler failed (depth)");
            }
        }

        if (global_set_layout_ == VK_NULL_HANDLE)
        {
            if (!shs::vk_create_render_path_global_descriptor_set_layout(vk_->device(), &global_set_layout_))
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout failed (render-path global)");
            }
        }

        if (descriptor_pool_ == VK_NULL_HANDLE)
        {
            if (!shs::vk_create_render_path_global_descriptor_pool(
                    vk_->device(),
                    static_cast<uint32_t>(kWorkerPoolRingSize),
                    &descriptor_pool_))
            {
                throw std::runtime_error("vkCreateDescriptorPool failed (render-path global)");
            }
        }

        if (frame_resources_.at_slot(0).global_set == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSet, kWorkerPoolRingSize> sets{};
            if (!shs::vk_allocate_descriptor_set_ring<kWorkerPoolRingSize>(
                    vk_->device(),
                    descriptor_pool_,
                    global_set_layout_,
                    sets))
            {
                throw std::runtime_error("vkAllocateDescriptorSets failed");
            }
            for (uint32_t i = 0; i < kWorkerPoolRingSize; ++i)
            {
                frame_resources_.at_slot(i).global_set = sets[i];
            }
        }

        if (deferred_set_layout_ == VK_NULL_HANDLE)
        {
            VkDescriptorSetLayoutBinding bindings[7]{};
            for (uint32_t i = 0; i < 7; ++i)
            {
                bindings[i].binding = i;
                bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                bindings[i].descriptorCount = 1;
                bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
            }

            VkDescriptorSetLayoutCreateInfo lci{};
            lci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            lci.bindingCount = 7;
            lci.pBindings = bindings;
            if (vkCreateDescriptorSetLayout(vk_->device(), &lci, nullptr, &deferred_set_layout_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorSetLayout failed (deferred gbuffer set)");
            }
        }

        if (deferred_descriptor_pool_ == VK_NULL_HANDLE)
        {
            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            pool_size.descriptorCount = 21u;

            VkDescriptorPoolCreateInfo pci{};
            pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pci.maxSets = 3u;
            pci.poolSizeCount = 1u;
            pci.pPoolSizes = &pool_size;
            if (vkCreateDescriptorPool(vk_->device(), &pci, nullptr, &deferred_descriptor_pool_) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorPool failed (deferred gbuffer set)");
            }
        }

        if (deferred_set_ == VK_NULL_HANDLE ||
            deferred_post_a_set_ == VK_NULL_HANDLE ||
            deferred_post_b_set_ == VK_NULL_HANDLE)
        {
            std::array<VkDescriptorSetLayout, 3> set_layouts = {
                deferred_set_layout_,
                deferred_set_layout_,
                deferred_set_layout_
            };
            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = deferred_descriptor_pool_;
            ai.descriptorSetCount = static_cast<uint32_t>(set_layouts.size());
            ai.pSetLayouts = set_layouts.data();
            std::array<VkDescriptorSet, 3> sets{};
            if (vkAllocateDescriptorSets(vk_->device(), &ai, sets.data()) != VK_SUCCESS)
            {
                throw std::runtime_error("vkAllocateDescriptorSets failed (deferred gbuffer set)");
            }
            deferred_set_ = sets[0];
            deferred_post_a_set_ = sets[1];
            deferred_post_b_set_ = sets[2];
        }
    }

    void update_global_descriptor_sets()
    {
        for (FrameResources& fr : frame_resources_)
        {
            if (fr.global_set == VK_NULL_HANDLE) continue;

            shs::VkRenderPathGlobalDescriptorFrameData frame_desc{};
            frame_desc.dst_set = fr.global_set;
            frame_desc.camera_buffer = fr.camera_buffer.buffer;
            frame_desc.camera_range = sizeof(CameraUBO);
            frame_desc.lights_buffer = fr.light_buffer.buffer;
            frame_desc.lights_range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(shs::CullingLightGPU);
            frame_desc.tile_counts_buffer = fr.tile_counts_buffer.buffer;
            frame_desc.tile_counts_range = fr.tile_counts_buffer.size;
            frame_desc.tile_indices_buffer = fr.tile_indices_buffer.buffer;
            frame_desc.tile_indices_range = fr.tile_indices_buffer.size;
            frame_desc.tile_depth_ranges_buffer = fr.tile_depth_ranges_buffer.buffer;
            frame_desc.tile_depth_ranges_range = fr.tile_depth_ranges_buffer.size;
            frame_desc.shadow_lights_buffer = fr.shadow_light_buffer.buffer;
            frame_desc.shadow_lights_range = static_cast<VkDeviceSize>(kMaxLights) * sizeof(ShadowLightGPU);
            frame_desc.sampler = depth_sampler_;
            frame_desc.depth_view = depth_target_.view;
            frame_desc.sun_shadow_view = sun_shadow_target_.sampled_view;
            frame_desc.local_shadow_view = local_shadow_target_.sampled_view;
            frame_desc.point_shadow_view = local_shadow_target_.sampled_view;

            if (!shs::vk_update_render_path_global_descriptor_set(vk_->device(), frame_desc))
            {
                throw std::runtime_error("vkUpdateDescriptorSets failed (render-path global)");
            }
        }
    }

    void update_deferred_descriptor_set()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;
        if (deferred_set_ == VK_NULL_HANDLE ||
            deferred_post_a_set_ == VK_NULL_HANDLE ||
            deferred_post_b_set_ == VK_NULL_HANDLE ||
            depth_sampler_ == VK_NULL_HANDLE)
        {
            return;
        }
        if (gbuffer_target_.colors[0].view == VK_NULL_HANDLE ||
            gbuffer_target_.colors[1].view == VK_NULL_HANDLE ||
            gbuffer_target_.colors[2].view == VK_NULL_HANDLE ||
            gbuffer_target_.colors[3].view == VK_NULL_HANDLE ||
            ao_target_.view == VK_NULL_HANDLE ||
            post_target_a_.view == VK_NULL_HANDLE ||
            post_target_b_.view == VK_NULL_HANDLE)
        {
            return;
        }

        const VkImageView history_view = shs::vk_render_path_history_color_view(temporal_resources_);
        const VkImageView history_fallback_view =
            (history_view != VK_NULL_HANDLE) ? history_view : post_target_a_.view;

        const auto update_one_set = [&](VkDescriptorSet set, VkImageView post_input_view) {
            VkDescriptorImageInfo infos[7]{};
            for (uint32_t i = 0; i < 4; ++i)
            {
                infos[i].sampler = depth_sampler_;
                infos[i].imageView = gbuffer_target_.colors[i].view;
                infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
            infos[4].sampler = depth_sampler_;
            infos[4].imageView = history_fallback_view;
            infos[4].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            infos[5].sampler = depth_sampler_;
            infos[5].imageView = ao_target_.view;
            infos[5].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            infos[6].sampler = depth_sampler_;
            infos[6].imageView = post_input_view;
            infos[6].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            std::array<VkWriteDescriptorSet, 7> writes{};
            for (uint32_t i = 0; i < 7; ++i)
            {
                writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[i].dstSet = set;
                writes[i].dstBinding = i;
                writes[i].descriptorCount = 1;
                writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[i].pImageInfo = &infos[i];
            }

            vkUpdateDescriptorSets(
                vk_->device(),
                static_cast<uint32_t>(writes.size()),
                writes.data(),
                0,
                nullptr);
        };

        update_one_set(deferred_set_, history_fallback_view);
        update_one_set(deferred_post_a_set_, post_target_a_.view);
        update_one_set(deferred_post_b_set_, post_target_b_.view);
    }

    VkDescriptorSet post_source_descriptor_set_from_context(
        const shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>& ctx) const
    {
        if (ctx.post_color_source == 1u) return deferred_post_a_set_;
        if (ctx.post_color_source == 2u) return deferred_post_b_set_;
        return VK_NULL_HANDLE;
    }

    VkImageView post_source_view_from_context(
        const shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>& ctx) const
    {
        if (ctx.post_color_source == 1u) return post_target_a_.view;
        if (ctx.post_color_source == 2u) return post_target_b_.view;
        return VK_NULL_HANDLE;
    }

    void destroy_pipelines()
    {
        if (!vk_ || vk_->device() == VK_NULL_HANDLE) return;

        auto destroy_pipeline = [&](VkPipeline& p) {
            if (p != VK_NULL_HANDLE)
            {
                vkDestroyPipeline(vk_->device(), p, nullptr);
                p = VK_NULL_HANDLE;
            }
        };
        auto destroy_layout = [&](VkPipelineLayout& l) {
            if (l != VK_NULL_HANDLE)
            {
                vkDestroyPipelineLayout(vk_->device(), l, nullptr);
                l = VK_NULL_HANDLE;
            }
        };

        destroy_pipeline(depth_pipeline_);
        destroy_layout(depth_pipeline_layout_);
        destroy_pipeline(shadow_pipeline_);
        destroy_layout(shadow_pipeline_layout_);

        destroy_pipeline(scene_pipeline_);
        destroy_pipeline(scene_wire_pipeline_);
        destroy_layout(scene_pipeline_layout_);
        destroy_pipeline(gbuffer_pipeline_);
        destroy_layout(gbuffer_pipeline_layout_);
        destroy_pipeline(ssao_pipeline_);
        destroy_layout(ssao_pipeline_layout_);
        destroy_pipeline(deferred_lighting_post_pipeline_);
        destroy_pipeline(deferred_lighting_pipeline_);
        destroy_pipeline(motion_blur_pipeline_);
        destroy_pipeline(motion_blur_scene_pipeline_);
        destroy_pipeline(dof_pipeline_);
        destroy_layout(deferred_lighting_pipeline_layout_);

        destroy_pipeline(depth_reduce_pipeline_);
        destroy_pipeline(compute_pipeline_);
        destroy_layout(compute_pipeline_layout_);

        pipeline_gen_ = 0;
    }

    void create_pipelines(bool force, const char* reason = "runtime")
    {
        if (!force && scene_pipeline_ != VK_NULL_HANDLE && pipeline_gen_ == vk_->swapchain_generation()) return;

        destroy_pipelines();

        const std::vector<char> shadow_vs_code = shs::vk_read_binary_file(SHS_VK_FP_SHADOW_VERT_SPV);
        const std::vector<char> scene_vs_code = shs::vk_read_binary_file(SHS_VK_FP_SCENE_VERT_SPV);
        const std::vector<char> scene_fs_code = shs::vk_read_binary_file(SHS_VK_FP_SCENE_FRAG_SPV);
        const std::vector<char> gbuffer_fs_code = shs::vk_read_binary_file(SHS_VK_FP_GBUFFER_FRAG_SPV);
        const std::vector<char> deferred_vs_code = shs::vk_read_binary_file(SHS_VK_FP_DEFERRED_VERT_SPV);
        const std::vector<char> ssao_fs_code = shs::vk_read_binary_file(SHS_VK_FP_SSAO_FRAG_SPV);
        const std::vector<char> deferred_fs_code = shs::vk_read_binary_file(SHS_VK_FP_DEFERRED_FRAG_SPV);
        const std::vector<char> motion_blur_fs_code = shs::vk_read_binary_file(SHS_VK_FP_MOTION_BLUR_FRAG_SPV);
        const std::vector<char> dof_fs_code = shs::vk_read_binary_file(SHS_VK_FP_DOF_FRAG_SPV);
        const std::vector<char> depth_reduce_cs_code = shs::vk_read_binary_file(SHS_VK_FP_DEPTH_REDUCE_COMP_SPV);
        const std::vector<char> cull_cs_code = shs::vk_read_binary_file(SHS_VK_FP_LIGHT_CULL_COMP_SPV);

        VkShaderModule shadow_vs = shs::vk_create_shader_module(vk_->device(), shadow_vs_code);
        VkShaderModule scene_vs = shs::vk_create_shader_module(vk_->device(), scene_vs_code);
        VkShaderModule scene_fs = shs::vk_create_shader_module(vk_->device(), scene_fs_code);
        VkShaderModule gbuffer_fs = shs::vk_create_shader_module(vk_->device(), gbuffer_fs_code);
        VkShaderModule deferred_vs = shs::vk_create_shader_module(vk_->device(), deferred_vs_code);
        VkShaderModule ssao_fs = shs::vk_create_shader_module(vk_->device(), ssao_fs_code);
        VkShaderModule deferred_fs = shs::vk_create_shader_module(vk_->device(), deferred_fs_code);
        VkShaderModule motion_blur_fs = shs::vk_create_shader_module(vk_->device(), motion_blur_fs_code);
        VkShaderModule dof_fs = shs::vk_create_shader_module(vk_->device(), dof_fs_code);
        VkShaderModule depth_reduce_cs = shs::vk_create_shader_module(vk_->device(), depth_reduce_cs_code);
        VkShaderModule cull_cs = shs::vk_create_shader_module(vk_->device(), cull_cs_code);

        const auto cleanup_modules = [&]() {
            if (shadow_vs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), shadow_vs, nullptr);
            if (scene_vs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), scene_vs, nullptr);
            if (scene_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), scene_fs, nullptr);
            if (gbuffer_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), gbuffer_fs, nullptr);
            if (deferred_vs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), deferred_vs, nullptr);
            if (ssao_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), ssao_fs, nullptr);
            if (deferred_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), deferred_fs, nullptr);
            if (motion_blur_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), motion_blur_fs, nullptr);
            if (dof_fs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), dof_fs, nullptr);
            if (depth_reduce_cs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), depth_reduce_cs, nullptr);
            if (cull_cs != VK_NULL_HANDLE) vkDestroyShaderModule(vk_->device(), cull_cs, nullptr);
        };

        VkPushConstantRange shadow_pc{};
        shadow_pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        shadow_pc.offset = 0;
        shadow_pc.size = sizeof(ShadowPush);

        VkPipelineLayoutCreateInfo sh_pl{};
        sh_pl.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        sh_pl.pushConstantRangeCount = 1;
        sh_pl.pPushConstantRanges = &shadow_pc;
        if (vkCreatePipelineLayout(vk_->device(), &sh_pl, nullptr, &shadow_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (shadow)");
        }

        VkPushConstantRange draw_pc{};
        draw_pc.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        draw_pc.offset = 0;
        draw_pc.size = sizeof(DrawPush);

        VkPipelineLayoutCreateInfo pli{};
        pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pli.setLayoutCount = 1;
        pli.pSetLayouts = &global_set_layout_;
        pli.pushConstantRangeCount = 1;
        pli.pPushConstantRanges = &draw_pc;
        if (vkCreatePipelineLayout(vk_->device(), &pli, nullptr, &depth_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (depth)");
        }
        if (vkCreatePipelineLayout(vk_->device(), &pli, nullptr, &scene_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (scene)");
        }
        if (vkCreatePipelineLayout(vk_->device(), &pli, nullptr, &gbuffer_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (gbuffer)");
        }

        std::array<VkDescriptorSetLayout, 2> deferred_set_layouts = {global_set_layout_, deferred_set_layout_};
        VkPipelineLayoutCreateInfo dli{};
        dli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        dli.setLayoutCount = static_cast<uint32_t>(deferred_set_layouts.size());
        dli.pSetLayouts = deferred_set_layouts.data();
        if (vkCreatePipelineLayout(vk_->device(), &dli, nullptr, &ssao_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (ssao)");
        }
        if (vkCreatePipelineLayout(vk_->device(), &dli, nullptr, &deferred_lighting_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (deferred)");
        }

        VkPipelineLayoutCreateInfo cli{};
        cli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        cli.setLayoutCount = 1;
        cli.pSetLayouts = &global_set_layout_;
        if (vkCreatePipelineLayout(vk_->device(), &cli, nullptr, &compute_pipeline_layout_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreatePipelineLayout failed (compute)");
        }

        VkPipelineShaderStageCreateInfo shadow_stage{};
        shadow_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shadow_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        shadow_stage.module = shadow_vs;
        shadow_stage.pName = "main";

        VkPipelineShaderStageCreateInfo depth_stage{};
        depth_stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        depth_stage.stage = VK_SHADER_STAGE_VERTEX_BIT;
        depth_stage.module = scene_vs;
        depth_stage.pName = "main";

        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        VkVertexInputAttributeDescription attrs[2]{};
        attrs[0].location = 0;
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);
        attrs[1].location = 1;
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, normal);

        VkPipelineVertexInputStateCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vi.vertexBindingDescriptionCount = 1;
        vi.pVertexBindingDescriptions = &binding;
        vi.vertexAttributeDescriptionCount = 2;
        vi.pVertexAttributeDescriptions = attrs;

        VkPipelineInputAssemblyStateCreateInfo ia{};
        ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo vp{};
        vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        vp.viewportCount = 1;
        vp.scissorCount = 1;

        VkPipelineRasterizationStateCreateInfo rs{};
        rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = VK_CULL_MODE_NONE;
        rs.frontFace = VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.0f;

        VkPipelineMultisampleStateCreateInfo ms{};
        ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineDepthStencilStateCreateInfo ds_depth{};
        ds_depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds_depth.depthTestEnable = VK_TRUE;
        ds_depth.depthWriteEnable = VK_TRUE;
        ds_depth.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkDynamicState dyn_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dyn{};
        dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dyn.dynamicStateCount = 2;
        dyn.pDynamicStates = dyn_states;

        VkGraphicsPipelineCreateInfo gp_shadow{};
        gp_shadow.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_shadow.stageCount = 1;
        gp_shadow.pStages = &shadow_stage;
        gp_shadow.pVertexInputState = &vi;
        gp_shadow.pInputAssemblyState = &ia;
        gp_shadow.pViewportState = &vp;
        gp_shadow.pRasterizationState = &rs;
        gp_shadow.pMultisampleState = &ms;
        gp_shadow.pDepthStencilState = &ds_depth;
        gp_shadow.pDynamicState = &dyn;
        gp_shadow.layout = shadow_pipeline_layout_;
        gp_shadow.renderPass = sun_shadow_target_.render_pass;
        gp_shadow.subpass = 0;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_shadow, nullptr, &shadow_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (shadow)");
        }

        VkGraphicsPipelineCreateInfo gp_depth{};
        gp_depth.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_depth.stageCount = 1;
        gp_depth.pStages = &depth_stage;
        gp_depth.pVertexInputState = &vi;
        gp_depth.pInputAssemblyState = &ia;
        gp_depth.pViewportState = &vp;
        gp_depth.pRasterizationState = &rs;
        gp_depth.pMultisampleState = &ms;
        gp_depth.pDepthStencilState = &ds_depth;
        gp_depth.pDynamicState = &dyn;
        gp_depth.layout = depth_pipeline_layout_;
        gp_depth.renderPass = depth_target_.render_pass;
        gp_depth.subpass = 0;

        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_depth, nullptr, &depth_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (depth)");
        }

        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = scene_vs;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = scene_fs;
        stages[1].pName = "main";

        VkPipelineDepthStencilStateCreateInfo ds_scene{};
        ds_scene.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds_scene.depthTestEnable = vk_->has_depth_attachment() ? VK_TRUE : VK_FALSE;
        ds_scene.depthWriteEnable = vk_->has_depth_attachment() ? VK_TRUE : VK_FALSE;
        ds_scene.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

        VkPipelineColorBlendAttachmentState cba{};
        cba.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        cba.blendEnable = VK_FALSE;

        VkPipelineColorBlendStateCreateInfo cb{};
        cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        cb.attachmentCount = 1;
        cb.pAttachments = &cba;

        VkGraphicsPipelineCreateInfo gp_scene{};
        gp_scene.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_scene.stageCount = 2;
        gp_scene.pStages = stages;
        gp_scene.pVertexInputState = &vi;
        gp_scene.pInputAssemblyState = &ia;
        gp_scene.pViewportState = &vp;
        gp_scene.pRasterizationState = &rs;
        gp_scene.pMultisampleState = &ms;
        gp_scene.pDepthStencilState = &ds_scene;
        gp_scene.pColorBlendState = &cb;
        gp_scene.pDynamicState = &dyn;
        gp_scene.layout = scene_pipeline_layout_;
        gp_scene.renderPass = vk_->render_pass();
        gp_scene.subpass = 0;

        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_scene, nullptr, &scene_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (scene)");
        }

        VkPipelineShaderStageCreateInfo gbuffer_stages[2]{};
        gbuffer_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        gbuffer_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        gbuffer_stages[0].module = scene_vs;
        gbuffer_stages[0].pName = "main";
        gbuffer_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        gbuffer_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        gbuffer_stages[1].module = gbuffer_fs;
        gbuffer_stages[1].pName = "main";

        std::array<VkPipelineColorBlendAttachmentState, 4> gbuffer_cba{};
        for (auto& a : gbuffer_cba)
        {
            a = cba;
        }
        VkPipelineColorBlendStateCreateInfo gbuffer_cb{};
        gbuffer_cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        gbuffer_cb.attachmentCount = static_cast<uint32_t>(gbuffer_cba.size());
        gbuffer_cb.pAttachments = gbuffer_cba.data();

        VkGraphicsPipelineCreateInfo gp_gbuffer = gp_scene;
        gp_gbuffer.pStages = gbuffer_stages;
        gp_gbuffer.pColorBlendState = &gbuffer_cb;
        gp_gbuffer.pDepthStencilState = &ds_depth;
        gp_gbuffer.layout = gbuffer_pipeline_layout_;
        gp_gbuffer.renderPass = gbuffer_target_.render_pass;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_gbuffer, nullptr, &gbuffer_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (gbuffer)");
        }

        VkPipelineShaderStageCreateInfo ssao_stages[2]{};
        ssao_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssao_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        ssao_stages[0].module = deferred_vs;
        ssao_stages[0].pName = "main";
        ssao_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        ssao_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        ssao_stages[1].module = ssao_fs;
        ssao_stages[1].pName = "main";

        VkPipelineShaderStageCreateInfo deferred_stages[2]{};
        deferred_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        deferred_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        deferred_stages[0].module = deferred_vs;
        deferred_stages[0].pName = "main";
        deferred_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        deferred_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        deferred_stages[1].module = deferred_fs;
        deferred_stages[1].pName = "main";

        VkPipelineVertexInputStateCreateInfo vi_fullscreen{};
        vi_fullscreen.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

        VkPipelineInputAssemblyStateCreateInfo ia_fullscreen{};
        ia_fullscreen.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        ia_fullscreen.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineDepthStencilStateCreateInfo ds_deferred{};
        ds_deferred.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        ds_deferred.depthTestEnable = VK_FALSE;
        ds_deferred.depthWriteEnable = VK_FALSE;
        ds_deferred.depthCompareOp = VK_COMPARE_OP_ALWAYS;

        VkGraphicsPipelineCreateInfo gp_ssao{};
        gp_ssao.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_ssao.stageCount = 2;
        gp_ssao.pStages = ssao_stages;
        gp_ssao.pVertexInputState = &vi_fullscreen;
        gp_ssao.pInputAssemblyState = &ia_fullscreen;
        gp_ssao.pViewportState = &vp;
        gp_ssao.pRasterizationState = &rs;
        gp_ssao.pMultisampleState = &ms;
        gp_ssao.pDepthStencilState = &ds_deferred;
        gp_ssao.pColorBlendState = &cb;
        gp_ssao.pDynamicState = &dyn;
        gp_ssao.layout = ssao_pipeline_layout_;
        gp_ssao.renderPass = ao_target_.render_pass;
        gp_ssao.subpass = 0;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_ssao, nullptr, &ssao_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (ssao)");
        }

        VkGraphicsPipelineCreateInfo gp_deferred{};
        gp_deferred.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        gp_deferred.stageCount = 2;
        gp_deferred.pStages = deferred_stages;
        gp_deferred.pVertexInputState = &vi_fullscreen;
        gp_deferred.pInputAssemblyState = &ia_fullscreen;
        gp_deferred.pViewportState = &vp;
        gp_deferred.pRasterizationState = &rs;
        gp_deferred.pMultisampleState = &ms;
        gp_deferred.pDepthStencilState = &ds_deferred;
        gp_deferred.pColorBlendState = &cb;
        gp_deferred.pDynamicState = &dyn;
        gp_deferred.layout = deferred_lighting_pipeline_layout_;
        gp_deferred.renderPass = vk_->render_pass();
        gp_deferred.subpass = 0;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_deferred, nullptr, &deferred_lighting_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (deferred)");
        }

        VkGraphicsPipelineCreateInfo gp_deferred_post = gp_deferred;
        gp_deferred_post.renderPass = post_target_a_.render_pass;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_deferred_post, nullptr, &deferred_lighting_post_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (deferred post)");
        }

        VkPipelineShaderStageCreateInfo motion_blur_stages[2]{};
        motion_blur_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        motion_blur_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        motion_blur_stages[0].module = deferred_vs;
        motion_blur_stages[0].pName = "main";
        motion_blur_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        motion_blur_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        motion_blur_stages[1].module = motion_blur_fs;
        motion_blur_stages[1].pName = "main";

        VkGraphicsPipelineCreateInfo gp_motion_blur = gp_deferred;
        gp_motion_blur.pStages = motion_blur_stages;
        gp_motion_blur.renderPass = post_target_b_.render_pass;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_motion_blur, nullptr, &motion_blur_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (motion blur)");
        }

        VkGraphicsPipelineCreateInfo gp_motion_blur_scene = gp_deferred;
        gp_motion_blur_scene.pStages = motion_blur_stages;
        gp_motion_blur_scene.renderPass = vk_->render_pass();
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_motion_blur_scene, nullptr, &motion_blur_scene_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (motion blur scene)");
        }

        VkPipelineShaderStageCreateInfo dof_stages[2]{};
        dof_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        dof_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        dof_stages[0].module = deferred_vs;
        dof_stages[0].pName = "main";
        dof_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        dof_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        dof_stages[1].module = dof_fs;
        dof_stages[1].pName = "main";

        VkGraphicsPipelineCreateInfo gp_dof = gp_deferred;
        gp_dof.pStages = dof_stages;
        gp_dof.renderPass = vk_->render_pass();
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_dof, nullptr, &dof_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (dof)");
        }

        VkPipelineInputAssemblyStateCreateInfo ia_lines = ia;
        ia_lines.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

        VkPipelineDepthStencilStateCreateInfo ds_wire = ds_scene;
        ds_wire.depthWriteEnable = VK_FALSE;

        VkGraphicsPipelineCreateInfo gp_scene_wire = gp_scene;
        gp_scene_wire.pInputAssemblyState = &ia_lines;
        gp_scene_wire.pDepthStencilState = &ds_wire;
        if (vkCreateGraphicsPipelines(vk_->device(), VK_NULL_HANDLE, 1, &gp_scene_wire, nullptr, &scene_wire_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateGraphicsPipelines failed (scene wire)");
        }

        VkComputePipelineCreateInfo cp{};
        cp.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cp.layout = compute_pipeline_layout_;
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = depth_reduce_cs;
        cp.stage.pName = "main";
        if (vkCreateComputePipelines(vk_->device(), VK_NULL_HANDLE, 1, &cp, nullptr, &depth_reduce_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateComputePipelines failed (depth reduce)");
        }

        cp.stage.module = cull_cs;
        cp.stage.pName = "main";
        if (vkCreateComputePipelines(vk_->device(), VK_NULL_HANDLE, 1, &cp, nullptr, &compute_pipeline_) != VK_SUCCESS)
        {
            cleanup_modules();
            throw std::runtime_error("vkCreateComputePipelines failed");
        }

        cleanup_modules();
        pipeline_gen_ = vk_->swapchain_generation();
        ++pipeline_rebuild_count_;
        pipeline_last_rebuild_reason_ = (reason && *reason) ? reason : "runtime";
    }

    shs::RenderPathLightGridRuntimeLayout make_active_light_grid_runtime_layout(uint32_t frame_w, uint32_t frame_h) const
    {
        return shs::make_render_path_light_grid_runtime_layout(
            render_path_executor_.active_plan(),
            render_path_executor_.active_recipe(),
            render_path_executor_.active_resource_plan(),
            frame_w,
            frame_h);
    }

    void ensure_render_targets(uint32_t w, uint32_t h)
    {
        if (w == 0 || h == 0) return;

        const VkFormat swapchain_fmt = vk_->swapchain_format();

        const shs::RenderPathLightGridRuntimeLayout desired_layout =
            make_active_light_grid_runtime_layout(w, h);

        const bool extent_matches =
            depth_target_.w == w &&
            depth_target_.h == h &&
            gbuffer_target_.w == w &&
            gbuffer_target_.h == h &&
            ao_target_.w == w &&
            ao_target_.h == h &&
            post_target_a_.w == w &&
            post_target_a_.h == h &&
            post_target_b_.w == w &&
            post_target_b_.h == h;
        const bool format_matches =
            post_target_a_.format == swapchain_fmt &&
            post_target_b_.format == swapchain_fmt;
        const bool temporal_matches = shs::vk_render_path_temporal_resources_allocation_equal(
            temporal_resources_,
            render_path_executor_.active_resource_plan(),
            w,
            h,
            swapchain_fmt);
        const bool light_grid_matches =
            shs::light_grid_runtime_layout_allocation_equal(light_grid_layout_, desired_layout);

        if (extent_matches &&
            format_matches &&
            temporal_matches &&
            light_grid_matches)
        {
            return;
        }
        ++render_target_rebuild_count_;
        if (!extent_matches) render_target_last_rebuild_reason_ = "extent";
        else if (!format_matches) render_target_last_rebuild_reason_ = "format";
        else if (!temporal_matches) render_target_last_rebuild_reason_ = "temporal";
        else if (!light_grid_matches) render_target_last_rebuild_reason_ = "light-grid";
        else render_target_last_rebuild_reason_ = "runtime";
        create_depth_target(w, h);
        create_gbuffer_target(w, h);
        create_ao_target(w, h);
        create_post_color_target(post_target_a_, w, h, swapchain_fmt);
        create_post_color_target(post_target_b_, w, h, swapchain_fmt);
        if (!shs::vk_ensure_render_path_temporal_resources(
                vk_->device(),
                vk_->physical_device(),
                render_path_executor_.active_resource_plan(),
                w,
                h,
                swapchain_fmt,
                temporal_resources_))
        {
            throw std::runtime_error("Failed to ensure temporal history resources");
        }
        ensure_shadow_targets();
        light_grid_layout_ = desired_layout;
        light_tile_size_ = light_grid_layout_.tile_size;
        cluster_z_slices_ = light_grid_layout_.cluster_z_slices;
        tile_w_ = light_grid_layout_.tile_count_x;
        tile_h_ = light_grid_layout_.tile_count_y;
        create_or_resize_tile_buffers(light_grid_layout_);
        update_deferred_descriptor_set();
        update_global_descriptor_sets();
        create_pipelines(true, "targets-recreated");
    }

    void refresh_active_composition_recipe()
    {
        const shs::RenderPathPreset active_path = shs::render_path_preset_for_mode(active_technique_);
        if (!composition_cycle_order_.empty())
        {
            size_t any_match = composition_cycle_order_.size();
            for (size_t i = 0; i < composition_cycle_order_.size(); ++i)
            {
                const auto& c = composition_cycle_order_[i];
                if (c.path_preset == active_path && c.technique_preset == render_technique_preset_)
                {
                    if (any_match == composition_cycle_order_.size()) any_match = i;
                    if (c.post_stack == shs::RenderCompositionPostStackPreset::Default)
                    {
                        active_composition_index_ = i;
                        active_composition_recipe_ = c;
                        return;
                    }
                }
            }
            if (any_match < composition_cycle_order_.size())
            {
                active_composition_index_ = any_match;
                active_composition_recipe_ = composition_cycle_order_[any_match];
                return;
            }
        }
        active_composition_recipe_ = shs::make_builtin_render_composition_recipe(
            active_path,
            render_technique_preset_,
            "composition_vk");
    }

    void apply_composition_post_stack_state()
    {
        const shs::RenderCompositionPostStackState stack =
            shs::resolve_render_composition_post_stack_state(
                active_composition_recipe_.path_preset,
                active_composition_recipe_.post_stack);
        composition_ssao_enabled_ = stack.enable_ssao;
        composition_taa_enabled_ = stack.enable_taa;
        composition_motion_blur_enabled_ = stack.enable_motion_blur;
        composition_depth_of_field_enabled_ = stack.enable_depth_of_field;
    }

    bool active_ssao_pass_enabled() const
    {
        return path_has_ssao_pass_ && composition_ssao_enabled_;
    }

    bool active_taa_pass_enabled() const
    {
        return path_has_taa_pass_ && composition_taa_enabled_;
    }

    bool active_motion_blur_pass_enabled() const
    {
        return path_has_motion_blur_pass_ && composition_motion_blur_enabled_;
    }

    bool active_depth_of_field_pass_enabled() const
    {
        return path_has_depth_of_field_pass_ && composition_depth_of_field_enabled_;
    }

    void apply_render_technique_preset(
        shs::RenderTechniquePreset preset,
        bool refresh_composition = true)
    {
        render_technique_preset_ = preset;
        render_technique_recipe_ = shs::make_builtin_render_technique_recipe(
            render_technique_preset_,
            "render_tech_vk");
        shading_variant_ = shs::render_technique_shader_variant(render_technique_preset_);
        tonemap_exposure_ = render_technique_recipe_.tonemap_exposure;
        tonemap_gamma_ = render_technique_recipe_.tonemap_gamma;
        if (refresh_composition)
        {
            refresh_active_composition_recipe();
        }
    }

    size_t find_composition_index(
        shs::RenderPathPreset path_preset,
        shs::RenderTechniquePreset technique_preset) const
    {
        for (size_t i = 0; i < composition_cycle_order_.size(); ++i)
        {
            const auto& c = composition_cycle_order_[i];
            if (c.path_preset == path_preset && c.technique_preset == technique_preset)
            {
                return i;
            }
        }
        return 0u;
    }

    void apply_render_composition_resolved(const shs::RenderCompositionResolved& resolved)
    {
        apply_render_technique_preset(resolved.composition.technique_preset, false);
        const shs::RenderPathResolvedState resolved_path_state =
            render_path_executor_.resolve_recipe(resolved.path_recipe, ctx_, &pass_contract_registry_);
        const bool plan_valid = render_path_executor_.apply_resolved(resolved_path_state);
        (void)consume_active_render_path_apply_result(plan_valid);
        apply_composition_post_stack_state();
    }

    bool apply_render_composition_by_index(size_t index)
    {
        if (composition_cycle_order_.empty()) return false;
        active_composition_index_ = index % composition_cycle_order_.size();
        const shs::RenderCompositionRecipe& composition = composition_cycle_order_[active_composition_index_];

        apply_render_technique_preset(composition.technique_preset, false);
        shs::RenderCompositionResolved resolved =
            shs::resolve_builtin_render_composition_recipe(
                composition,
                shs::RenderBackendType::Vulkan,
                "render_path_vk",
                "render_tech_vk");

        // Force Modern-Extreme specific overrides
        if (composition.name == "composition_modern_extreme")
        {
            resolved.path_recipe.light_volume_provider = shs::RenderPathLightVolumeProvider::ClusteredGrid;
            resolved.path_recipe.runtime_defaults.shadow_occlusion_enabled = true;
            resolved.path_recipe.view_culling = shs::RenderPathCullingMode::FrustumAndOcclusion;
            resolved.path_recipe.name = "path_clustered_forward_modern_extreme";
        }
        const shs::RenderPathResolvedState resolved_path_state =
            render_path_executor_.resolve_recipe(resolved.path_recipe, ctx_, &pass_contract_registry_);
        const bool plan_valid = render_path_executor_.apply_resolved(resolved_path_state);
        const bool ok = consume_active_render_path_apply_result(plan_valid);
        if (ok)
        {
            active_composition_index_ = index % composition_cycle_order_.size();
            active_composition_recipe_ = composition;
        }
        else
        {
            refresh_active_composition_recipe();
        }
        apply_composition_post_stack_state();
        return ok;
    }

    void apply_technique_profile(shs::TechniqueMode mode, const shs::TechniqueProfile& profile)
    {
        active_technique_ = mode;

        profile_depth_prepass_enabled_ = profile_has_pass(profile, shs::PassId::DepthPrepass);
        enable_light_culling_ =
            profile_has_pass(profile, shs::PassId::LightCulling) ||
            profile_has_pass(profile, shs::PassId::ClusterLightAssign);

        shs::LightCullingMode mode_hint = shs::default_light_culling_mode_for_mode(mode);
        if (!enable_light_culling_)
        {
            mode_hint = shs::LightCullingMode::None;
        }
        culling_mode_ = mode_hint;

        const bool has_forward_lighting =
            profile_has_pass(profile, shs::PassId::PBRForward) ||
            profile_has_pass(profile, shs::PassId::PBRForwardPlus) ||
            profile_has_pass(profile, shs::PassId::PBRForwardClustered);
        const bool has_deferred_lighting =
            profile_has_pass(profile, shs::PassId::DeferredLighting) ||
            profile_has_pass(profile, shs::PassId::DeferredLightingTiled);
        const bool has_gbuffer = profile_has_pass(profile, shs::PassId::GBuffer);
        path_has_ssao_pass_ = profile_has_pass(profile, shs::PassId::SSAO);
        path_has_motion_blur_pass_ = profile_has_pass(profile, shs::PassId::MotionBlur);
        path_has_depth_of_field_pass_ = profile_has_pass(profile, shs::PassId::DepthOfField);
        path_has_taa_pass_ = profile_has_pass(profile, shs::PassId::TAA);
        enable_scene_pass_ = has_forward_lighting;
        if (!has_forward_lighting && !has_deferred_lighting && !has_gbuffer)
        {
            enable_scene_pass_ = true;
        }

        temporal_settings_.accumulation_enabled = path_has_taa_pass_;
        temporal_settings_.jitter_enabled = path_has_taa_pass_;
        if (!path_has_taa_pass_)
        {
            shs::vk_render_path_invalidate_history_color(temporal_resources_);
        }

        refresh_depth_prepass_state();
        use_forward_plus_ = (culling_mode_ != shs::LightCullingMode::None);
        technique_switch_accum_sec_ = 0.0f;
        refresh_active_composition_recipe();
        apply_composition_post_stack_state();
    }

    void apply_technique_mode(shs::TechniqueMode mode)
    {
        const shs::TechniqueProfile profile = shs::make_default_technique_profile(mode);
        apply_technique_profile(mode, profile);
    }

    void init_render_path_registry()
    {
        pass_contract_registry_ =
            shs::make_standard_pass_contract_registry_for_backend(shs::RenderBackendType::Vulkan);
        pass_contract_registry_sw_ =
            shs::make_standard_pass_contract_registry_for_backend(shs::RenderBackendType::Software);
        if (pass_contract_registry_.ids().empty())
        {
            std::fprintf(stderr, "[render-path][stress][error] Standard pass contract registry is empty.\n");
        }
        if (pass_contract_registry_sw_.ids().empty())
        {
            std::fprintf(stderr, "[render-path][stress][error] Software pass contract registry is empty.\n");
        }

        const bool ok = render_path_executor_.register_builtin_presets(
            shs::RenderBackendType::Vulkan,
            "render_path_vk");
        if (!ok)
        {
            std::fprintf(stderr, "[render-path][stress][error] Failed to register one or more builtin presets.\n");
        }

        build_frame_pass_dispatcher();
        pass_dispatch_warning_emitted_ = false;
    }

    void refresh_semantic_debug_targets()
    {
        semantic_debug_targets_ = shs::collect_render_path_visual_debug_semantics(
            render_path_executor_.active_resource_plan());
        if (semantic_debug_targets_.empty())
        {
            semantic_debug_enabled_ = false;
            semantic_debug_index_ = 0u;
            active_semantic_debug_ = shs::PassSemantic::Unknown;
            return;
        }

        if (!semantic_debug_enabled_ || active_semantic_debug_ == shs::PassSemantic::Unknown)
        {
            semantic_debug_index_ = 0u;
            active_semantic_debug_ = semantic_debug_targets_[0];
            return;
        }

        for (size_t i = 0; i < semantic_debug_targets_.size(); ++i)
        {
            if (semantic_debug_targets_[i] == active_semantic_debug_)
            {
                semantic_debug_index_ = i;
                return;
            }
        }

        semantic_debug_index_ = 0u;
        active_semantic_debug_ = semantic_debug_targets_[0];
    }

    void cycle_semantic_debug_target()
    {
        refresh_semantic_debug_targets();
        if (semantic_debug_targets_.empty())
        {
            std::fprintf(stderr, "[render-path][debug] Semantic debug target unavailable for current path.\n");
            return;
        }

        if (!semantic_debug_enabled_)
        {
            semantic_debug_enabled_ = true;
            semantic_debug_index_ = 0u;
            active_semantic_debug_ = semantic_debug_targets_[semantic_debug_index_];
        }
        else
        {
            const size_t next = semantic_debug_index_ + 1u;
            if (next >= semantic_debug_targets_.size())
            {
                semantic_debug_enabled_ = false;
                semantic_debug_index_ = 0u;
                active_semantic_debug_ = shs::PassSemantic::Unknown;
            }
            else
            {
                semantic_debug_index_ = next;
                active_semantic_debug_ = semantic_debug_targets_[semantic_debug_index_];
            }
        }

        const char* state = semantic_debug_enabled_ ? "ON" : "OFF";
        const char* semantic_name = semantic_debug_enabled_
            ? shs::pass_semantic_name(active_semantic_debug_)
            : "none";
        std::fprintf(stderr, "[render-path][debug] Semantic debug: %s (%s)\n", state, semantic_name);
    }

    void cycle_framebuffer_debug_target()
    {
        constexpr std::array<FramebufferDebugPreset, 15> kCycle = {
            FramebufferDebugPreset::FinalComposite,
            FramebufferDebugPreset::Albedo,
            FramebufferDebugPreset::Normal,
            FramebufferDebugPreset::Material,
            FramebufferDebugPreset::Depth,
            FramebufferDebugPreset::AmbientOcclusion,
            FramebufferDebugPreset::LightGrid,
            FramebufferDebugPreset::LightClusters,
            FramebufferDebugPreset::Shadow,
            FramebufferDebugPreset::ColorHDR,
            FramebufferDebugPreset::ColorLDR,
            FramebufferDebugPreset::Motion,
            FramebufferDebugPreset::DoFCircleOfConfusion,
            FramebufferDebugPreset::DoFBlur,
            FramebufferDebugPreset::DoFFactor
        };

        auto cycle_index = [&](FramebufferDebugPreset preset) -> int {
            for (size_t i = 0; i < kCycle.size(); ++i)
            {
                if (kCycle[i] == preset) return static_cast<int>(i);
            }
            return -1;
        };

        int idx = cycle_index(framebuffer_debug_preset_);
        if (idx < 0)
        {
            idx = 0;
        }
        const size_t next = (static_cast<size_t>(idx) + 1u) % kCycle.size();
        framebuffer_debug_preset_ = kCycle[next];

        const bool enabled = framebuffer_debug_preset_ != FramebufferDebugPreset::FinalComposite;
        const bool needs_motion = framebuffer_debug_preset_requires_motion_pass(framebuffer_debug_preset_);
        const bool needs_dof = framebuffer_debug_preset_requires_dof_pass(framebuffer_debug_preset_);
        bool supported =
            (!needs_motion || active_motion_blur_pass_enabled()) &&
            (!needs_dof || active_depth_of_field_pass_enabled());
        bool auto_switched_path = false;

        if (enabled && !supported && needs_dof && !active_depth_of_field_pass_enabled())
        {
            const size_t deferred_index = find_composition_index(
                shs::RenderPathPreset::Deferred,
                render_technique_preset_);
            if (apply_render_composition_by_index(deferred_index))
            {
                auto_switched_path = true;
            }
            else
            {
                const size_t tiled_index = find_composition_index(
                    shs::RenderPathPreset::TiledDeferred,
                    render_technique_preset_);
                if (apply_render_composition_by_index(tiled_index))
                {
                    auto_switched_path = true;
                }
            }

            supported =
                (!needs_motion || active_motion_blur_pass_enabled()) &&
                (!needs_dof || active_depth_of_field_pass_enabled());
        }

        const char* state = enabled ? "ON" : "OFF";
        const char* status = enabled ? (supported ? "ready" : "missing-pass") : "idle";
        std::fprintf(
            stderr,
            "[render-path][debug] Framebuffer debug (F5): %s (%s, %s)\n",
            state,
            framebuffer_debug_preset_name(framebuffer_debug_preset_),
            status);
        if (auto_switched_path)
        {
            std::fprintf(
                stderr,
                "[render-path][debug] Auto-switched to DoF-capable composition: %s\n",
                active_composition_recipe_.name.c_str());
        }
    }

    uint32_t active_semantic_debug_mode() const
    {
        const uint32_t preset_mode = semantic_debug_mode_for_framebuffer_preset(framebuffer_debug_preset_);
        if (preset_mode != 0u)
        {
            return preset_mode;
        }
        if (semantic_debug_enabled_)
        {
            return semantic_debug_mode_for_semantic(active_semantic_debug_);
        }
        return 0u;
    }

    bool consume_active_render_path_apply_result(bool plan_valid)
    {
        const shs::RenderPathExecutionPlan& plan = render_path_executor_.active_plan();
        const shs::RenderPathRecipe& recipe = render_path_executor_.active_recipe();
        const shs::RenderPathResourcePlan& resource_plan = render_path_executor_.active_resource_plan();
        const shs::RenderPathBarrierPlan& barrier_plan = render_path_executor_.active_barrier_plan();

        for (const auto& w : plan.warnings)
        {
            std::fprintf(stderr, "[render-path][stress][warn] %s\n", w.c_str());
        }
        for (const auto& e : plan.errors)
        {
            std::fprintf(stderr, "[render-path][stress][error] %s\n", e.c_str());
        }
        for (const auto& w : resource_plan.warnings)
        {
            std::fprintf(stderr, "[render-path][stress][resource-warn] %s\n", w.c_str());
        }
        for (const auto& e : resource_plan.errors)
        {
            std::fprintf(stderr, "[render-path][stress][resource-error] %s\n", e.c_str());
        }
        for (const auto& w : barrier_plan.warnings)
        {
            std::fprintf(stderr, "[render-path][stress][barrier-warn] %s\n", w.c_str());
        }
        for (const auto& e : barrier_plan.errors)
        {
            std::fprintf(stderr, "[render-path][stress][barrier-error] %s\n", e.c_str());
        }
        pass_dispatch_warning_emitted_ = false;
        refresh_semantic_debug_targets();

        light_tile_size_ = std::max(1u, recipe.light_tile_size);
        cluster_z_slices_ = std::max(1u, recipe.cluster_z_slices);
        if (const auto* grid = shs::find_render_path_resource_by_semantic(resource_plan, shs::PassSemantic::LightGrid))
        {
            light_tile_size_ = std::max(1u, grid->tile_size);
        }
        if (const auto* clusters = shs::find_render_path_resource_by_semantic(resource_plan, shs::PassSemantic::LightClusters))
        {
            cluster_z_slices_ = std::max(1u, clusters->layers);
        }
        barrier_edge_count_ = static_cast<uint32_t>(barrier_plan.edges.size());
        barrier_memory_edge_count_ = shs::render_path_barrier_memory_edge_count(barrier_plan);
        barrier_layout_edge_count_ = shs::render_path_barrier_layout_transition_count(barrier_plan);
        barrier_alias_class_count_ = static_cast<uint32_t>(barrier_plan.alias_classes.size());
        barrier_alias_slot_count_ = shs::render_path_alias_slot_count(barrier_plan);

        if (!plan_valid)
        {
            std::fprintf(
                stderr,
                "[render-path][stress] Recipe '%s' invalid. Falling back to default technique profile.\n",
                recipe.name.c_str());
            apply_technique_mode(recipe.technique_mode);
            return false;
        }

        const shs::TechniqueProfile profile = shs::make_technique_profile(plan);
        apply_technique_profile(plan.technique_mode, profile);
        enable_scene_occlusion_ = plan.runtime_state.view_occlusion_enabled;
        enable_light_occlusion_ = plan.runtime_state.shadow_occlusion_enabled;
        shadow_settings_.enable = plan.runtime_state.enable_shadows;

        std::fprintf(
            stderr,
            "[render-path][stress] Applied recipe '%s' (%s), passes:%zu, barriers:%u(mem:%u layout:%u), alias-class:%u slots:%u.\n",
            plan.recipe_name.c_str(),
            plan_valid ? "valid" : "invalid",
            plan.pass_chain.size(),
            barrier_edge_count_,
            barrier_memory_edge_count_,
            barrier_layout_edge_count_,
            barrier_alias_class_count_,
            barrier_alias_slot_count_);
        return true;
    }

    bool apply_render_path_recipe_by_index(size_t index)
    {
        if (!render_path_executor_.has_recipes())
        {
            apply_technique_mode(shs::TechniqueMode::Deferred);
            return false;
        }

        const shs::RenderPathResolvedState resolved_path_state =
            render_path_executor_.resolve_index(index, ctx_, &pass_contract_registry_);
        const bool plan_valid = render_path_executor_.apply_resolved(resolved_path_state);
        return consume_active_render_path_apply_result(plan_valid);
    }

    void cycle_render_path_recipe()
    {
        if (!render_path_executor_.has_recipes()) return;
        (void)apply_render_path_recipe_by_index(render_path_executor_.active_index() + 1u);
    }

    void cycle_lighting_technique()
    {
        apply_render_technique_preset(
            shs::next_render_technique_preset(render_technique_preset_));
        technique_switch_accum_sec_ = 0.0f;
    }

    void cycle_render_composition_recipe()
    {
        // Locked to Forward Classic
    }

    void configure_render_path_defaults()
    {
        init_render_path_registry();
        
        // Explicitly setup Forward Classic + SSAO
        shs::RenderCompositionRecipe recipe{};
        recipe.name = "forward_classic_ssao";
        recipe.path_preset = shs::RenderPathPreset::Forward;
        recipe.technique_preset = shs::RenderTechniquePreset::PBR;
        recipe.post_stack = shs::RenderCompositionPostStackPreset::Default;

        shs::RenderCompositionResolved resolved = shs::resolve_builtin_render_composition_recipe(
            recipe,
            shs::RenderBackendType::Vulkan,
            "path_vk",
            "tech_vk");

        // Override pass chain to include SSAO
        resolved.path_recipe.pass_chain = {
            shs::make_render_path_pass_entry(shs::PassId::ShadowMap, true),
            shs::make_render_path_pass_entry(shs::PassId::DepthPrepass, true),
            shs::make_render_path_pass_entry(shs::PassId::SSAO, true),
            shs::make_render_path_pass_entry(shs::PassId::PBRForward, true),
            shs::make_render_path_pass_entry(shs::PassId::Tonemap, true)
        };

        active_composition_recipe_ = recipe;
        apply_render_composition_resolved(resolved);

        print_composition_catalog();
    }

    void refresh_depth_prepass_state()
    {
        const bool needs_depth_for_culling =
            enable_light_culling_ &&
            culling_mode_ == shs::LightCullingMode::TiledDepthRange;
        enable_depth_prepass_ = profile_depth_prepass_enabled_ || needs_depth_for_culling;
    }

    void update_culling_debug_stats(uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot) || tile_w_ == 0 || tile_h_ == 0)
        {
            cull_debug_total_refs_ = 0;
            cull_debug_non_empty_lists_ = 0;
            cull_debug_list_count_ = 0;
            cull_debug_max_list_size_ = 0;
            return;
        }
        const GpuBuffer& tile_counts_buffer = frame_resources_.at_slot(frame_slot).tile_counts_buffer;
        if (!tile_counts_buffer.mapped || tile_counts_buffer.size < sizeof(uint32_t))
        {
            cull_debug_total_refs_ = 0;
            cull_debug_non_empty_lists_ = 0;
            cull_debug_list_count_ = 0;
            cull_debug_max_list_size_ = 0;
            return;
        }

        uint32_t list_count = tile_w_ * tile_h_;
        if (culling_mode_ == shs::LightCullingMode::Clustered)
        {
            list_count *= cluster_z_slices_;
        }
        const uint32_t capacity = static_cast<uint32_t>(tile_counts_buffer.size / sizeof(uint32_t));
        list_count = std::min(list_count, capacity);

        const uint32_t* counts = reinterpret_cast<const uint32_t*>(tile_counts_buffer.mapped);
        uint64_t total_refs = 0;
        uint32_t non_empty = 0;
        uint32_t max_list = 0;
        for (uint32_t i = 0; i < list_count; ++i)
        {
            const uint32_t c = std::min(counts[i], kMaxLightsPerTile);
            total_refs += static_cast<uint64_t>(c);
            if (c > 0) ++non_empty;
            if (c > max_list) max_list = c;
        }

        cull_debug_total_refs_ = total_refs;
        cull_debug_non_empty_lists_ = non_empty;
        cull_debug_list_count_ = list_count;
        cull_debug_max_list_size_ = max_list;
    }

    void rebuild_instance_cull_shapes()
    {
        if (instance_cull_shapes_.size() != instances_.size())
        {
            instance_cull_shapes_.resize(instances_.size());
        }
        for (size_t i = 0; i < instances_.size(); ++i)
        {
            shs::SceneShape shape{};
            shape.shape = cull_shape_for_mesh(instances_[i].mesh_kind);
            shape.transform = shs::jolt::to_jph(instance_models_[i]);
            shape.stable_id = static_cast<uint32_t>(i);
            instance_cull_shapes_[i] = shape;
        }
    }

    void update_visibility_from_cell(const shs::CullingCell& cell)
    {
        if (instance_visible_mask_.size() != instances_.size())
        {
            instance_visible_mask_.assign(instances_.size(), 1u);
        }

        if (instance_cull_shapes_.size() != instances_.size())
        {
            rebuild_instance_cull_shapes();
        }

        const shs::CullResult instance_cull = shs::cull_vs_cell(std::span<const shs::SceneShape>{instance_cull_shapes_}, cell);
        frustum_visible_instance_indices_.clear();
        frustum_visible_instance_indices_.reserve(instances_.size());
        uint32_t visible_instances = 0;
        const size_t cull_count = std::min(instance_visible_mask_.size(), instance_cull.classes.size());
        for (size_t i = 0; i < cull_count; ++i)
        {
            const bool visible = shs::cull_class_is_visible(instance_cull.classes[i], true);
            instance_visible_mask_[i] = visible ? 1u : 0u;
            if (visible)
            {
                ++visible_instances;
                frustum_visible_instance_indices_.push_back(static_cast<uint32_t>(i));
            }
        }
        for (size_t i = cull_count; i < instance_visible_mask_.size(); ++i)
        {
            instance_visible_mask_[i] = 0u;
        }
        visible_instance_count_ = visible_instances;

        const shs::AABB floor_ws = shs::transform_aabb(floor_local_aabb_, floor_model_);
        const shs::CullClass floor_class = shs::classify_aabb_vs_cell(floor_ws, cell);
        floor_visible_ = shs::cull_class_is_visible(floor_class, true);
    }

    void apply_scene_software_occlusion()
    {
        if (!enable_scene_occlusion_)
        {
            return;
        }

        const size_t expected = static_cast<size_t>(kSceneOccW) * static_cast<size_t>(kSceneOccH);
        if (scene_occlusion_depth_.size() != expected)
        {
            scene_occlusion_depth_.assign(expected, 1.0f);
        }
        else
        {
            std::fill(scene_occlusion_depth_.begin(), scene_occlusion_depth_.end(), 1.0f);
        }

        std::vector<uint32_t> sorted = frustum_visible_instance_indices_;
        std::sort(
            sorted.begin(),
            sorted.end(),
            [&](uint32_t a, uint32_t b)
            {
                if (a >= instance_models_.size() || b >= instance_models_.size()) return a < b;
                const shs::AABB aa = shs::transform_aabb(local_aabb_for_mesh(instances_[a].mesh_kind), instance_models_[a]);
                const shs::AABB bb = shs::transform_aabb(local_aabb_for_mesh(instances_[b].mesh_kind), instance_models_[b]);
                const float da = shs::culling_sw::view_depth_of_aabb_center(aa, camera_ubo_.view);
                const float db = shs::culling_sw::view_depth_of_aabb_center(bb, camera_ubo_.view);
                return da < db;
            });

        uint32_t visible_instances = 0;
        for (const uint32_t idx : sorted)
        {
            if (idx >= instance_models_.size() || idx >= instance_visible_mask_.size()) continue;
            const shs::AABB world_box = shs::transform_aabb(local_aabb_for_mesh(instances_[idx].mesh_kind), instance_models_[idx]);
            const shs::culling_sw::ScreenRectDepth rect = shs::culling_sw::project_aabb_to_screen_rect(
                world_box,
                camera_ubo_.view_proj,
                kSceneOccW,
                kSceneOccH);
            const bool occluded = shs::culling_sw::is_rect_occluded(
                std::span<const float>(scene_occlusion_depth_.data(), scene_occlusion_depth_.size()),
                kSceneOccW,
                kSceneOccH,
                rect,
                1e-4f);

            if (occluded)
            {
                instance_visible_mask_[idx] = 0u;
                continue;
            }

            instance_visible_mask_[idx] = 1u;
            ++visible_instances;
            shs::culling_sw::rasterize_mesh_depth_transformed(
                std::span<float>(scene_occlusion_depth_.data(), scene_occlusion_depth_.size()),
                kSceneOccW,
                kSceneOccH,
                occluder_mesh_for_mesh(instances_[idx].mesh_kind),
                instance_models_[idx],
                camera_ubo_.view_proj);
        }
        visible_instance_count_ = visible_instances;
    }

    void build_light_occlusion_depth_from_scene()
    {
        if (!enable_light_occlusion_)
        {
            return;
        }

        const size_t expected = static_cast<size_t>(kLightOccW) * static_cast<size_t>(kLightOccH);
        if (light_occlusion_depth_.size() != expected)
        {
            light_occlusion_depth_.assign(expected, 1.0f);
        }
        else
        {
            std::fill(light_occlusion_depth_.begin(), light_occlusion_depth_.end(), 1.0f);
        }

        for (size_t idx = 0; idx < instance_visible_mask_.size() && idx < instance_models_.size(); ++idx)
        {
            if (instance_visible_mask_[idx] == 0u) continue;
            shs::culling_sw::rasterize_mesh_depth_transformed(
                std::span<float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
                kLightOccW,
                kLightOccH,
                occluder_mesh_for_mesh(instances_[idx].mesh_kind),
                instance_models_[idx],
                camera_ubo_.view_proj);
        }

        if (floor_visible_)
        {
            shs::culling_sw::rasterize_mesh_depth_transformed(
                std::span<float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
                kLightOccW,
                kLightOccH,
                floor_occluder_mesh_,
                floor_model_,
                camera_ubo_.view_proj);
        }
    }

    void refresh_visible_object_bounds_for_light_prefilter()
    {
        visible_object_aabbs_.clear();
        if (light_object_cull_mode_ == shs::LightObjectCullMode::None) return;
        visible_object_aabbs_.reserve(visible_instance_count_ + (floor_visible_ ? 1u : 0u));
        for (size_t i = 0; i < instance_visible_mask_.size() && i < instance_models_.size(); ++i)
        {
            if (instance_visible_mask_[i] == 0u) continue;
            visible_object_aabbs_.push_back(shs::transform_aabb(local_aabb_for_mesh(instances_[i].mesh_kind), instance_models_[i]));
        }
        if (floor_visible_)
        {
            visible_object_aabbs_.push_back(shs::transform_aabb(floor_local_aabb_, floor_model_));
        }
    }

    bool passes_light_object_prefilter(const shs::CullingLightGPU& packed) const
    {
        if (light_object_cull_mode_ == shs::LightObjectCullMode::None) return true;
        if (visible_object_aabbs_.empty()) return false;

        if (light_object_cull_mode_ == shs::LightObjectCullMode::SphereAabb)
        {
            shs::Sphere s{};
            s.center = glm::vec3(packed.cull_sphere);
            s.radius = std::max(packed.cull_sphere.w, 0.0f);
            for (const shs::AABB& obj : visible_object_aabbs_)
            {
                if (shs::intersect_sphere_aabb(s, obj)) return true;
            }
            return false;
        }

        shs::AABB light_box{};
        light_box.minv = glm::vec3(packed.cull_aabb_min);
        light_box.maxv = glm::vec3(packed.cull_aabb_max);
        for (const shs::AABB& obj : visible_object_aabbs_)
        {
            if (shs::intersect_aabb_aabb(light_box, obj)) return true;
        }
        return false;
    }

    static shs::InputState make_runtime_input_state_from_latch(
        const shs::RuntimeInputLatch& latch,
        bool pending_quit_action)
    {
        shs::InputState input{};
        input.forward = latch.forward;
        input.backward = latch.backward;
        input.left = latch.left;
        input.right = latch.right;
        input.ascend = latch.ascend;
        input.descend = latch.descend;
        input.boost = latch.boost;
        input.look_active = latch.left_mouse_down || latch.right_mouse_down;

        float mouse_dx = latch.mouse_dx_accum;
        float mouse_dy = latch.mouse_dy_accum;
        if (std::abs(mouse_dx) > FreeCamera::kMouseSpikeThreshold || std::abs(mouse_dy) > FreeCamera::kMouseSpikeThreshold)
        {
            mouse_dx = 0.0f;
            mouse_dy = 0.0f;
        }
        mouse_dx = std::clamp(mouse_dx, -FreeCamera::kMouseDeltaClamp, FreeCamera::kMouseDeltaClamp);
        mouse_dy = std::clamp(mouse_dy, -FreeCamera::kMouseDeltaClamp, FreeCamera::kMouseDeltaClamp);

        input.look_dx = -mouse_dx;
        input.look_dy = mouse_dy;
        input.quit = pending_quit_action || latch.quit_requested;
        return input;
    }

    void update_frame_data(float dt, float t, uint32_t w, uint32_t h, uint32_t frame_slot)
    {
        if (phase_g_config_.enabled && phase_g_state_.started && !phase_g_state_.finished)
        {
            apply_phase_g_camera_tour(dt, t);
        }

        const float aspect = (h > 0) ? (static_cast<float>(w) / static_cast<float>(h)) : 1.0f;
        const shs::InputState input = make_runtime_input_state_from_latch(input_latch_, pending_quit_action_);
        pending_quit_action_ = false;
        runtime_actions_.clear();
        shs::emit_human_actions(input, runtime_actions_, camera_.move_speed, 2.0f, camera_.look_speed);
        runtime_state_ = shs::reduce_runtime_state(runtime_state_, runtime_actions_, dt);
        if (runtime_state_.quit_requested) running_ = false;
        camera_.pos = runtime_state_.camera.pos;
        camera_.yaw = runtime_state_.camera.yaw;
        camera_.pitch = runtime_state_.camera.pitch;
        input_latch_ = shs::clear_runtime_input_frame_deltas(input_latch_);

        const glm::vec3 cam_pos = camera_.pos;
        camera_ubo_.view = camera_.view_matrix();
        const glm::mat4 base_proj = shs::perspective_lh_no(glm::radians(62.0f), aspect, kDemoNearZ, kDemoFarZ);
        temporal_state_.frame_index = ctx_.frame_index;
        temporal_state_.previous_view_proj = temporal_state_.current_view_proj;
        const bool temporal_active =
            active_taa_pass_enabled() &&
            temporal_settings_.accumulation_enabled &&
            supports_swapchain_history_copy();
        temporal_state_.jitter_ndc = (temporal_settings_.jitter_enabled && temporal_active)
            ? shs::compute_taa_jitter_ndc(temporal_state_.frame_index, w, h, temporal_settings_.jitter_scale)
            : glm::vec2(0.0f);
        temporal_state_.jitter_pixels = glm::vec2(
            0.5f * temporal_state_.jitter_ndc.x * static_cast<float>(w),
            0.5f * temporal_state_.jitter_ndc.y * static_cast<float>(h));
        camera_ubo_.proj = shs::add_projection_jitter_ndc(base_proj, temporal_state_.jitter_ndc);
        camera_ubo_.view_proj = camera_ubo_.proj * camera_ubo_.view;
        temporal_state_.current_view_proj = camera_ubo_.view_proj;
        camera_ubo_.camera_pos_time = glm::vec4(cam_pos, t);
        camera_ubo_.sun_dir_intensity = glm::vec4(glm::normalize(glm::vec3(-0.35f, -1.0f, -0.18f)), 1.45f);
        camera_ubo_.screen_tile_lightcount = glm::uvec4(w, h, tile_w_, active_light_count_);
        camera_ubo_.params = glm::uvec4(tile_h_, kMaxLightsPerTile, light_tile_size_, static_cast<uint32_t>(culling_mode_));
        const uint32_t semantic_debug_mode = active_semantic_debug_mode();
        const uint32_t semantic_debug_id =
            (semantic_debug_mode_for_framebuffer_preset(framebuffer_debug_preset_) != 0u)
                ? semantic_debug_mode
                : static_cast<uint32_t>(active_semantic_debug_);
        camera_ubo_.culling_params = glm::uvec4(
            cluster_z_slices_,
            shading_variant_,
            semantic_debug_mode,
            semantic_debug_id);
        camera_ubo_.depth_params = glm::vec4(kDemoNearZ, kDemoFarZ, 0.0f, 0.0f);
        camera_ubo_.exposure_gamma = glm::vec4(tonemap_exposure_, tonemap_gamma_, 0.0f, 0.0f);
        camera_ubo_.temporal_params = glm::vec4(
            temporal_active ? 1.0f : 0.0f,
            (temporal_active && shs::vk_render_path_history_color_valid(temporal_resources_)) ? 1.0f : 0.0f,
            std::clamp(temporal_settings_.history_blend, 0.0f, 1.0f),
            0.0f);
        // Keep directional shadow optional and subtle in this stress demo
        // so local-light behavior remains readable.
        const float dir_shadow_strength =
            (shadow_settings_.enable && enable_sun_shadow_)
                ? std::clamp(sun_shadow_strength_, 0.0f, 1.0f)
                : 0.0f;
        camera_ubo_.sun_shadow_params = glm::vec4(dir_shadow_strength, 0.0012f, 0.0030f, 2.0f);
        camera_ubo_.sun_shadow_filter = glm::vec4(
            shadow_settings_.quality.pcf_step,
            (shadow_settings_.enable && enable_sun_shadow_) ? 1.0f : 0.0f,
            0.0f,
            0.0f);

        for (size_t i = 0; i < instances_.size(); ++i)
        {
            const Instance& inst = instances_[i];
            const float bob = std::sin(t * 1.15f + inst.phase) * 0.24f;
            const glm::vec3 rot = inst.base_rot + inst.rot_speed * t;
            glm::mat4 m(1.0f);
            m = glm::translate(m, inst.base_pos + glm::vec3(0.0f, bob, 0.0f));
            m = glm::rotate(m, rot.x, glm::vec3(1.0f, 0.0f, 0.0f));
            m = glm::rotate(m, rot.y, glm::vec3(0.0f, 1.0f, 0.0f));
            m = glm::rotate(m, rot.z, glm::vec3(0.0f, 0.0f, 1.0f));
            m = glm::scale(m, glm::vec3(inst.scale));
            instance_models_[i] = m;
        }

        rebuild_instance_cull_shapes();
        const shs::CullingCell camera_cell = shs::extract_frustum_cell(
            camera_ubo_.view_proj,
            shs::CullingCellKind::CameraFrustumPerspective);
        update_visibility_from_cell(camera_cell);
        apply_scene_software_occlusion();
        build_light_occlusion_depth_from_scene();
        refresh_visible_object_bounds_for_light_prefilter();

        shs::AABB shadow_scene_aabb = shadow_scene_static_bounds_ready_
            ? shadow_scene_static_aabb_
            : shs::AABB{};
        if (!shadow_scene_static_bounds_ready_)
        {
            shadow_scene_aabb.expand(glm::vec3(-1.0f));
            shadow_scene_aabb.expand(glm::vec3(1.0f));
        }

        const glm::vec3 sun_dir = glm::normalize(glm::vec3(camera_ubo_.sun_dir_intensity));
        const shs::LightCamera sun_cam = shs::build_dir_light_camera_aabb(
            sun_dir,
            shadow_scene_aabb,
            14.0f,
            kSunShadowMapSize);
        sun_shadow_view_proj_ = sun_cam.viewproj;
        camera_ubo_.sun_shadow_view_proj = sun_shadow_view_proj_;

        if (shadow_lights_gpu_.size() != kMaxLights) shadow_lights_gpu_.assign(kMaxLights, ShadowLightGPU{});
        std::fill(shadow_lights_gpu_.begin(), shadow_lights_gpu_.end(), ShadowLightGPU{});
        local_shadow_casters_.clear();

        const auto build_local_shadow_vp = [&](const glm::vec3& pos_ws, const glm::vec3& dir_ws, float fov_rad, float range) -> glm::mat4 {
            const glm::vec3 dir = shs::normalize_or(dir_ws, glm::vec3(0.0f, -1.0f, 0.0f));
            glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
            if (std::abs(glm::dot(dir, up)) > 0.95f) up = glm::vec3(0.0f, 0.0f, 1.0f);
            const glm::mat4 v = glm::lookAtLH(pos_ws, pos_ws + dir, up);
            const glm::mat4 p = glm::perspectiveLH_NO(
                std::clamp(fov_rad, glm::radians(25.0f), glm::radians(150.0f)),
                1.0f,
                kShadowNearZ,
                std::max(range, kShadowNearZ + 0.2f));
            return p * v;
        };

        uint32_t used_spot_shadow = 0;
        uint32_t used_point_shadow = 0;
        uint32_t used_rect_shadow = 0;
        uint32_t used_tube_shadow = 0;

        const auto light_in_frustum = [&](const shs::Sphere& bounds) -> bool {
            shs::Sphere s = bounds;
            if (culling_mode_ == shs::LightCullingMode::TiledDepthRange)
            {
                // Keep tiled-depth conservative enough to avoid edge popping,
                // but still frustum-bound so light distribution matches other modes.
                s.radius = std::max(s.radius * 1.20f, s.radius + 0.75f);
            }
            else
            {
                // Slightly conservative light visibility to avoid edge flicker
                // when culling animated/orbiting lights against the camera frustum.
                s.radius = std::max(s.radius * 1.08f, s.radius + 0.25f);
            }
            shs::Sphere light_bounds = s; // Copy
            light_bounds.radius = std::max(light_bounds.radius, 0.0f); // Ensure valid
            const shs::CullClass light_class = shs::classify_sphere_vs_cell(light_bounds, camera_cell);
            return shs::cull_class_is_visible(light_class, true);
        };

        const auto light_in_occlusion = [&](const shs::Sphere& bounds) -> bool {
            if (!enable_light_occlusion_) return true;
            if (light_occlusion_depth_.empty()) return true;
            const shs::AABB light_box = shs::aabb_from_sphere(bounds);
            const shs::culling_sw::ScreenRectDepth rect = shs::culling_sw::project_aabb_to_screen_rect(
                light_box,
                camera_ubo_.view_proj,
                kLightOccW,
                kLightOccH);
            if (!rect.valid) return true;
            return !shs::culling_sw::is_rect_occluded(
                std::span<const float>(light_occlusion_depth_.data(), light_occlusion_depth_.size()),
                kLightOccW,
                kLightOccH,
                rect,
                1e-4f);
        };

        light_set_.clear_local_lights();
        const uint32_t lc = std::min<uint32_t>(active_light_count_, static_cast<uint32_t>(light_anim_.size()));
        uint32_t visible_light_count = 0;
        light_volume_debug_draws_.clear();
        light_volume_debug_draws_.reserve(lc);
        light_frustum_rejected_ = 0;
        light_occlusion_rejected_ = 0;
        light_prefilter_rejected_ = 0;
        for (uint32_t i = 0; i < lc; ++i)
        {
            const LightAnim& la = light_anim_[i];
            const float a = la.angle0 + la.speed * t;
            const float orbit_r = std::max(2.0f, la.orbit_radius * light_orbit_scale_);
            const float y = (la.height + light_height_bias_) + std::sin(a * 1.7f + la.phase) * 1.2f;
            const glm::vec3 p(std::cos(a) * orbit_r, y, std::sin(a) * orbit_r);
            float shape_range = la.range;
            switch (la.type)
            {
                case shs::LightType::RectArea:
                {
                    const float hx = std::max(0.10f, la.shape_params.x);
                    const float hy = std::max(0.10f, la.shape_params.y);
                    // Keep rect-area depth comparable to panel footprint.
                    shape_range = std::max(0.90f, std::max(hx, hy) * 2.25f);
                    break;
                }
                case shs::LightType::TubeArea:
                {
                    const float half_len = std::max(0.10f, la.shape_params.x);
                    const float radius = std::max(0.05f, la.shape_params.y);
                    // Capsule influence radius should stay tied to tube dimensions.
                    shape_range = std::max(0.90f, (half_len + radius) * 2.00f);
                    break;
                }
                case shs::LightType::Spot:
                {
                    // Keep cone depth in a practical range for scene readability.
                    shape_range = std::clamp(la.range, 2.20f, 7.50f);
                    break;
                }
                case shs::LightType::Point:
                default:
                {
                    shape_range = std::clamp(la.range, 1.20f, 6.80f);
                    break;
                }
            }
            const float tuned_range = std::max(0.60f, shape_range * light_range_scale_);
            const float tuned_intensity = std::max(0.0f, la.intensity * light_intensity_scale_);

            switch (la.type)
            {
                case shs::LightType::Spot:
                {
                    shs::SpotLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.direction_ws = la.direction_ws;
                    l.inner_angle_rad = la.spot_inner_outer.x;
                    l.outer_angle_rad = la.spot_inner_outer.y;
                    const shs::Sphere light_bounds = shs::spot_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_spot_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.spot &&
                        used_spot_shadow < std::min<uint32_t>(shadow_settings_.budget.max_spot, kMaxSpotShadowMaps))
                    {
                        const uint32_t layer = used_spot_shadow++;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        ShadowLightGPU sh{};
                        sh.light_view_proj = build_local_shadow_vp(l.common.position_ws, l.direction_ws, l.outer_angle_rad * 2.0f, l.common.range);
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.72f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            camera_ubo_.sun_shadow_params.w);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::SpotMap2D),
                            layer,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::SpotMap2D;
                        caster.layer_base = layer;
                        caster.position_ws = l.common.position_ws;
                        caster.direction_ws = l.direction_ws;
                        caster.range = l.common.range;
                        caster.outer_angle_rad = l.outer_angle_rad;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.spots.push_back(l);
                    gpu_lights_[light_index] = shs::make_spot_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Cone;
                        d.model = make_spot_volume_debug_model(
                            l.common.position_ws,
                            l.direction_ws,
                            l.common.range,
                            l.outer_angle_rad);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.08f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
                case shs::LightType::RectArea:
                {
                    shs::RectAreaLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.direction_ws = la.direction_ws;
                    l.right_ws = la.rect_right_ws;
                    l.half_extents = glm::vec2(la.shape_params.x, la.shape_params.y);
                    const shs::Sphere light_bounds = shs::rect_area_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_rect_area_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.rect_area_proxy &&
                        used_spot_shadow < kMaxSpotShadowMaps &&
                        used_rect_shadow < shadow_settings_.budget.max_rect_area)
                    {
                        ++used_rect_shadow;
                        const uint32_t layer = used_spot_shadow++;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        const float proxy_fov = glm::radians(76.0f);
                        ShadowLightGPU sh{};
                        sh.light_view_proj = build_local_shadow_vp(l.common.position_ws, l.direction_ws, proxy_fov, l.common.range);
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.62f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            1.0f);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::AreaProxySpotMap2D),
                            layer,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::AreaProxySpotMap2D;
                        caster.layer_base = layer;
                        caster.position_ws = l.common.position_ws;
                        caster.direction_ws = l.direction_ws;
                        caster.range = l.common.range;
                        caster.outer_angle_rad = proxy_fov * 0.5f;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.rect_areas.push_back(l);
                    gpu_lights_[light_index] = shs::make_rect_area_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Box;
                        d.model = make_rect_volume_debug_model(
                            l.common.position_ws,
                            l.direction_ws,
                            l.right_ws,
                            l.half_extents.x,
                            l.half_extents.y,
                            l.common.range);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.06f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
                case shs::LightType::TubeArea:
                {
                    shs::TubeAreaLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    l.axis_ws = la.direction_ws;
                    l.half_length = la.shape_params.x;
                    l.radius = la.shape_params.y;
                    const shs::Sphere light_bounds = shs::tube_area_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_tube_area_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.tube_area_proxy &&
                        used_spot_shadow < kMaxSpotShadowMaps &&
                        used_tube_shadow < shadow_settings_.budget.max_tube_area)
                    {
                        ++used_tube_shadow;
                        const uint32_t layer = used_spot_shadow++;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        const glm::vec3 dir = shs::normalize_or(l.axis_ws, glm::vec3(1.0f, 0.0f, 0.0f));
                        const float proxy_fov = glm::radians(70.0f);
                        ShadowLightGPU sh{};
                        sh.light_view_proj = build_local_shadow_vp(l.common.position_ws, dir, proxy_fov, l.common.range);
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.58f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            1.0f);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::AreaProxySpotMap2D),
                            layer,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::AreaProxySpotMap2D;
                        caster.layer_base = layer;
                        caster.position_ws = l.common.position_ws;
                        caster.direction_ws = dir;
                        caster.range = l.common.range;
                        caster.outer_angle_rad = proxy_fov * 0.5f;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.tube_areas.push_back(l);
                    gpu_lights_[light_index] = shs::make_tube_area_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Box;
                        d.model = make_tube_volume_debug_model(
                            l.common.position_ws,
                            l.axis_ws,
                            l.half_length,
                            l.radius);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.05f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
                case shs::LightType::Point:
                default:
                {
                    shs::PointLight l{};
                    l.common.position_ws = p;
                    l.common.range = tuned_range;
                    l.common.color = la.color;
                    l.common.intensity = tuned_intensity;
                    l.common.attenuation_model = la.attenuation_model;
                    l.common.attenuation_power = la.attenuation_power;
                    l.common.attenuation_bias = la.attenuation_bias;
                    l.common.attenuation_cutoff = la.attenuation_cutoff;
                    l.common.flags = shs::LightFlagsDefault;
                    const shs::Sphere light_bounds = shs::point_light_culling_sphere(l);
                    if (!light_in_frustum(light_bounds))
                    {
                        ++light_frustum_rejected_;
                        break;
                    }
                    if (!light_in_occlusion(light_bounds))
                    {
                        ++light_occlusion_rejected_;
                        break;
                    }
                    const shs::CullingLightGPU packed = shs::make_point_culling_light(l);
                    if (!passes_light_object_prefilter(packed))
                    {
                        ++light_prefilter_rejected_;
                        break;
                    }
                    const uint32_t light_index = visible_light_count;
                    if (shadow_settings_.enable &&
                        shadow_settings_.point &&
                        used_point_shadow < std::min<uint32_t>(shadow_settings_.budget.max_point, kMaxPointShadowLights))
                    {
                        const uint32_t layer_base = kMaxSpotShadowMaps + (used_point_shadow * kPointShadowFaceCount);
                        ++used_point_shadow;
                        l.common.flags |= shs::LightFlagAffectsShadows;
                        ShadowLightGPU sh{};
                        sh.position_range = glm::vec4(l.common.position_ws, l.common.range);
                        sh.shadow_params = glm::vec4(
                            0.68f,
                            camera_ubo_.sun_shadow_params.y,
                            camera_ubo_.sun_shadow_params.z,
                            camera_ubo_.sun_shadow_params.w);
                        sh.meta = glm::uvec4(
                            static_cast<uint32_t>(shs::ShadowTechnique::PointCube),
                            layer_base,
                            0u,
                            1u);
                        shadow_lights_gpu_[light_index] = sh;

                        LocalShadowCaster caster{};
                        caster.light_index = light_index;
                        caster.technique = shs::ShadowTechnique::PointCube;
                        caster.layer_base = layer_base;
                        caster.position_ws = l.common.position_ws;
                        caster.range = l.common.range;
                        caster.strength = sh.shadow_params.x;
                        local_shadow_casters_.push_back(caster);
                    }
                    light_set_.points.push_back(l);
                    gpu_lights_[light_index] = shs::make_point_culling_light(l);
                    {
                        LightVolumeDebugDraw d{};
                        d.mesh = DebugVolumeMeshKind::Sphere;
                        d.model = make_point_volume_debug_model(l.common.position_ws, l.common.range);
                        const glm::vec3 c = glm::clamp(l.common.color * 1.04f, glm::vec3(0.05f), glm::vec3(1.0f));
                        d.color = glm::vec4(c, 1.0f);
                        light_volume_debug_draws_.push_back(d);
                    }
                    visible_light_count++;
                    break;
                }
            }
        }
        visible_light_count_ = visible_light_count;
        camera_ubo_.screen_tile_lightcount.w = visible_light_count_;
        if (!frame_resources_.valid_slot(frame_slot))
        {
            throw std::runtime_error("Invalid frame slot for dynamic uploads");
        }
        FrameResources& fr = frame_resources_.at_slot(frame_slot);
        std::memcpy(fr.camera_buffer.mapped, &camera_ubo_, sizeof(CameraUBO));

        if (visible_light_count_ > 0u)
        {
            std::memcpy(fr.light_buffer.mapped, gpu_lights_.data(), static_cast<size_t>(visible_light_count_) * sizeof(shs::CullingLightGPU));
        }
        std::memcpy(fr.shadow_light_buffer.mapped, shadow_lights_gpu_.data(), static_cast<size_t>(kMaxLights) * sizeof(ShadowLightGPU));

        point_count_active_ = static_cast<uint32_t>(light_set_.points.size());
        spot_count_active_ = static_cast<uint32_t>(light_set_.spots.size());
        rect_count_active_ = static_cast<uint32_t>(light_set_.rect_areas.size());
        tube_count_active_ = static_cast<uint32_t>(light_set_.tube_areas.size());
        spot_shadow_count_ = used_spot_shadow;
        point_shadow_count_ = used_point_shadow;
    }

    void begin_render_pass_depth(VkCommandBuffer cmd)
    {
        VkClearValue clear{};
        clear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = depth_target_.render_pass;
        bi.framebuffer = depth_target_.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {depth_target_.w, depth_target_.h};
        bi.clearValueCount = 1;
        bi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    }

    void begin_render_pass_scene(VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi)
    {
        VkClearValue clear[2]{};
        clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = fi.render_pass;
        bi.framebuffer = fi.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = fi.extent;
        bi.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
        bi.pClearValues = clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_SECONDARY_COMMAND_BUFFERS);
    }

    void begin_render_pass_gbuffer(VkCommandBuffer cmd)
    {
        if (gbuffer_target_.render_pass == VK_NULL_HANDLE || gbuffer_target_.framebuffer == VK_NULL_HANDLE) return;

        std::array<VkClearValue, 5> clear{};
        for (uint32_t i = 0; i < 4; ++i)
        {
            clear[i].color = {{0.0f, 0.0f, 0.0f, 0.0f}};
        }
        clear[4].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = gbuffer_target_.render_pass;
        bi.framebuffer = gbuffer_target_.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {gbuffer_target_.w, gbuffer_target_.h};
        bi.clearValueCount = static_cast<uint32_t>(clear.size());
        bi.pClearValues = clear.data();
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }

    void begin_render_pass_ssao(VkCommandBuffer cmd)
    {
        if (ao_target_.render_pass == VK_NULL_HANDLE || ao_target_.framebuffer == VK_NULL_HANDLE) return;

        VkClearValue clear{};
        clear.color = {{1.0f, 1.0f, 1.0f, 1.0f}};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = ao_target_.render_pass;
        bi.framebuffer = ao_target_.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {ao_target_.w, ao_target_.h};
        bi.clearValueCount = 1;
        bi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }

    void begin_render_pass_post(VkCommandBuffer cmd, const PostColorTarget& target)
    {
        if (target.render_pass == VK_NULL_HANDLE || target.framebuffer == VK_NULL_HANDLE) return;

        VkClearValue clear{};
        clear.color = {{0.03f, 0.035f, 0.045f, 1.0f}};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = target.render_pass;
        bi.framebuffer = target.framebuffer;
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {target.w, target.h};
        bi.clearValueCount = 1;
        bi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }

    void set_viewport_scissor(VkCommandBuffer cmd, uint32_t w, uint32_t h, bool flip_y)
    {
        shs::vk_cmd_set_viewport_scissor(cmd, w, h, flip_y);
    }

    void begin_render_pass_shadow(
        VkCommandBuffer cmd,
        const LayeredDepthTarget& target,
        uint32_t layer)
    {
        VkClearValue clear{};
        clear.depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        bi.renderPass = target.render_pass;
        bi.framebuffer = target.framebuffers[layer];
        bi.renderArea.offset = {0, 0};
        bi.renderArea.extent = {target.w, target.h};
        bi.clearValueCount = 1;
        bi.pClearValues = &clear;
        vkCmdBeginRenderPass(cmd, &bi, VK_SUBPASS_CONTENTS_INLINE);
    }

    glm::mat4 make_point_shadow_face_view_proj(const glm::vec3& light_pos, float range, uint32_t face) const
    {
        static const glm::vec3 dirs[6] = {
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(-1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, -1.0f),
        };
        static const glm::vec3 ups[6] = {
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
        };
        const uint32_t f = std::min<uint32_t>(face, 5u);
        const glm::mat4 v = glm::lookAtLH(light_pos, light_pos + dirs[f], ups[f]);
        const glm::mat4 p = glm::perspectiveLH_NO(glm::radians(90.0f), 1.0f, kShadowNearZ, std::max(range, kShadowNearZ + 0.2f));
        return p * v;
    }

    glm::mat4 make_local_shadow_view_proj(const LocalShadowCaster& caster) const
    {
        if (caster.technique == shs::ShadowTechnique::PointCube)
        {
            return glm::mat4(1.0f);
        }
        const glm::vec3 dir = shs::normalize_or(caster.direction_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        glm::vec3 up(0.0f, 1.0f, 0.0f);
        if (std::abs(glm::dot(dir, up)) > 0.95f) up = glm::vec3(0.0f, 0.0f, 1.0f);
        const glm::mat4 v = glm::lookAtLH(caster.position_ws, caster.position_ws + dir, up);
        const glm::mat4 p = glm::perspectiveLH_NO(
            std::clamp(caster.outer_angle_rad * 2.0f, glm::radians(25.0f), glm::radians(150.0f)),
            1.0f,
            kShadowNearZ,
            std::max(caster.range, kShadowNearZ + 0.2f));
        return p * v;
    }

    void draw_shadow_scene(VkCommandBuffer cmd, const glm::mat4& light_view_proj, shs::CullingCellKind cell_kind)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline_);
        const shs::CullingCell shadow_cell = shs::extract_frustum_cell(light_view_proj, cell_kind);
        if (instance_cull_shapes_.size() != instances_.size())
        {
            rebuild_instance_cull_shapes();
        }
        
        const VkDeviceSize vb_off = 0;
        const shs::AABB floor_ws = shs::transform_aabb(floor_local_aabb_, floor_model_);
        const bool floor_in_shadow_cell = shs::cull_class_is_visible(
            shs::classify_aabb_vs_cell(floor_ws, shadow_cell),
            true);

        if (floor_in_shadow_cell && !floor_indices_.empty() && floor_vertex_buffer_.buffer != VK_NULL_HANDLE)
        {
            vkCmdBindVertexBuffers(cmd, 0, 1, &floor_vertex_buffer_.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, floor_index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);
            ShadowPush pc{};
            pc.light_view_proj = light_view_proj;
            pc.model = floor_model_;
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
            vkCmdDrawIndexed(cmd, static_cast<uint32_t>(floor_indices_.size()), 1, 0, 0, 0);
        }

        const shs::CullResult shadow_cull = shs::cull_vs_cell(std::span<const shs::SceneShape>{instance_cull_shapes_}, shadow_cell);
        for (size_t idx : shadow_cull.visible_indices)
        {
            if (idx >= instance_models_.size()) continue;
            const uint32_t i = static_cast<uint32_t>(idx);
            const Instance::MeshKind mesh_kind = instances_[i].mesh_kind;
            const GpuBuffer& vb = vertex_buffer_for_mesh(mesh_kind);
            const GpuBuffer& ib = index_buffer_for_mesh(mesh_kind);
            const uint32_t index_count = index_count_for_mesh(mesh_kind);
            if (vb.buffer == VK_NULL_HANDLE || ib.buffer == VK_NULL_HANDLE || index_count == 0u) continue;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT32);
            ShadowPush pc{};
            pc.light_view_proj = light_view_proj;
            pc.model = instance_models_[i];
            vkCmdPushConstants(cmd, shadow_pipeline_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &pc);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    void record_shadow_passes(VkCommandBuffer cmd)
    {
        if (!shadow_settings_.enable) return;
        if (shadow_pipeline_ == VK_NULL_HANDLE || shadow_pipeline_layout_ == VK_NULL_HANDLE) return;
        if (sun_shadow_target_.render_pass == VK_NULL_HANDLE || sun_shadow_target_.framebuffers.empty()) return;
        if (local_shadow_target_.render_pass == VK_NULL_HANDLE || local_shadow_target_.framebuffers.empty()) return;

        begin_render_pass_shadow(cmd, sun_shadow_target_, 0u);
        set_viewport_scissor(cmd, sun_shadow_target_.w, sun_shadow_target_.h, true);
        draw_shadow_scene(cmd, sun_shadow_view_proj_, shs::CullingCellKind::CascadeFrustum);
        vkCmdEndRenderPass(cmd);

        for (const LocalShadowCaster& caster : local_shadow_casters_)
        {
            if (caster.technique == shs::ShadowTechnique::PointCube)
            {
                for (uint32_t face = 0; face < kPointShadowFaceCount; ++face)
                {
                    const uint32_t layer = caster.layer_base + face;
                    if (layer >= local_shadow_target_.framebuffers.size()) continue;
                    const glm::mat4 vp = make_point_shadow_face_view_proj(caster.position_ws, caster.range, face);
                    begin_render_pass_shadow(cmd, local_shadow_target_, layer);
                    set_viewport_scissor(cmd, local_shadow_target_.w, local_shadow_target_.h, true);
                    draw_shadow_scene(cmd, vp, shs::CullingCellKind::PointShadowFaceFrustum);
                    vkCmdEndRenderPass(cmd);
                }
            }
            else
            {
                if (caster.layer_base >= local_shadow_target_.framebuffers.size()) continue;
                const glm::mat4 vp = make_local_shadow_view_proj(caster);
                begin_render_pass_shadow(cmd, local_shadow_target_, caster.layer_base);
                set_viewport_scissor(cmd, local_shadow_target_.w, local_shadow_target_.h, true);
                draw_shadow_scene(cmd, vp, shs::CullingCellKind::SpotShadowFrustum);
                vkCmdEndRenderPass(cmd);
            }
        }
    }

    glm::mat4 make_point_volume_debug_model(const glm::vec3& pos_ws, float range) const
    {
        const float r = std::max(range, 0.10f);
        // Source sphere mesh radius is 0.5, so multiply by 2*r for target radius r.
        return glm::translate(glm::mat4(1.0f), pos_ws) * glm::scale(glm::mat4(1.0f), glm::vec3(r * 2.0f));
    }

    glm::mat4 make_spot_volume_debug_model(
        const glm::vec3& pos_ws,
        const glm::vec3& dir_ws,
        float range,
        float outer_angle_rad) const
    {
        const glm::vec3 dir = shs::normalize_or(dir_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        const float h = std::max(range, 0.25f);
        const float base_radius =
            std::tan(std::max(outer_angle_rad, glm::radians(3.0f))) * h;

        glm::vec3 bx{};
        glm::vec3 by{};
        glm::vec3 bz{};
        // Cone mesh tip is at +Y, so align +Y to -dir and offset center so tip sits at light position.
        basis_from_axis(-dir, bx, by, bz);
        const glm::vec3 center = pos_ws + dir * (h * 0.5f);
        return model_from_basis_and_scale(center, bx, by, bz, glm::vec3(base_radius, h, base_radius));
    }

    glm::mat4 make_rect_volume_debug_model(
        const glm::vec3& pos_ws,
        const glm::vec3& dir_ws,
        const glm::vec3& right_ws,
        float half_x,
        float half_y,
        float extent_z) const
    {
        glm::vec3 fwd = shs::normalize_or(dir_ws, glm::vec3(0.0f, -1.0f, 0.0f));
        glm::vec3 right = right_ws - fwd * glm::dot(right_ws, fwd);
        right = shs::normalize_or(right, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 up = shs::normalize_or(glm::cross(fwd, right), glm::vec3(0.0f, 1.0f, 0.0f));
        right = shs::normalize_or(glm::cross(up, fwd), right);

        // RectArea bounds are BoxShape with half-extents (hx + r, hy + r, r)
        // Source box mesh is centered and unit-sized, so scale by 2x half-extents.
        const float ex = std::max((half_x + extent_z) * 2.0f, 0.10f);
        const float ey = std::max((half_y + extent_z) * 2.0f, 0.10f);
        const float ez = std::max(extent_z * 2.0f, 0.10f);
        return model_from_basis_and_scale(pos_ws, right, up, fwd, glm::vec3(ex, ey, ez));
    }

    glm::mat4 make_tube_volume_debug_model(
        const glm::vec3& pos_ws,
        const glm::vec3& axis_ws,
        float half_length,
        float radius) const
    {
        glm::vec3 axis = shs::normalize_or(axis_ws, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 up_hint = safe_perp_axis(axis);
        glm::vec3 up = shs::normalize_or(glm::cross(axis, up_hint), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::vec3 side = shs::normalize_or(glm::cross(up, axis), glm::vec3(0.0f, 0.0f, 1.0f));

        // TubeArea bounds is a CapsuleShape, length = 2*half_length + 2*radius, width = 2*radius.
        // We debug draw it using a Box that encapsulates the capsule bounds exactly.
        const float ex = std::max((half_length + radius) * 2.0f, 0.10f);
        const float ey = std::max(radius * 2.0f, 0.10f);
        const float ez = std::max(radius * 2.0f, 0.10f);
        return model_from_basis_and_scale(pos_ws, axis, up, side, glm::vec3(ex, ey, ez));
    }

    void draw_light_volumes_debug(VkCommandBuffer cmd, VkPipelineLayout layout, uint32_t frame_slot)
    {
        if (!show_light_volumes_debug_) return;
        if (light_volume_debug_draws_.empty()) return;
        if (!frame_resources_.valid_slot(frame_slot)) return;
        if (scene_wire_pipeline_ == VK_NULL_HANDLE) return;

        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) return;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, scene_wire_pipeline_);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);

        const uint32_t draw_count = std::min<uint32_t>(static_cast<uint32_t>(light_volume_debug_draws_.size()), 512u);
        for (uint32_t i = 0; i < draw_count; ++i)
        {
            const LightVolumeDebugDraw& d = light_volume_debug_draws_[i];
            const GpuBuffer* vb = nullptr;
            const GpuBuffer* ib = nullptr;
            uint32_t index_count = 0u;

            switch (d.mesh)
            {
                case DebugVolumeMeshKind::Sphere:
                    vb = &vertex_buffer_;
                    ib = &sphere_line_index_buffer_;
                    index_count = static_cast<uint32_t>(sphere_line_indices_.size());
                    break;
                case DebugVolumeMeshKind::Cone:
                    vb = &cone_vertex_buffer_;
                    ib = &cone_line_index_buffer_;
                    index_count = static_cast<uint32_t>(cone_line_indices_.size());
                    break;
                case DebugVolumeMeshKind::Box:
                    vb = &box_vertex_buffer_;
                    ib = &box_line_index_buffer_;
                    index_count = static_cast<uint32_t>(box_line_indices_.size());
                    break;
            }

            if (!vb || !ib) continue;
            if (vb->buffer == VK_NULL_HANDLE || ib->buffer == VK_NULL_HANDLE || index_count == 0u) continue;

            const VkDeviceSize vb_off = 0;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb->buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib->buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush pc{};
            pc.model = d.model;
            pc.base_color = d.color;
            // Unlit, colored wireframe overlay.
            pc.material_params = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
            vkCmdPushConstants(
                cmd,
                layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                sizeof(DrawPush),
                &pc);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    void draw_floor(VkCommandBuffer cmd, VkPipelineLayout layout)
    {
        if (!floor_visible_) return;

        const VkDeviceSize vb_off = 0;
        vkCmdBindVertexBuffers(cmd, 0, 1, &floor_vertex_buffer_.buffer, &vb_off);
        vkCmdBindIndexBuffer(cmd, floor_index_buffer_.buffer, 0, VK_INDEX_TYPE_UINT32);

        DrawPush floor_pc{};
        floor_pc.model = floor_model_;
        floor_pc.base_color = floor_material_color_;
        floor_pc.material_params = floor_material_params_;
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &floor_pc);
        vkCmdDrawIndexed(cmd, static_cast<uint32_t>(floor_indices_.size()), 1, 0, 0, 0);
    }

    void draw_sphere_range(VkCommandBuffer cmd, VkPipelineLayout layout, uint32_t start, uint32_t end)
    {
        const VkDeviceSize vb_off = 0;
        for (uint32_t i = start; i < end; ++i)
        {
            if (i >= instance_visible_mask_.size() || instance_visible_mask_[i] == 0u) continue;
            const Instance::MeshKind mesh_kind = instances_[i].mesh_kind;
            const GpuBuffer& vb = vertex_buffer_for_mesh(mesh_kind);
            const GpuBuffer& ib = index_buffer_for_mesh(mesh_kind);
            const uint32_t index_count = index_count_for_mesh(mesh_kind);
            if (vb.buffer == VK_NULL_HANDLE || ib.buffer == VK_NULL_HANDLE || index_count == 0u) continue;
            vkCmdBindVertexBuffers(cmd, 0, 1, &vb.buffer, &vb_off);
            vkCmdBindIndexBuffer(cmd, ib.buffer, 0, VK_INDEX_TYPE_UINT32);

            DrawPush pc{};
            pc.model = instance_models_[i];
            pc.base_color = instances_[i].base_color;
            pc.material_params = glm::vec4(
                instances_[i].metallic,
                instances_[i].roughness,
                instances_[i].ao,
                0.0f);
            vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DrawPush), &pc);
            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    bool record_secondary_batch(
        VkRenderPass rp,
        VkFramebuffer fb,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        uint32_t w,
        uint32_t h,
        bool flip_y,
        uint32_t frame_slot,
        uint32_t start,
        uint32_t end,
        bool draw_floor_here,
        VkCommandBuffer& out)
    {
        out = VK_NULL_HANDLE;
        if (start >= end && !draw_floor_here) return true;
        if (!frame_resources_.valid_slot(frame_slot)) return false;
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) return false;
        
        out = vk_->get_secondary_command_buffer(frame_slot);
        if (out == VK_NULL_HANDLE) return false;

        VkCommandBufferInheritanceInfo inh{};
        inh.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO;
        inh.renderPass = rp;
        inh.subpass = 0;
        inh.framebuffer = fb;

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT | VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT;
        bi.pInheritanceInfo = &inh;
        if (vkBeginCommandBuffer(out, &bi) != VK_SUCCESS) return false;

        set_viewport_scissor(out, w, h, flip_y);
        vkCmdBindPipeline(out, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(out, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);
        if (draw_floor_here) draw_floor(out, layout);
        if (start < end) draw_sphere_range(out, layout, start, end);

        return vkEndCommandBuffer(out) == VK_SUCCESS;
    }

    bool record_secondary_lists(
        VkRenderPass rp,
        VkFramebuffer fb,
        VkPipeline pipeline,
        VkPipelineLayout layout,
        uint32_t w,
        uint32_t h,
        bool flip_y,
        bool include_floor,
        uint32_t frame_slot,
        std::vector<VkCommandBuffer>& out)
    {
        out.clear();

        if (!use_multithread_recording_ || !jobs_ || instances_.empty())
        {
            return true;
        }

        // We can spawn as many workers as makes sense, limit to e.g. 16 or instances.
        const uint32_t workers = std::min<uint32_t>(16u, static_cast<uint32_t>(instances_.size()));
        if (workers <= 1) return true;
        if (frame_slot >= shs::VulkanRenderBackend::kMaxFramesInFlight) return false;

        std::vector<VkCommandBuffer> tmp(workers, VK_NULL_HANDLE);
        std::atomic<bool> ok{true};
        shs::WaitGroup wg{};

        const uint32_t n = static_cast<uint32_t>(instances_.size());
        const uint32_t batch = (n + workers - 1) / workers;

        for (uint32_t wi = 0; wi < workers; ++wi)
        {
            const uint32_t start = wi * batch;
            const uint32_t end = std::min(n, start + batch);
            if (start >= end) continue;

            wg.add(1);
            jobs_->enqueue([&, wi, start, end]() {
                const bool draw_floor_here = include_floor && (wi == 0);
                if (!record_secondary_batch(rp, fb, pipeline, layout, w, h, flip_y, frame_slot, start, end, draw_floor_here, tmp[wi]))
                {
                    ok.store(false, std::memory_order_release);
                }
                wg.done();
            });
        }

        wg.wait();
        if (!ok.load(std::memory_order_acquire)) return false;

        for (VkCommandBuffer cb : tmp)
        {
            if (cb != VK_NULL_HANDLE) out.push_back(cb);
        }
        return true;
    }

    bool reset_worker_pools_for_frame(uint32_t frame_slot)
    {
        (void)frame_slot;
        // worker command pools are now centrally managed and automatically reset by the Vulkan backend!
        return true;
    }

    void record_inline_scene(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, uint32_t w, uint32_t h, uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) throw std::runtime_error("Invalid frame slot for scene recording");
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) throw std::runtime_error("Scene descriptor set unavailable");
        set_viewport_scissor(cmd, w, h, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);
        draw_floor(cmd, layout);
        draw_sphere_range(cmd, layout, 0, static_cast<uint32_t>(instances_.size()));
    }

    void record_inline_depth(VkCommandBuffer cmd, VkPipeline pipeline, VkPipelineLayout layout, uint32_t w, uint32_t h, uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) throw std::runtime_error("Invalid frame slot for depth recording");
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE) throw std::runtime_error("Depth descriptor set unavailable");
        set_viewport_scissor(cmd, w, h, true);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &global_set, 0, nullptr);
        draw_floor(cmd, layout);
        draw_sphere_range(cmd, layout, 0, static_cast<uint32_t>(instances_.size()));
    }

    VkPipelineStageFlags2 stage_flags_to_stage2(VkPipelineStageFlags stages) const
    {
        (void)stages;
        VkPipelineStageFlags2 out = 0;
#if defined(VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT)
        if ((stages & VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT) != 0) out |= VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT)
        if ((stages & VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT) != 0) out |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT)
        if ((stages & VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT) != 0) out |= VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)
        if ((stages & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) != 0) out |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)
        if ((stages & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0) out |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
#endif
#if defined(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        if (out == 0) out = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
#endif
        return out;
    }

    VkAccessFlags2 access_flags_to_access2(VkAccessFlags access) const
    {
        (void)access;
        VkAccessFlags2 out = 0;
#if defined(VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        if ((access & VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT) != 0) out |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
#endif
#if defined(VK_ACCESS_2_SHADER_READ_BIT)
        if ((access & VK_ACCESS_SHADER_READ_BIT) != 0) out |= VK_ACCESS_2_SHADER_READ_BIT;
#endif
#if defined(VK_ACCESS_2_SHADER_WRITE_BIT)
        if ((access & VK_ACCESS_SHADER_WRITE_BIT) != 0) out |= VK_ACCESS_2_SHADER_WRITE_BIT;
#endif
        return out;
    }

    void cmd_memory_barrier(
        VkCommandBuffer cmd,
        VkPipelineStageFlags src_stage,
        VkAccessFlags src_access,
        VkPipelineStageFlags dst_stage,
        VkAccessFlags dst_access)
    {
        if (cmd == VK_NULL_HANDLE) return;

#if defined(VK_STRUCTURE_TYPE_DEPENDENCY_INFO) && defined(VK_STRUCTURE_TYPE_MEMORY_BARRIER_2)
        if (vk_ && vk_->supports_synchronization2())
        {
            VkMemoryBarrier2 b2{};
            b2.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            b2.srcStageMask = stage_flags_to_stage2(src_stage);
            b2.srcAccessMask = access_flags_to_access2(src_access);
            b2.dstStageMask = stage_flags_to_stage2(dst_stage);
            b2.dstAccessMask = access_flags_to_access2(dst_access);

            VkDependencyInfo dep{};
            dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &b2;
            if (vk_->cmd_pipeline_barrier2(cmd, dep)) return;
        }
#endif

        VkMemoryBarrier b{};
        b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        b.srcAccessMask = src_access;
        b.dstAccessMask = dst_access;
        vkCmdPipelineBarrier(
            cmd,
            src_stage,
            dst_stage,
            0,
            1,
            &b,
            0,
            nullptr,
            0,
            nullptr);
    }

    shs::PassId resolve_compiled_pass_kind(const shs::RenderPathCompiledPass& pass) const
    {
        return shs::pass_id_is_standard(pass.pass_id) ? pass.pass_id : shs::parse_pass_id(pass.id);
    }

    bool emit_graph_barrier_from_edge(VkCommandBuffer cmd, const shs::RenderPathBarrierEdge& edge)
    {
        if (cmd == VK_NULL_HANDLE) return false;
        if (!edge.requires_memory_barrier) return false;
        const shs::VkRenderPathBarrierTemplate barrier =
            shs::vk_make_render_path_barrier_template(edge);
        if (!barrier.valid) return false;
        cmd_memory_barrier(
            cmd,
            barrier.src_stage,
            barrier.src_access,
            barrier.dst_stage,
            barrier.dst_access);
        ++frame_graph_barrier_edges_emitted_;
        return true;
    }

    bool emit_graph_barriers_for_semantics(
        VkCommandBuffer cmd,
        shs::PassId from_pass_kind,
        std::initializer_list<shs::PassSemantic> semantics,
        shs::PassId to_pass_kind = shs::PassId::Unknown)
    {
        const shs::RenderPathBarrierPlan& plan = render_path_executor_.active_barrier_plan();
        bool emitted_any = false;
        for (const shs::PassSemantic semantic : semantics)
        {
            const shs::RenderPathBarrierEdge* edge = shs::find_render_path_barrier_edge(
                plan,
                semantic,
                from_pass_kind,
                to_pass_kind);
            if (!edge) continue;
            if (emit_graph_barrier_from_edge(cmd, *edge))
            {
                emitted_any = true;
            }
        }
        return emitted_any;
    }

    bool emit_graph_barrier_depth_to_light_culling(VkCommandBuffer cmd)
    {
        const shs::RenderPathBarrierPlan& plan = render_path_executor_.active_barrier_plan();
        const shs::RenderPathBarrierEdge* edge = shs::find_render_path_barrier_edge(
            plan,
            shs::PassSemantic::Depth,
            shs::PassId::Unknown,
            shs::PassId::LightCulling);
        if (!edge)
        {
            ++frame_graph_barrier_fallback_count_;
            return false;
        }
        if (!emit_graph_barrier_from_edge(cmd, *edge))
        {
            ++frame_graph_barrier_fallback_count_;
            return false;
        }
        return true;
    }

    bool emit_graph_barrier_gbuffer_to_consumers(VkCommandBuffer cmd)
    {
        if (emit_graph_barriers_for_semantics(
                cmd,
                shs::PassId::GBuffer,
                {
                    shs::PassSemantic::Depth,
                    shs::PassSemantic::Albedo,
                    shs::PassSemantic::Normal,
                    shs::PassSemantic::Material}))
        {
            return true;
        }
        ++frame_graph_barrier_fallback_count_;
        return false;
    }

    bool emit_graph_barrier_ssao_to_consumer(VkCommandBuffer cmd)
    {
        if (emit_graph_barriers_for_semantics(
                cmd,
                shs::PassId::SSAO,
                {shs::PassSemantic::AmbientOcclusion}))
        {
            return true;
        }
        ++frame_graph_barrier_fallback_count_;
        return false;
    }

    bool emit_graph_barrier_deferred_to_consumer(VkCommandBuffer cmd, shs::PassId deferred_pass_kind)
    {
        if (!shs::pass_id_is_standard(deferred_pass_kind))
        {
            ++frame_graph_barrier_fallback_count_;
            return false;
        }
        if (emit_graph_barriers_for_semantics(
                cmd,
                deferred_pass_kind,
                {
                    shs::PassSemantic::ColorHDR,
                    shs::PassSemantic::MotionVectors}))
        {
            return true;
        }
        ++frame_graph_barrier_fallback_count_;
        return false;
    }

    bool emit_graph_barrier_motion_blur_to_consumer(VkCommandBuffer cmd)
    {
        if (emit_graph_barriers_for_semantics(
                cmd,
                shs::PassId::MotionBlur,
                {shs::PassSemantic::ColorLDR},
                shs::PassId::DepthOfField))
        {
            return true;
        }
        ++frame_graph_barrier_fallback_count_;
        return false;
    }

    bool emit_graph_barrier_light_culling_to_consumer(VkCommandBuffer cmd)
    {
        const shs::RenderPathBarrierPlan& plan = render_path_executor_.active_barrier_plan();
        const shs::RenderPathBarrierEdge* edge = shs::find_render_path_barrier_edge(
            plan,
            shs::PassSemantic::LightGrid,
            shs::PassId::LightCulling,
            shs::PassId::Unknown);
        if (!edge)
        {
            edge = shs::find_render_path_barrier_edge(
                plan,
                shs::PassSemantic::LightIndexList,
                shs::PassId::LightCulling,
                shs::PassId::Unknown);
        }
        if (!edge)
        {
            ++frame_graph_barrier_fallback_count_;
            return false;
        }
        if (!emit_graph_barrier_from_edge(cmd, *edge))
        {
            ++frame_graph_barrier_fallback_count_;
            return false;
        }
        return true;
    }

    bool supports_swapchain_history_copy() const
    {
        if (!vk_) return false;
        return shs::vk_render_path_supports_swapchain_history_copy(vk_->swapchain_usage_flags());
    }

    bool prepare_post_source_from_scene_color(
        shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>& ctx)
    {
        if (!ctx.fi) return false;
        if (ctx.post_color_valid) return true;
        if (!ctx.scene_pass_executed) return false;
        if (post_target_a_.image == VK_NULL_HANDLE || post_target_a_.view == VK_NULL_HANDLE) return false;
        if (!supports_swapchain_history_copy())
        {
            if (!post_color_copy_support_warning_emitted_)
            {
                std::fprintf(
                    stderr,
                    "[render-path][post] Disabled scene-color copy: swapchain image does not support TRANSFER_SRC usage.\n");
                post_color_copy_support_warning_emitted_ = true;
            }
            return false;
        }

        const VkImage swapchain_image = vk_->swapchain_image(ctx.fi->image_index);
        if (swapchain_image == VK_NULL_HANDLE) return false;

        const uint32_t copy_w = std::min(ctx.fi->extent.width, post_target_a_.w);
        const uint32_t copy_h = std::min(ctx.fi->extent.height, post_target_a_.h);
        if (copy_w == 0 || copy_h == 0) return false;

        VkAccessFlags post_src_access = 0u;
        VkPipelineStageFlags post_src_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        if (post_target_a_layout_ == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
        {
            post_src_access = VK_ACCESS_SHADER_READ_BIT;
            post_src_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (post_target_a_layout_ == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
        {
            post_src_access = VK_ACCESS_TRANSFER_WRITE_BIT;
            post_src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }

        if (!shs::vk_render_path_record_swapchain_copy_to_shader_read_image(
                *vk_,
                ctx.fi->cmd,
                swapchain_image,
                ctx.fi->extent,
                post_target_a_.image,
                VkExtent2D{copy_w, copy_h},
                post_target_a_layout_,
                post_src_access,
                post_src_stage))
        {
            return false;
        }
        post_target_a_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        ctx.post_color_valid = true;
        ctx.post_color_source = 1u;
        return true;
    }

    void ensure_history_color_shader_read_layout(VkCommandBuffer cmd)
    {
        shs::vk_render_path_ensure_history_color_shader_read_layout(*vk_, cmd, temporal_resources_);
    }

    void record_history_color_copy(VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi)
    {
        if (!active_taa_pass_enabled()) return;
        if (!temporal_settings_.accumulation_enabled) return;
        if (!supports_swapchain_history_copy())
        {
            if (!temporal_resources_.history_copy_support_warning_emitted)
            {
                std::fprintf(
                    stderr,
                    "[render-path][temporal] Disabled: swapchain image does not support TRANSFER_SRC usage.\n");
                temporal_resources_.history_copy_support_warning_emitted = true;
            }
            return;
        }
        if (shs::vk_render_path_history_color_view(temporal_resources_) == VK_NULL_HANDLE) return;
        const VkImage swapchain_image = vk_->swapchain_image(fi.image_index);
        if (swapchain_image == VK_NULL_HANDLE) return;
        (void)shs::vk_render_path_record_history_color_copy(
            *vk_,
            cmd,
            swapchain_image,
            fi.extent,
            temporal_resources_);
    }

    bool ensure_phase_f_snapshot_readback_buffer(uint32_t w, uint32_t h, VkFormat format)
    {
        if (!phase_f_config_.enabled) return false;
        if (w == 0u || h == 0u) return false;
        if (!phase_f_swapchain_snapshot_supported_format(format))
        {
            return false;
        }

        const VkDeviceSize desired_bytes = static_cast<VkDeviceSize>(w) * static_cast<VkDeviceSize>(h) * 4u;
        if (phase_f_snapshot_readback_buffer_.buffer != VK_NULL_HANDLE &&
            phase_f_snapshot_readback_buffer_.size == desired_bytes &&
            phase_f_snapshot_readback_w_ == w &&
            phase_f_snapshot_readback_h_ == h &&
            phase_f_snapshot_readback_format_ == format)
        {
            return true;
        }

        const VkMemoryPropertyFlags host_flags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        create_buffer(
            desired_bytes,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            host_flags,
            phase_f_snapshot_readback_buffer_,
            true);
        phase_f_snapshot_readback_w_ = w;
        phase_f_snapshot_readback_h_ = h;
        phase_f_snapshot_readback_format_ = format;
        return phase_f_snapshot_readback_buffer_.mapped != nullptr;
    }

    bool record_phase_f_snapshot_copy(VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi)
    {
        if (!phase_f_snapshot_request_armed_) return false;
        if (phase_f_snapshot_copy_submitted_) return true;
        if (!phase_f_config_.enabled) return false;
        if (!supports_swapchain_history_copy())
        {
            phase_f_snapshot_failed_ = true;
            phase_f_snapshot_request_armed_ = false;
            std::fprintf(stderr, "[phase-f] Snapshot skipped: swapchain transfer-src unsupported.\n");
            phase_f_write_json_line(
                "{\"event\":\"snapshot_result\",\"ok\":false,\"entry\":" +
                std::to_string(phase_f_active_entry_slot_ + 1u) +
                ",\"path\":\"" + phase_f_snapshot_path_ + "\",\"reason\":\"swapchain_transfer_src_unsupported\"}");
            return false;
        }

        const VkFormat swapchain_format = vk_->swapchain_format();
        if (!ensure_phase_f_snapshot_readback_buffer(fi.extent.width, fi.extent.height, swapchain_format))
        {
            phase_f_snapshot_failed_ = true;
            phase_f_snapshot_request_armed_ = false;
            std::fprintf(stderr, "[phase-f] Snapshot skipped: unsupported format/readback buffer setup failed.\n");
            phase_f_write_json_line(
                "{\"event\":\"snapshot_result\",\"ok\":false,\"entry\":" +
                std::to_string(phase_f_active_entry_slot_ + 1u) +
                ",\"path\":\"" + phase_f_snapshot_path_ + "\",\"reason\":\"readback_buffer_setup_failed\"}");
            return false;
        }

        const VkImage swapchain_image = vk_->swapchain_image(fi.image_index);
        if (swapchain_image == VK_NULL_HANDLE)
        {
            phase_f_snapshot_failed_ = true;
            phase_f_snapshot_request_armed_ = false;
            return false;
        }

        if (!shs::vk_render_path_record_swapchain_copy_to_host_buffer(
                *vk_,
                cmd,
                swapchain_image,
                fi.extent,
                phase_f_snapshot_readback_buffer_.buffer))
        {
            phase_f_snapshot_failed_ = true;
            phase_f_snapshot_request_armed_ = false;
            phase_f_write_json_line(
                "{\"event\":\"snapshot_result\",\"ok\":false,\"entry\":" +
                std::to_string(phase_f_active_entry_slot_ + 1u) +
                ",\"path\":\"" + phase_f_snapshot_path_ + "\",\"reason\":\"copy_failed\"}");
            return false;
        }

        phase_f_snapshot_copy_submitted_ = true;
        return true;
    }

    bool write_phase_f_snapshot_from_readback()
    {
        if (!phase_f_snapshot_copy_submitted_) return false;
        if (!phase_f_snapshot_request_armed_) return false;
        if (phase_f_snapshot_path_.empty()) return false;
        if (phase_f_snapshot_readback_buffer_.mapped == nullptr) return false;
        if (phase_f_snapshot_readback_w_ == 0u || phase_f_snapshot_readback_h_ == 0u) return false;

        std::ofstream out(phase_f_snapshot_path_, std::ios::binary);
        if (!out) return false;
        out << "P6\n" << phase_f_snapshot_readback_w_ << " " << phase_f_snapshot_readback_h_ << "\n255\n";

        const uint8_t* src = reinterpret_cast<const uint8_t*>(phase_f_snapshot_readback_buffer_.mapped);
        const size_t row_stride = static_cast<size_t>(phase_f_snapshot_readback_w_) * 4u;
        for (uint32_t y = 0; y < phase_f_snapshot_readback_h_; ++y)
        {
            const uint8_t* row = src + static_cast<size_t>(y) * row_stride;
            for (uint32_t x = 0; x < phase_f_snapshot_readback_w_; ++x)
            {
                const uint8_t* px = row + static_cast<size_t>(x) * 4u;
                char rgb[3]{};
                switch (phase_f_snapshot_readback_format_)
                {
                    case VK_FORMAT_B8G8R8A8_UNORM:
                    case VK_FORMAT_B8G8R8A8_SRGB:
                        rgb[0] = static_cast<char>(px[2]);
                        rgb[1] = static_cast<char>(px[1]);
                        rgb[2] = static_cast<char>(px[0]);
                        break;
                    case VK_FORMAT_R8G8B8A8_UNORM:
                    case VK_FORMAT_R8G8B8A8_SRGB:
                    default:
                        rgb[0] = static_cast<char>(px[0]);
                        rgb[1] = static_cast<char>(px[1]);
                        rgb[2] = static_cast<char>(px[2]);
                        break;
                }
                out.write(rgb, 3);
            }
        }
        return out.good();
    }

    bool gpu_light_culler_enabled() const
    {
        return
            enable_light_culling_ &&
            vulkan_culler_backend_ == VulkanCullerBackend::GpuCompute &&
            compute_pipeline_layout_ != VK_NULL_HANDLE &&
            compute_pipeline_ != VK_NULL_HANDLE &&
            (culling_mode_ == shs::LightCullingMode::Tiled ||
             culling_mode_ == shs::LightCullingMode::TiledDepthRange ||
             culling_mode_ == shs::LightCullingMode::Clustered);
    }

    void clear_light_grid_cpu_buffers(uint32_t frame_slot)
    {
        if (!frame_resources_.valid_slot(frame_slot)) return;
        FrameResources& fr = frame_resources_.at_slot(frame_slot);
        if (fr.tile_counts_buffer.mapped && fr.tile_counts_buffer.size > 0)
        {
            std::memset(fr.tile_counts_buffer.mapped, 0, static_cast<size_t>(fr.tile_counts_buffer.size));
        }
        if (fr.tile_indices_buffer.mapped && fr.tile_indices_buffer.size > 0)
        {
            std::memset(fr.tile_indices_buffer.mapped, 0, static_cast<size_t>(fr.tile_indices_buffer.size));
        }
    }

    using FramePassExecutionContext =
        shs::VkRenderPathPassExecutionContext<shs::VulkanRenderBackend::FrameInfo>;

    shs::RenderPathExecutionPlan make_active_frame_execution_plan() const
    {
        const shs::RenderPathExecutionPlan& active_plan = render_path_executor_.active_plan();
        if (render_path_executor_.active_plan_valid() && !active_plan.pass_chain.empty())
        {
            return active_plan;
        }

        shs::RenderPathExecutionPlan fallback{};
        fallback.recipe_name = std::string("fallback_") + shs::technique_mode_name(active_technique_);
        fallback.backend = shs::RenderBackendType::Vulkan;
        fallback.technique_mode = active_technique_;
        fallback.valid = true;

        const shs::TechniqueProfile profile = shs::make_default_technique_profile(active_technique_);
        fallback.pass_chain.reserve(profile.passes.size());
        for (const auto& p : profile.passes)
        {
            fallback.pass_chain.push_back(shs::RenderPathCompiledPass{p.id, p.pass_id, p.required});
        }
        return fallback;
    }

    void draw_scene_clear_only(VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi, uint32_t frame_slot)
    {
        VkClearValue clear[2]{};
        clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = fi.render_pass;
        rp.framebuffer = fi.framebuffer;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = fi.extent;
        rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
        rp.pClearValues = clear;
        vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
        draw_light_volumes_debug(cmd, scene_pipeline_layout_, frame_slot);
        vkCmdEndRenderPass(cmd);
    }

    bool execute_pass_shadow_map(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        return shs::vk_execute_shadow_map_pass(
            ctx,
            pass,
            [this](VkCommandBuffer cmd) {
                record_shadow_passes(cmd);
            },
            [this](
                VkCommandBuffer cmd,
                VkPipelineStageFlags src_stage,
                VkAccessFlags src_access,
                VkPipelineStageFlags dst_stage,
                VkAccessFlags dst_access) {
                cmd_memory_barrier(cmd, src_stage, src_access, dst_stage, dst_access);
            });
    }

    bool execute_pass_depth_prepass(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        return shs::vk_execute_depth_prepass_pass(
            ctx,
            pass,
            depth_target_.render_pass,
            depth_target_.framebuffer,
            depth_target_.w,
            depth_target_.h,
            [this](VkCommandBuffer cmd) {
                begin_render_pass_depth(cmd);
            },
            [this](VkCommandBuffer cmd, uint32_t frame_slot) {
                record_inline_depth(
                    cmd,
                    depth_pipeline_,
                    depth_pipeline_layout_,
                    depth_target_.w,
                    depth_target_.h,
                    frame_slot);
            });
    }

    bool execute_pass_light_culling(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        const bool use_depth_range_reduction = (culling_mode_ == shs::LightCullingMode::TiledDepthRange);
        const uint32_t dispatch_z = (culling_mode_ == shs::LightCullingMode::Clustered) ? cluster_z_slices_ : 1u;

        return shs::vk_execute_light_culling_pass(
            ctx,
            pass,
            use_depth_range_reduction,
            dispatch_z,
            [this](uint32_t frame_slot) {
                clear_light_grid_cpu_buffers(frame_slot);
            },
            [this](
                VkCommandBuffer cmd,
                VkPipelineStageFlags src_stage,
                VkAccessFlags src_access,
                VkPipelineStageFlags dst_stage,
                VkAccessFlags dst_access) {
                const bool depth_to_compute =
                    (dst_stage & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0 &&
                    (src_stage & (VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)) != 0;
                const bool compute_to_fragment =
                    (src_stage & VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT) != 0 &&
                    (dst_stage & VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT) != 0;

                if (depth_to_compute)
                {
                    if (emit_graph_barrier_depth_to_light_culling(cmd)) return;
                }
                if (compute_to_fragment)
                {
                    if (emit_graph_barrier_light_culling_to_consumer(cmd)) return;
                }
                cmd_memory_barrier(cmd, src_stage, src_access, dst_stage, dst_access);
            },
            [this](VkCommandBuffer cmd, VkDescriptorSet global_set) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, depth_reduce_pipeline_);
                vkCmdBindDescriptorSets(
                    cmd,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    compute_pipeline_layout_,
                    0,
                    1,
                    &global_set,
                    0,
                    nullptr);
                vkCmdDispatch(cmd, tile_w_, tile_h_, 1);
            },
            [this](VkCommandBuffer cmd, VkDescriptorSet global_set, uint32_t z) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compute_pipeline_);
                vkCmdBindDescriptorSets(
                    cmd,
                    VK_PIPELINE_BIND_POINT_COMPUTE,
                    compute_pipeline_layout_,
                    0,
                    1,
                    &global_set,
                    0,
                    nullptr);
                vkCmdDispatch(cmd, tile_w_, tile_h_, z);
            });
    }

    bool execute_pass_scene(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        return shs::vk_execute_scene_pass(
            ctx,
            pass,
            [this]() {
                return vk_->has_depth_attachment();
            },
            [this](VkCommandBuffer cmd, const shs::VulkanRenderBackend::FrameInfo& fi) {
                begin_render_pass_scene(cmd, fi);
            },
            [this](VkCommandBuffer cmd, uint32_t frame_slot, uint32_t w, uint32_t h) {
                record_inline_scene(
                    cmd,
                    scene_pipeline_,
                    scene_pipeline_layout_,
                    w,
                    h,
                    frame_slot);
            },
            [this](VkCommandBuffer cmd, uint32_t frame_slot) {
                draw_light_volumes_debug(cmd, scene_pipeline_layout_, frame_slot);
            });
    }

    bool execute_pass_gbuffer(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        const bool gbuffer_ready =
            gbuffer_target_.render_pass != VK_NULL_HANDLE &&
            gbuffer_target_.framebuffer != VK_NULL_HANDLE &&
            gbuffer_pipeline_ != VK_NULL_HANDLE &&
            gbuffer_pipeline_layout_ != VK_NULL_HANDLE;

        return shs::vk_execute_gbuffer_pass(
            ctx,
            pass,
            gbuffer_ready,
            [this](VkCommandBuffer cmd) {
                begin_render_pass_gbuffer(cmd);
            },
            [this](VkCommandBuffer cmd, uint32_t frame_slot) {
                record_inline_scene(
                    cmd,
                    gbuffer_pipeline_,
                    gbuffer_pipeline_layout_,
                    gbuffer_target_.w,
                    gbuffer_target_.h,
                    frame_slot);
            },
            [this](
                VkCommandBuffer cmd,
                VkPipelineStageFlags src_stage,
                VkAccessFlags src_access,
                VkPipelineStageFlags dst_stage,
                VkAccessFlags dst_access) {
                const bool gbuffer_to_shader_read =
                    (src_stage &
                     (VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT)) != 0 &&
                    (dst_stage &
                     (VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT)) != 0;
                if (gbuffer_to_shader_read)
                {
                    if (emit_graph_barrier_gbuffer_to_consumers(cmd)) return;
                }
                cmd_memory_barrier(cmd, src_stage, src_access, dst_stage, dst_access);
            });
    }

    bool execute_pass_ssao(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.ssao_pass_executed) return true;
        if (!ctx.gbuffer_pass_executed) return false;

        if (!active_ssao_pass_enabled())
        {
            if (ao_target_.render_pass != VK_NULL_HANDLE && ao_target_.framebuffer != VK_NULL_HANDLE)
            {
                // Keep AO neutral when disabled so deferred shading remains stable.
                begin_render_pass_ssao(ctx.fi->cmd);
                vkCmdEndRenderPass(ctx.fi->cmd);
                if (!emit_graph_barrier_ssao_to_consumer(ctx.fi->cmd))
                {
                    cmd_memory_barrier(
                        ctx.fi->cmd,
                        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                        VK_ACCESS_SHADER_READ_BIT);
                }
            }
            return true;
        }

        const bool ssao_ready =
            ao_target_.render_pass != VK_NULL_HANDLE &&
            ao_target_.framebuffer != VK_NULL_HANDLE &&
            ssao_pipeline_ != VK_NULL_HANDLE &&
            ssao_pipeline_layout_ != VK_NULL_HANDLE &&
            deferred_set_ != VK_NULL_HANDLE &&
            ctx.global_set != VK_NULL_HANDLE;
        if (!ssao_ready) return false;

        begin_render_pass_ssao(ctx.fi->cmd);
        set_viewport_scissor(ctx.fi->cmd, ao_target_.w, ao_target_.h, true);
        vkCmdBindPipeline(ctx.fi->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, ssao_pipeline_);
        VkDescriptorSet sets[2] = {ctx.global_set, deferred_set_};
        vkCmdBindDescriptorSets(
            ctx.fi->cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            ssao_pipeline_layout_,
            0,
            2,
            sets,
            0,
            nullptr);
        vkCmdDraw(ctx.fi->cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(ctx.fi->cmd);

        if (!emit_graph_barrier_ssao_to_consumer(ctx.fi->cmd))
        {
            cmd_memory_barrier(
                ctx.fi->cmd,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT);
        }

        ctx.ssao_pass_executed = true;
        return true;
    }

    bool execute_pass_deferred_lighting(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        if (!ctx.fi) return false;
        if (ctx.deferred_lighting_pass_executed) return true;
        const shs::PassId deferred_pass_kind = resolve_compiled_pass_kind(pass);

        const bool chain_post =
            ctx.has_motion_blur_pass ||
            ctx.has_depth_of_field_pass;

        const bool deferred_ready_swapchain =
            deferred_lighting_pipeline_ != VK_NULL_HANDLE &&
            deferred_lighting_pipeline_layout_ != VK_NULL_HANDLE &&
            deferred_set_ != VK_NULL_HANDLE &&
            ctx.global_set != VK_NULL_HANDLE;
        const bool deferred_ready_post =
            deferred_lighting_post_pipeline_ != VK_NULL_HANDLE &&
            deferred_lighting_pipeline_layout_ != VK_NULL_HANDLE &&
            deferred_set_ != VK_NULL_HANDLE &&
            ctx.global_set != VK_NULL_HANDLE &&
            post_target_a_.render_pass != VK_NULL_HANDLE &&
            post_target_a_.framebuffer != VK_NULL_HANDLE;

        if (chain_post)
        {
            if (!deferred_ready_post) return false;
            begin_render_pass_post(ctx.fi->cmd, post_target_a_);
            set_viewport_scissor(ctx.fi->cmd, post_target_a_.w, post_target_a_.h, true);
            vkCmdBindPipeline(ctx.fi->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_lighting_post_pipeline_);
            VkDescriptorSet sets[2] = {ctx.global_set, deferred_set_};
            vkCmdBindDescriptorSets(
                ctx.fi->cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                deferred_lighting_pipeline_layout_,
                0,
                2,
                sets,
                0,
                nullptr);
            vkCmdDraw(ctx.fi->cmd, 3, 1, 0, 0);
            draw_light_volumes_debug(ctx.fi->cmd, scene_pipeline_layout_, ctx.frame_slot);
            vkCmdEndRenderPass(ctx.fi->cmd);
            post_target_a_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            if (!emit_graph_barrier_deferred_to_consumer(ctx.fi->cmd, deferred_pass_kind))
            {
                cmd_memory_barrier(
                    ctx.fi->cmd,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            }

            ctx.post_color_valid = true;
            ctx.post_color_source = 1u;
        }
        else
        {
            if (!deferred_ready_swapchain) return false;
            VkClearValue clear[2]{};
            clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
            clear[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = ctx.fi->render_pass;
            rp.framebuffer = ctx.fi->framebuffer;
            rp.renderArea.offset = {0, 0};
            rp.renderArea.extent = ctx.fi->extent;
            rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
            rp.pClearValues = clear;
            vkCmdBeginRenderPass(ctx.fi->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

            set_viewport_scissor(ctx.fi->cmd, ctx.fi->extent.width, ctx.fi->extent.height, true);
            vkCmdBindPipeline(ctx.fi->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, deferred_lighting_pipeline_);
            VkDescriptorSet sets[2] = {ctx.global_set, deferred_set_};
            vkCmdBindDescriptorSets(
                ctx.fi->cmd,
                VK_PIPELINE_BIND_POINT_GRAPHICS,
                deferred_lighting_pipeline_layout_,
                0,
                2,
                sets,
                0,
                nullptr);
            vkCmdDraw(ctx.fi->cmd, 3, 1, 0, 0);
            draw_light_volumes_debug(ctx.fi->cmd, scene_pipeline_layout_, ctx.frame_slot);
            vkCmdEndRenderPass(ctx.fi->cmd);
            ctx.scene_pass_executed = true;
        }

        ctx.deferred_lighting_pass_executed = true;
        return true;
    }

    bool execute_pass_motion_blur(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.motion_blur_pass_executed) return true;
        if (!active_motion_blur_pass_enabled()) return true;
        if (!ctx.post_color_valid)
        {
            if (!prepare_post_source_from_scene_color(ctx)) return true;
            if (!ctx.post_color_valid) return true;
        }

        const VkDescriptorSet post_set = post_source_descriptor_set_from_context(ctx);
        if (post_set == VK_NULL_HANDLE) return false;

        const bool output_to_post = ctx.has_depth_of_field_pass;
        VkPipeline pipe = output_to_post ? motion_blur_pipeline_ : motion_blur_scene_pipeline_;
        if (pipe == VK_NULL_HANDLE || deferred_lighting_pipeline_layout_ == VK_NULL_HANDLE || ctx.global_set == VK_NULL_HANDLE)
        {
            return false;
        }

        if (output_to_post)
        {
            if (post_target_b_.render_pass == VK_NULL_HANDLE || post_target_b_.framebuffer == VK_NULL_HANDLE) return false;
            begin_render_pass_post(ctx.fi->cmd, post_target_b_);
            set_viewport_scissor(ctx.fi->cmd, post_target_b_.w, post_target_b_.h, true);
        }
        else
        {
            VkClearValue clear[2]{};
            clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
            clear[1].depthStencil = {1.0f, 0};

            VkRenderPassBeginInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rp.renderPass = ctx.fi->render_pass;
            rp.framebuffer = ctx.fi->framebuffer;
            rp.renderArea.offset = {0, 0};
            rp.renderArea.extent = ctx.fi->extent;
            rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
            rp.pClearValues = clear;
            vkCmdBeginRenderPass(ctx.fi->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
            set_viewport_scissor(ctx.fi->cmd, ctx.fi->extent.width, ctx.fi->extent.height, true);
        }

        vkCmdBindPipeline(ctx.fi->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        VkDescriptorSet sets[2] = {ctx.global_set, post_set};
        vkCmdBindDescriptorSets(
            ctx.fi->cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            deferred_lighting_pipeline_layout_,
            0,
            2,
            sets,
            0,
            nullptr);
        vkCmdDraw(ctx.fi->cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(ctx.fi->cmd);

        if (output_to_post)
        {
            post_target_b_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            if (!emit_graph_barrier_motion_blur_to_consumer(ctx.fi->cmd))
            {
                cmd_memory_barrier(
                    ctx.fi->cmd,
                    VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    VK_ACCESS_SHADER_READ_BIT);
            }
            ctx.post_color_valid = true;
            ctx.post_color_source = 2u;
        }
        else
        {
            ctx.post_color_valid = false;
            ctx.post_color_source = 0u;
            ctx.scene_pass_executed = true;
        }

        ctx.motion_blur_pass_executed = true;
        return true;
    }

    bool execute_pass_depth_of_field(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        (void)pass;
        if (!ctx.fi) return false;
        if (ctx.depth_of_field_pass_executed) return true;
        if (!active_depth_of_field_pass_enabled()) return true;
        if (!ctx.post_color_valid)
        {
            if (!prepare_post_source_from_scene_color(ctx)) return true;
            if (!ctx.post_color_valid) return true;
        }

        if (dof_pipeline_ == VK_NULL_HANDLE || deferred_lighting_pipeline_layout_ == VK_NULL_HANDLE ||
            ctx.global_set == VK_NULL_HANDLE)
        {
            return false;
        }

        const VkDescriptorSet post_set = post_source_descriptor_set_from_context(ctx);
        if (post_set == VK_NULL_HANDLE) return false;

        VkClearValue clear[2]{};
        clear[0].color = {{0.03f, 0.035f, 0.045f, 1.0f}};
        clear[1].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo rp{};
        rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp.renderPass = ctx.fi->render_pass;
        rp.framebuffer = ctx.fi->framebuffer;
        rp.renderArea.offset = {0, 0};
        rp.renderArea.extent = ctx.fi->extent;
        rp.clearValueCount = vk_->has_depth_attachment() ? 2u : 1u;
        rp.pClearValues = clear;
        vkCmdBeginRenderPass(ctx.fi->cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);

        set_viewport_scissor(ctx.fi->cmd, ctx.fi->extent.width, ctx.fi->extent.height, true);
        vkCmdBindPipeline(ctx.fi->cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, dof_pipeline_);
        VkDescriptorSet sets[2] = {ctx.global_set, post_set};
        vkCmdBindDescriptorSets(
            ctx.fi->cmd,
            VK_PIPELINE_BIND_POINT_GRAPHICS,
            deferred_lighting_pipeline_layout_,
            0,
            2,
            sets,
            0,
            nullptr);
        vkCmdDraw(ctx.fi->cmd, 3, 1, 0, 0);
        vkCmdEndRenderPass(ctx.fi->cmd);

        ctx.post_color_valid = false;
        ctx.post_color_source = 0u;
        ctx.scene_pass_executed = true;
        ctx.depth_of_field_pass_executed = true;
        return true;
    }

    bool execute_pass_taa(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        (void)pass;
        if (!active_taa_pass_enabled()) return true;
        ctx.taa_pass_executed = true;
        return true;
    }

    bool execute_pass_noop(FramePassExecutionContext& ctx, const shs::RenderPathCompiledPass& pass)
    {
        (void)ctx;
        (void)pass;
        return true;
    }

    void build_frame_pass_dispatcher()
    {
        const auto wrap = [this](auto&& fn) {
            return [this, fn = std::forward<decltype(fn)>(fn)](
                       FramePassExecutionContext& c,
                       const shs::RenderPathCompiledPass& p) mutable {
                return execute_profiled_pass_handler(c, p, fn);
            };
        };

        shs::StandardRenderPathPassHandlers<FramePassExecutionContext> handlers{};
        handlers.shadow_map = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_shadow_map(c, p);
        });
        handlers.depth_prepass = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_depth_prepass(c, p);
        });
        handlers.light_culling = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_light_culling(c, p);
        });
        handlers.cluster_build = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_noop(c, p);
        });
        handlers.scene_forward = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_scene(c, p);
        });
        handlers.gbuffer = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_gbuffer(c, p);
        });
        handlers.ssao = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_ssao(c, p);
        });
        handlers.deferred_lighting = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_deferred_lighting(c, p);
        });
        handlers.tonemap = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_noop(c, p);
        });
        handlers.taa = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_taa(c, p);
        });
        handlers.motion_blur = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_motion_blur(c, p);
        });
        handlers.depth_of_field = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_depth_of_field(c, p);
        });
        handlers.fallback_noop = wrap([this](FramePassExecutionContext& c, const shs::RenderPathCompiledPass& p) {
            return execute_pass_noop(c, p);
        });

        const bool ok = shs::register_standard_render_path_handlers(frame_pass_dispatcher_, handlers);
        if (!ok)
        {
            std::fprintf(stderr, "[render-path][dispatch][error] Failed to register standard pass handlers.\n");
        }
    }

    void draw_frame(float dt, float t)
    {
        int dw = 0;
        int dh = 0;
        SDL_Vulkan_GetDrawableSize(win_, &dw, &dh);
        if (dw <= 0 || dh <= 0)
        {
            SDL_Delay(16);
            return;
        }

        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = dw;
        frame.height = dh;

        shs::VulkanRenderBackend::FrameInfo fi{};
        if (!vk_->begin_frame(ctx_, frame, fi))
        {
            SDL_Delay(2);
            return;
        }
        const uint32_t frame_slot = shs::vk_frame_slot(frame.frame_index, kWorkerPoolRingSize);
        const VkDescriptorSet global_set = frame_resources_.at_slot(frame_slot).global_set;
        if (global_set == VK_NULL_HANDLE)
        {
            throw std::runtime_error("Frame descriptor set unavailable");
        }
        collect_gpu_pass_timing_results(frame_slot);

        const uint64_t current_swapchain_gen = vk_->swapchain_generation();
        if (observed_swapchain_generation_ != current_swapchain_gen)
        {
            ++swapchain_generation_change_count_;
            observed_swapchain_generation_ = current_swapchain_gen;
        }

        ensure_render_targets(fi.extent.width, fi.extent.height);
        if (pipeline_gen_ != vk_->swapchain_generation())
        {
            create_pipelines(true, "swapchain-generation");
        }
        update_culling_debug_stats(frame_slot);

        update_frame_data(dt, t, fi.extent.width, fi.extent.height, frame_slot);

        std::vector<VkCommandBuffer> depth_secondaries{};
        std::vector<VkCommandBuffer> scene_secondaries{};
        if (use_multithread_recording_)
        {
            if ((enable_depth_prepass_ || enable_scene_pass_) && !reset_worker_pools_for_frame(frame_slot))
            {
                throw std::runtime_error("Failed to reset worker command pools");
            }

            if (enable_depth_prepass_ &&
                !record_secondary_lists(
                    depth_target_.render_pass,
                    depth_target_.framebuffer,
                    depth_pipeline_,
                    depth_pipeline_layout_,
                    depth_target_.w,
                    depth_target_.h,
                    true,
                    true,
                    frame_slot,
                    depth_secondaries))
            {
                throw std::runtime_error("Failed to record depth secondary command buffers");
            }
            if (enable_scene_pass_ &&
                !record_secondary_lists(
                    fi.render_pass,
                    fi.framebuffer,
                    scene_pipeline_,
                    scene_pipeline_layout_,
                    fi.extent.width,
                    fi.extent.height,
                    true,
                    true,
                    frame_slot,
                    scene_secondaries))
            {
                throw std::runtime_error("Failed to record scene secondary command buffers");
            }
        }

        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (vkBeginCommandBuffer(fi.cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer failed");
        }
        if (vkBeginCommandBuffer(fi.compute_cmd, &bi) != VK_SUCCESS)
        {
            throw std::runtime_error("vkBeginCommandBuffer compute_cmd failed");
        }
        begin_gpu_pass_timing_recording(fi.cmd, frame_slot);
        ensure_history_color_shader_read_layout(fi.cmd);

        const shs::RenderPathExecutionPlan plan = make_active_frame_execution_plan();
        frame_graph_barrier_edges_emitted_ = 0u;
        frame_graph_barrier_fallback_count_ = 0u;
        const auto plan_has_pass = [&plan](shs::PassId pass_id) -> bool {
            for (const auto& p : plan.pass_chain)
            {
                if (p.pass_id == pass_id) return true;
                if (shs::parse_pass_id(p.id) == pass_id) return true;
            }
            return false;
        };
        FramePassExecutionContext pass_ctx{};
        pass_ctx.fi = &fi;
        pass_ctx.frame_slot = frame_slot;
        pass_ctx.global_set = global_set;
        pass_ctx.depth_secondaries = &depth_secondaries;
        pass_ctx.scene_secondaries = &scene_secondaries;
        pass_ctx.depth_prepass_enabled = enable_depth_prepass_;
        pass_ctx.scene_enabled = enable_scene_pass_;
        pass_ctx.light_culling_enabled = enable_light_culling_;
        pass_ctx.gpu_light_culler_enabled = gpu_light_culler_enabled();
        pass_ctx.has_motion_blur_pass =
            plan_has_pass(shs::PassId::MotionBlur) && active_motion_blur_pass_enabled();
        pass_ctx.has_depth_of_field_pass =
            plan_has_pass(shs::PassId::DepthOfField) && active_depth_of_field_pass_enabled();
        pass_ctx.post_color_valid = false;
        pass_ctx.post_color_source = 0u;

        shs::RenderPathPassDispatchResult dispatch_result{};
        const bool dispatch_ok = frame_pass_dispatcher_.execute(plan, pass_ctx, &dispatch_result);
        finalize_gpu_pass_timing_recording(frame_slot);
        dispatch_total_cpu_ms_ = dispatch_result.total_cpu_ms;
        dispatch_slowest_pass_cpu_ms_ = dispatch_result.slowest_cpu_ms;
        dispatch_slowest_pass_id_ = dispatch_result.slowest_pass_id;
        if (!dispatch_result.warnings.empty() && !pass_dispatch_warning_emitted_)
        {
            for (const auto& w : dispatch_result.warnings)
            {
                std::fprintf(stderr, "[render-path][dispatch][warn] %s\n", w.c_str());
            }
            pass_dispatch_warning_emitted_ = true;
        }
        if (!dispatch_ok || !dispatch_result.errors.empty())
        {
            const std::string err = dispatch_result.errors.empty()
                ? std::string("Render-path pass dispatch failed.")
                : dispatch_result.errors.front();
            throw std::runtime_error(err);
        }

        frame_gbuffer_pass_executed_ = pass_ctx.gbuffer_pass_executed;
        frame_ssao_pass_executed_ = pass_ctx.ssao_pass_executed;
        frame_deferred_lighting_pass_executed_ = pass_ctx.deferred_lighting_pass_executed;
        frame_motion_blur_pass_executed_ = pass_ctx.motion_blur_pass_executed;
        frame_depth_of_field_pass_executed_ = pass_ctx.depth_of_field_pass_executed;
        frame_taa_pass_executed_ = pass_ctx.taa_pass_executed;
        frame_deferred_emulated_scene_pass_ = pass_ctx.deferred_emulated_scene_pass;
        if (frame_deferred_emulated_scene_pass_ && !deferred_emulation_warning_emitted_)
        {
            std::fprintf(
                stderr,
                "[render-path][deferred][warn] Deferred pass chain is active, but lighting is currently emulated via scene pass.\n");
            deferred_emulation_warning_emitted_ = true;
        }

        if (!pass_ctx.scene_pass_executed)
        {
            draw_scene_clear_only(fi.cmd, fi, frame_slot);
        }

        record_history_color_copy(fi.cmd, fi);
        (void)record_phase_f_snapshot_copy(fi.cmd, fi);

        if (vkEndCommandBuffer(fi.cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkEndCommandBuffer failed");
        }
        if (fi.has_compute_work && vkEndCommandBuffer(fi.compute_cmd) != VK_SUCCESS)
        {
            throw std::runtime_error("vkEndCommandBuffer compute_cmd failed");
        }

        vk_->end_frame(fi);
        if (phase_f_snapshot_copy_submitted_)
        {
            vk_->wait_idle();
            const bool wrote = write_phase_f_snapshot_from_readback();
            if (!wrote)
            {
                phase_f_snapshot_failed_ = true;
                std::fprintf(stderr, "[phase-f] Snapshot write failed: %s\n", phase_f_snapshot_path_.c_str());
                phase_f_write_json_line(
                    "{\"event\":\"snapshot_result\",\"ok\":false,\"entry\":" +
                    std::to_string(phase_f_active_entry_slot_ + 1u) +
                    ",\"path\":\"" + phase_f_snapshot_path_ + "\"}");
            }
            else
            {
                phase_f_snapshot_completed_ = true;
                std::fprintf(stderr, "[phase-f] Snapshot saved: %s\n", phase_f_snapshot_path_.c_str());
                phase_f_write_json_line(
                    "{\"event\":\"snapshot_result\",\"ok\":true,\"entry\":" +
                    std::to_string(phase_f_active_entry_slot_ + 1u) +
                    ",\"path\":\"" + phase_f_snapshot_path_ + "\"}");
            }
            phase_f_snapshot_copy_submitted_ = false;
            phase_f_snapshot_request_armed_ = false;
        }
        ctx_.frame_index++;
    }

    void update_window_title(float avg_ms)
    {
        const char* mode_name = shs::technique_mode_name(active_technique_);
        const char* light_tech_name = lighting_technique_name(render_technique_preset_);
        const char* composition_name = active_composition_recipe_.name.empty()
            ? "n/a"
            : active_composition_recipe_.name.c_str();
        const char* post_stack_name =
            shs::render_composition_post_stack_preset_name(active_composition_recipe_.post_stack);
        const shs::RenderPathRecipe& active_recipe = render_path_executor_.active_recipe();
        const shs::RenderPathResourcePlan& resource_plan = render_path_executor_.active_resource_plan();
        const char* recipe_name = active_recipe.name.empty() ? "n/a" : active_recipe.name.c_str();
        const char* recipe_status = render_path_executor_.active_plan_valid() ? "OK" : "Fallback";
        const char* cull_name = shs::light_culling_mode_name(culling_mode_);
        const char* culler_backend = vulkan_culler_backend_name(vulkan_culler_backend_);
        const char* rec_mode = use_multithread_recording_ ? "MT-secondary" : "inline";
        const float switch_in = auto_cycle_technique_ ? std::max(0.0f, kTechniqueSwitchPeriodSec - technique_switch_accum_sec_) : 0.0f;
        const size_t comp_total = composition_cycle_order_.size();
        const size_t comp_slot = (comp_total > 0u) ? (active_composition_index_ % comp_total) + 1u : 0u;
        const char* phase_f_state = "off";
        if (phase_f_config_.enabled)
        {
            switch (phase_f_stage_)
            {
                case PhaseFBenchmarkStage::Warmup: phase_f_state = "warmup"; break;
                case PhaseFBenchmarkStage::Sample: phase_f_state = "sample"; break;
                case PhaseFBenchmarkStage::AwaitSnapshot: phase_f_state = "snapshot"; break;
                case PhaseFBenchmarkStage::Disabled:
                default: phase_f_state = phase_f_finished_ ? "done" : "idle"; break;
            }
        }
        const size_t phase_f_total = phase_f_plan_indices_.size();
        const size_t phase_f_slot =
            (phase_f_total > 0u && phase_f_active_entry_slot_ < phase_f_total) ? (phase_f_active_entry_slot_ + 1u) : 0u;
        const char* phase_g_state = "off";
        if (phase_g_config_.enabled)
        {
            if (phase_g_state_.finished) phase_g_state = "done";
            else if (phase_g_state_.started) phase_g_state = "run";
            else phase_g_state = "idle";
        }
        const double avg_refs = (cull_debug_list_count_ > 0)
            ? static_cast<double>(cull_debug_total_refs_) / static_cast<double>(cull_debug_list_count_)
            : 0.0;
        const uint32_t visible_draws = visible_instance_count_ + (floor_visible_ ? 1u : 0u);
        const uint32_t total_draws = static_cast<uint32_t>(instances_.size()) + 1u;
        const uint32_t culled_total = (active_light_count_ > visible_light_count_) ? (active_light_count_ - visible_light_count_) : 0u;
        const bool framebuffer_debug_enabled = framebuffer_debug_preset_ != FramebufferDebugPreset::FinalComposite;
        const bool framebuffer_debug_supported =
            (!framebuffer_debug_preset_requires_motion_pass(framebuffer_debug_preset_) || active_motion_blur_pass_enabled()) &&
            (!framebuffer_debug_preset_requires_dof_pass(framebuffer_debug_preset_) || active_depth_of_field_pass_enabled());
        const char* framebuffer_debug_state = framebuffer_debug_enabled ? "on" : "off";
        const char* framebuffer_debug_name = framebuffer_debug_preset_name(framebuffer_debug_preset_);
        const char* framebuffer_debug_availability = framebuffer_debug_enabled
            ? (framebuffer_debug_supported ? "ready" : "missing")
            : "idle";
        const bool semantic_debug_has_resource = semantic_debug_enabled_ &&
            (shs::find_render_path_resource_by_semantic(resource_plan, active_semantic_debug_) != nullptr);
        const char* semantic_debug_state = semantic_debug_enabled_ ? "on" : "off";
        const char* semantic_debug_name = semantic_debug_enabled_
            ? shs::pass_semantic_name(active_semantic_debug_)
            : "none";
        const char* semantic_debug_availability = semantic_debug_enabled_
            ? (semantic_debug_has_resource ? "ready" : "missing")
            : (semantic_debug_targets_.empty() ? "n/a" : "idle");
        const bool deferred_mode =
            (active_technique_ == shs::TechniqueMode::Deferred) ||
            (active_technique_ == shs::TechniqueMode::TiledDeferred);
        const char* deferred_state = deferred_mode
            ? (frame_deferred_emulated_scene_pass_ ? "emul" : "native")
            : "n/a";
        const bool temporal_copy_supported = supports_swapchain_history_copy();
        const bool temporal_enabled = active_taa_pass_enabled() && temporal_settings_.accumulation_enabled;
        const char* temporal_jitter_state = temporal_enabled
            ? (temporal_copy_supported ? "on" : "fallback")
            : "off";
        const char* taa_state = frame_taa_pass_executed_ ? "on" : "off";
        const char* dispatch_slowest_pass_name =
            dispatch_slowest_pass_id_.empty() ? "n/a" : dispatch_slowest_pass_id_.c_str();
        const char* gpu_slowest_pass_name =
            gpu_pass_slowest_id_.empty() ? "n/a" : gpu_pass_slowest_id_.c_str();
        const char* gpu_timing_state =
            gpu_pass_timing_state_.empty() ? "n/a" : gpu_pass_timing_state_.c_str();
        const double gpu_total_ms = gpu_pass_timing_valid_ ? gpu_pass_total_ms_ : 0.0;
        const double gpu_slowest_ms = gpu_pass_timing_valid_ ? gpu_pass_slowest_ms_ : 0.0;
        const char* target_rebuild_reason =
            render_target_last_rebuild_reason_.empty() ? "none" : render_target_last_rebuild_reason_.c_str();
        const char* pipeline_rebuild_reason =
            pipeline_last_rebuild_reason_.empty() ? "none" : pipeline_last_rebuild_reason_.c_str();

        char title[1024];
        std::snprintf(
            title,
            sizeof(title),
            "%s | comp:%s[%zu/%zu] pst:%s pf:%s[%zu/%zu] pg:%s[c:%llu] | light:%s exp:%.2f g:%.2f | rpath:%s(%s) mode:%s def:%s[g:%s a:%s l:%s t:%s m:%s d:%s] | tmp:%s j(%.3f,%.3f) | dbg:F5 %s/%s(%s) F8 %s/%s(%s) | cull:%s(%s) | rec:%s rsrc:%zu bind:%zu br:%u/%u lay:%u alias:%u/%u gbr:%u fb:%u cpu:%.2fms slow:%s %.2f gpu:%s %.2f slow:%s %.2f s:%u r:%u | rb:t%llu(%s) p%llu(%s) sg:%llu | lights:%u/%u[p:%u s:%u r:%u t:%u] | lvol:%s occ:%s/%s lobj:%s culled:%u[f:%u o:%u p:%u] | shad:sun:%s(%.2f) spot:%u point:%u | cfg:orb%.2f h%.1f r%.2f i%.2f | draws:%u/%u | tile:%ux%u sz:%u z:%u | refs:%llu avg:%.1f max:%u nz:%u/%u | lightsw:%s %.1fs | %.2f ms",
            kAppName,
            composition_name,
            comp_slot,
            comp_total,
            post_stack_name,
            phase_f_state,
            phase_f_slot,
            phase_f_total,
            phase_g_state,
            static_cast<unsigned long long>(phase_g_state_.cycles),
            light_tech_name,
            tonemap_exposure_,
            tonemap_gamma_,
            recipe_name,
            recipe_status,
            mode_name,
            deferred_state,
            frame_gbuffer_pass_executed_ ? "on" : "off",
            frame_ssao_pass_executed_ ? "on" : "off",
            frame_deferred_lighting_pass_executed_ ? "on" : "off",
            taa_state,
            frame_motion_blur_pass_executed_ ? "on" : "off",
            frame_depth_of_field_pass_executed_ ? "on" : "off",
            temporal_jitter_state,
            temporal_state_.jitter_ndc.x,
            temporal_state_.jitter_ndc.y,
            framebuffer_debug_state,
            framebuffer_debug_name,
            framebuffer_debug_availability,
            semantic_debug_state,
            semantic_debug_name,
            semantic_debug_availability,
            cull_name,
            culler_backend,
            rec_mode,
            resource_plan.resources.size(),
            resource_plan.pass_bindings.size(),
            barrier_edge_count_,
            barrier_memory_edge_count_,
            barrier_layout_edge_count_,
            barrier_alias_class_count_,
            barrier_alias_slot_count_,
            frame_graph_barrier_edges_emitted_,
            frame_graph_barrier_fallback_count_,
            dispatch_total_cpu_ms_,
            dispatch_slowest_pass_name,
            dispatch_slowest_pass_cpu_ms_,
            gpu_timing_state,
            gpu_total_ms,
            gpu_slowest_pass_name,
            gpu_slowest_ms,
            gpu_pass_sample_count_,
            gpu_pass_rejected_sample_count_,
            static_cast<unsigned long long>(render_target_rebuild_count_),
            target_rebuild_reason,
            static_cast<unsigned long long>(pipeline_rebuild_count_),
            pipeline_rebuild_reason,
            static_cast<unsigned long long>(swapchain_generation_change_count_),
            visible_light_count_,
            active_light_count_,
            point_count_active_,
            spot_count_active_,
            rect_count_active_,
            tube_count_active_,
            show_light_volumes_debug_ ? "on" : "off",
            enable_scene_occlusion_ ? "on" : "off",
            enable_light_occlusion_ ? "on" : "off",
            shs::light_object_cull_mode_name(light_object_cull_mode_),
            culled_total,
            light_frustum_rejected_,
            light_occlusion_rejected_,
            light_prefilter_rejected_,
            (shadow_settings_.enable && enable_sun_shadow_) ? "on" : "off",
            sun_shadow_strength_,
            spot_shadow_count_,
            point_shadow_count_,
            light_orbit_scale_,
            light_height_bias_,
            light_range_scale_,
            light_intensity_scale_,
            visible_draws,
            total_draws,
            tile_w_,
            tile_h_,
            light_tile_size_,
            cluster_z_slices_,
            static_cast<unsigned long long>(cull_debug_total_refs_),
            avg_refs,
            cull_debug_max_list_size_,
            cull_debug_non_empty_lists_,
            cull_debug_list_count_,
            auto_cycle_technique_ ? "auto" : "manual",
            switch_in,
            avg_ms);
        SDL_SetWindowTitle(win_, title);
    }

    void handle_event(const SDL_Event& e)
    {
        if (e.type == SDL_QUIT) pending_quit_action_ = true;

        if (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP)
        {
            const bool down = (e.type == SDL_KEYDOWN);
            switch (e.key.keysym.sym)
            {
                case SDLK_w:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetForward,
                        down);
                    break;
                case SDLK_s:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetBackward,
                        down);
                    break;
                case SDLK_a:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetLeft,
                        down);
                    break;
                case SDLK_d:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetRight,
                        down);
                    break;
                case SDLK_q:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetDescend,
                        down);
                    break;
                case SDLK_e:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetAscend,
                        down);
                    break;
                case SDLK_LSHIFT:
                case SDLK_RSHIFT:
                    shs::append_runtime_input_event(
                        pending_input_events_,
                        shs::RuntimeInputEventType::SetBoost,
                        down);
                    break;
                default:
                    break;
            }
        }

        if (e.type == SDL_MOUSEBUTTONDOWN || e.type == SDL_MOUSEBUTTONUP)
        {
            const bool down = (e.type == SDL_MOUSEBUTTONDOWN);
            if (e.button.button == SDL_BUTTON_LEFT)
            {
                shs::append_runtime_input_event(
                    pending_input_events_,
                    shs::RuntimeInputEventType::SetLeftMouseDown,
                    down);
            }
            if (e.button.button == SDL_BUTTON_RIGHT)
            {
                shs::append_runtime_input_event(
                    pending_input_events_,
                    shs::RuntimeInputEventType::SetRightMouseDown,
                    down);
            }
        }

        if (e.type == SDL_MOUSEMOTION)
        {
            pending_input_events_.push_back(
                shs::make_mouse_delta_input_event(
                    static_cast<float>(e.motion.xrel),
                    static_cast<float>(e.motion.yrel)));
        }

        if (e.type == SDL_KEYDOWN)
        {
            pending_keydown_actions_.push_back(e.key.keysym.sym);
        }

        if (e.type == SDL_WINDOWEVENT &&
            (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_RESIZED))
        {
            vk_->request_resize(e.window.data1, e.window.data2);
        }
    }

    void main_loop()
    {
        running_ = true;
        runtime_state_.camera.pos = camera_.pos;
        runtime_state_.camera.yaw = camera_.yaw;
        runtime_state_.camera.pitch = camera_.pitch;
        runtime_state_.quit_requested = false;
        input_latch_ = shs::RuntimeInputLatch{};
        pending_input_events_.clear();

        using clock = std::chrono::steady_clock;
        auto last = clock::now();
        auto title_t0 = last;
        float ema_ms = 16.0f;

        while (running_)
        {
            SDL_Event e{};
            while (SDL_PollEvent(&e))
            {
                handle_event(e);
            }
            input_latch_ = shs::reduce_runtime_input_latch(input_latch_, pending_input_events_);
            pending_input_events_.clear();
            apply_pending_keydown_actions();

            const bool look_drag = input_latch_.left_mouse_down || input_latch_.right_mouse_down;
            if (look_drag != relative_mouse_mode_)
            {
                relative_mouse_mode_ = look_drag;
                SDL_SetRelativeMouseMode(relative_mouse_mode_ ? SDL_TRUE : SDL_FALSE);
                input_latch_ = shs::clear_runtime_input_frame_deltas(input_latch_);
            }

            auto now = clock::now();
            float dt = std::chrono::duration<float>(now - last).count();
            last = now;
            dt = std::clamp(dt, 1.0f / 240.0f, 1.0f / 15.0f);
            time_sec_ += dt;
            if (auto_cycle_technique_)
            {
                technique_switch_accum_sec_ += dt;
                if (technique_switch_accum_sec_ >= kTechniqueSwitchPeriodSec)
                {
                    cycle_lighting_technique();
                    technique_switch_accum_sec_ = 0.0f;
                }
            }

            auto cpu_t0 = clock::now();
            draw_frame(dt, time_sec_);
            auto cpu_t1 = clock::now();

            const float frame_ms = std::chrono::duration<float, std::milli>(cpu_t1 - cpu_t0).count();
            ema_ms = glm::mix(ema_ms, frame_ms, 0.08f);
            phase_f_step_after_frame(frame_ms, ema_ms);
            phase_g_step_after_frame(frame_ms, ema_ms, dt);

            if (std::chrono::duration<float>(now - title_t0).count() >= 0.20f)
            {
                update_window_title(ema_ms);
                title_t0 = now;
            }
        }

        if (vk_ && vk_->device() != VK_NULL_HANDLE)
        {
            (void)vkDeviceWaitIdle(vk_->device());
        }
        if (relative_mouse_mode_)
        {
            SDL_SetRelativeMouseMode(SDL_FALSE);
            relative_mouse_mode_ = false;
        }
    }

    void apply_pending_keydown_actions()
    {
        for (SDL_Keycode key : pending_keydown_actions_)
        {
            switch (key)
            {
                case SDLK_ESCAPE:
                    pending_quit_action_ = true;
                    break;
                case SDLK_F1:
                    use_multithread_recording_ = !use_multithread_recording_;
                    break;
                case SDLK_F2:
                    // cycle_render_path_recipe();
                    break;
                case SDLK_TAB:
                    // cycle_render_path_recipe();
                    break;
                case SDLK_F3:
                    // cycle_render_composition_recipe();
                    break;
                case SDLK_F4:
                    // cycle_lighting_technique();
                    break;
                case SDLK_F5:
                    cycle_framebuffer_debug_target();
                    break;
                case SDLK_F6:
                    vulkan_culler_backend_ =
                        (vulkan_culler_backend_ == VulkanCullerBackend::GpuCompute)
                            ? VulkanCullerBackend::Disabled
                            : VulkanCullerBackend::GpuCompute;
                    break;
                case SDLK_F7:
                    show_light_volumes_debug_ = !show_light_volumes_debug_;
                    break;
                case SDLK_F8:
                    cycle_semantic_debug_target();
                    break;
                case SDLK_F9:
                    if (!active_taa_pass_enabled())
                    {
                        std::fprintf(
                            stderr,
                            "[render-path][temporal] Active composition has TAA disabled.\n");
                        break;
                    }
                    temporal_settings_.accumulation_enabled = !temporal_settings_.accumulation_enabled;
                    temporal_settings_.jitter_enabled = temporal_settings_.accumulation_enabled;
                    std::fprintf(
                        stderr,
                        "[render-path][temporal] Accumulation+jitter: %s\n",
                        temporal_settings_.accumulation_enabled ? "ON" : "OFF");
                    if (temporal_settings_.accumulation_enabled && !supports_swapchain_history_copy())
                    {
                        std::fprintf(
                            stderr,
                            "[render-path][temporal] Warning: swapchain transfer-src unsupported, temporal history copy disabled.\n");
                    }
                    break;
                case SDLK_F10:
                    print_controls();
                    print_composition_catalog();
                    break;
                case SDLK_F11:
                    auto_cycle_technique_ = !auto_cycle_technique_;
                    technique_switch_accum_sec_ = 0.0f;
                    break;
                case SDLK_F12:
                    enable_sun_shadow_ = !enable_sun_shadow_;
                    break;
                case SDLK_1:
                    light_orbit_scale_ = std::clamp(light_orbit_scale_ - 0.10f, 0.35f, 2.50f);
                    break;
                case SDLK_2:
                    light_orbit_scale_ = std::clamp(light_orbit_scale_ + 0.10f, 0.35f, 2.50f);
                    break;
                case SDLK_3:
                    light_height_bias_ = std::clamp(light_height_bias_ - 0.25f, -3.0f, 6.0f);
                    break;
                case SDLK_4:
                    light_height_bias_ = std::clamp(light_height_bias_ + 0.25f, -3.0f, 6.0f);
                    break;
                case SDLK_5:
                    light_range_scale_ = std::clamp(light_range_scale_ - 0.10f, 0.50f, 2.00f);
                    break;
                case SDLK_6:
                    light_range_scale_ = std::clamp(light_range_scale_ + 0.10f, 0.50f, 2.00f);
                    break;
                case SDLK_7:
                    light_intensity_scale_ = std::clamp(light_intensity_scale_ - 0.10f, 0.30f, 2.50f);
                    break;
                case SDLK_8:
                    light_intensity_scale_ = std::clamp(light_intensity_scale_ + 0.10f, 0.30f, 2.50f);
                    break;
                case SDLK_9:
                    sun_shadow_strength_ = std::clamp(sun_shadow_strength_ - 0.05f, 0.0f, 1.0f);
                    break;
                case SDLK_0:
                    sun_shadow_strength_ = std::clamp(sun_shadow_strength_ + 0.05f, 0.0f, 1.0f);
                    break;
                case SDLK_r:
                    light_orbit_scale_ = 1.0f;
                    light_height_bias_ = 0.0f;
                    light_range_scale_ = 1.0f;
                    light_intensity_scale_ = 1.0f;
                    enable_sun_shadow_ = false;
                    sun_shadow_strength_ = 0.0f;
                    break;
                case SDLK_MINUS:
                case SDLK_KP_MINUS:
                    active_light_count_ = (active_light_count_ > 64u) ? (active_light_count_ - 64u) : 64u;
                    break;
                case SDLK_EQUALS:
                case SDLK_PLUS:
                case SDLK_KP_PLUS:
                    active_light_count_ = std::min<uint32_t>(kMaxLights, active_light_count_ + 64u);
                    break;
                default:
                    break;
            }
        }
        pending_keydown_actions_.clear();
    }


    // cleanup consolidated at the top

    struct LightAnim
    {
        shs::LightType type = shs::LightType::Point;
        float angle0 = 0.0f;
        float orbit_radius = 6.0f;
        float height = 2.6f;
        float speed = 1.0f;
        float range = 4.8f;
        float phase = 0.0f;
        glm::vec3 color{1.0f};
        float intensity = 6.0f;
        shs::LightAttenuationModel attenuation_model = shs::LightAttenuationModel::Smooth;
        float attenuation_power = 1.0f;
        float attenuation_bias = 0.05f;
        float attenuation_cutoff = 0.0f;
        glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
        glm::vec3 rect_right_ws{1.0f, 0.0f, 0.0f};
        glm::vec2 spot_inner_outer{glm::radians(16.0f), glm::radians(26.0f)};
        glm::vec4 shape_params{0.0f};
    };

private:
    bool cleaned_up_ = false;
    bool running_ = false;
    bool sdl_ready_ = false;
    SDL_Window* win_ = nullptr;

    shs::Context ctx_{};
    std::vector<std::unique_ptr<shs::IRenderBackend>> keep_{};
    shs::VulkanRenderBackend* vk_ = nullptr;

    std::unique_ptr<shs::ThreadPoolJobSystem> jobs_{};
    uint32_t worker_count_ = 1;
    std::vector<WorkerPool> worker_pools_{};

    std::vector<Vertex> vertices_{};
    std::vector<uint32_t> indices_{};
    std::vector<Vertex> floor_vertices_{};
    std::vector<uint32_t> floor_indices_{};
    std::vector<Vertex> cone_vertices_{};
    std::vector<uint32_t> cone_indices_{};
    std::vector<Vertex> box_vertices_{};
    std::vector<uint32_t> box_indices_{};
    std::vector<uint32_t> sphere_line_indices_{};
    std::vector<uint32_t> cone_line_indices_{};
    std::vector<uint32_t> box_line_indices_{};
    std::vector<Vertex> capsule_vertices_{};
    std::vector<uint32_t> capsule_indices_{};
    std::vector<Vertex> cylinder_vertices_{};
    std::vector<uint32_t> cylinder_indices_{};
    std::vector<Instance> instances_{};
    std::vector<glm::mat4> instance_models_{};
    std::vector<uint8_t> instance_visible_mask_{};
    std::vector<uint32_t> frustum_visible_instance_indices_{};
    std::vector<shs::SceneShape> instance_cull_shapes_{};
    JPH::ShapeRefC sphere_shape_jolt_{};
    JPH::ShapeRefC box_shape_jolt_{};
    JPH::ShapeRefC cone_shape_jolt_{};
    JPH::ShapeRefC capsule_shape_jolt_{};
    JPH::ShapeRefC cylinder_shape_jolt_{};
    std::vector<LightAnim> light_anim_{};
    shs::LightSet light_set_{};
    std::vector<shs::CullingLightGPU> gpu_lights_{};
    std::vector<ShadowLightGPU> shadow_lights_gpu_{};
    std::vector<LocalShadowCaster> local_shadow_casters_{};
    std::vector<shs::AABB> visible_object_aabbs_{};
    shs::DebugMesh sphere_occluder_mesh_{};
    shs::DebugMesh cone_occluder_mesh_{};
    shs::DebugMesh box_occluder_mesh_{};
    shs::DebugMesh capsule_occluder_mesh_{};
    shs::DebugMesh cylinder_occluder_mesh_{};
    shs::DebugMesh floor_occluder_mesh_{};
    std::vector<float> scene_occlusion_depth_{};
    std::vector<float> light_occlusion_depth_{};
    glm::mat4 sun_shadow_view_proj_{1.0f};
    shs::AABB sphere_local_aabb_{};
    shs::AABB cone_local_aabb_{};
    shs::AABB box_local_aabb_{};
    shs::AABB capsule_local_aabb_{};
    shs::AABB cylinder_local_aabb_{};
    shs::Sphere sphere_local_bound_{};
    shs::Sphere cone_local_bound_{};
    shs::Sphere box_local_bound_{};
    shs::Sphere capsule_local_bound_{};
    shs::Sphere cylinder_local_bound_{};
    shs::AABB floor_local_aabb_{};
    shs::AABB shadow_scene_static_aabb_{};
    bool shadow_scene_static_bounds_ready_ = false;
    glm::mat4 floor_model_{1.0f};
    glm::vec4 floor_material_color_{1.0f};
    glm::vec4 floor_material_params_{0.0f, 0.72f, 1.0f, 0.0f};

    GpuBuffer vertex_buffer_{};
    GpuBuffer index_buffer_{};
    GpuBuffer floor_vertex_buffer_{};
    GpuBuffer floor_index_buffer_{};
    GpuBuffer cone_vertex_buffer_{};
    GpuBuffer cone_index_buffer_{};
    GpuBuffer box_vertex_buffer_{};
    GpuBuffer box_index_buffer_{};
    GpuBuffer sphere_line_index_buffer_{};
    GpuBuffer cone_line_index_buffer_{};
    GpuBuffer box_line_index_buffer_{};
    GpuBuffer capsule_vertex_buffer_{};
    GpuBuffer capsule_index_buffer_{};
    GpuBuffer cylinder_vertex_buffer_{};
    GpuBuffer cylinder_index_buffer_{};
    shs::VkFrameRing<FrameResources, kWorkerPoolRingSize> frame_resources_{};

    CameraUBO camera_ubo_{};
    DepthTarget depth_target_{};
    GBufferTarget gbuffer_target_{};
    AmbientOcclusionTarget ao_target_{};
    PostColorTarget post_target_a_{};
    PostColorTarget post_target_b_{};
    shs::VkRenderPathTemporalResources temporal_resources_{};
    VkImageLayout post_target_a_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout post_target_b_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
    bool post_color_copy_support_warning_emitted_ = false;
    LayeredDepthTarget sun_shadow_target_{};
    LayeredDepthTarget local_shadow_target_{};

    VkDescriptorSetLayout global_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout deferred_set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool deferred_descriptor_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet deferred_set_ = VK_NULL_HANDLE;
    VkDescriptorSet deferred_post_a_set_ = VK_NULL_HANDLE;
    VkDescriptorSet deferred_post_b_set_ = VK_NULL_HANDLE;
    VkSampler depth_sampler_ = VK_NULL_HANDLE;

    VkPipelineLayout shadow_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline shadow_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout depth_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline depth_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout scene_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline scene_pipeline_ = VK_NULL_HANDLE;
    VkPipeline scene_wire_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout gbuffer_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline gbuffer_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout ssao_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline ssao_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout deferred_lighting_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline deferred_lighting_pipeline_ = VK_NULL_HANDLE;
    VkPipeline deferred_lighting_post_pipeline_ = VK_NULL_HANDLE;
    VkPipeline motion_blur_pipeline_ = VK_NULL_HANDLE;
    VkPipeline motion_blur_scene_pipeline_ = VK_NULL_HANDLE;
    VkPipeline dof_pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout compute_pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline depth_reduce_pipeline_ = VK_NULL_HANDLE;
    VkPipeline compute_pipeline_ = VK_NULL_HANDLE;

    uint64_t pipeline_gen_ = 0;
    uint64_t observed_swapchain_generation_ = 0;
    uint64_t swapchain_generation_change_count_ = 0;
    uint64_t render_target_rebuild_count_ = 0;
    uint64_t pipeline_rebuild_count_ = 0;
    std::string render_target_last_rebuild_reason_{"init"};
    std::string pipeline_last_rebuild_reason_{"init"};
    uint32_t tile_w_ = 0;
    uint32_t tile_h_ = 0;
    uint32_t active_light_count_ = kDefaultLightCount;
    uint32_t visible_light_count_ = 0;
    uint32_t visible_instance_count_ = 0;
    bool floor_visible_ = true;
    uint32_t point_count_active_ = 0;
    uint32_t spot_count_active_ = 0;
    uint32_t rect_count_active_ = 0;
    uint32_t tube_count_active_ = 0;
    uint32_t point_shadow_count_ = 0;
    uint32_t spot_shadow_count_ = 0;
    bool show_light_volumes_debug_ = false;
    std::vector<LightVolumeDebugDraw> light_volume_debug_draws_{};
    bool enable_scene_occlusion_ = false;
    bool enable_light_occlusion_ = false;
    shs::LightObjectCullMode light_object_cull_mode_ = shs::LightObjectCullMode::None;
    uint32_t light_frustum_rejected_ = 0;
    uint32_t light_occlusion_rejected_ = 0;
    uint32_t light_prefilter_rejected_ = 0;
    float light_orbit_scale_ = 1.0f;
    float light_height_bias_ = 0.0f;
    float light_range_scale_ = 1.0f;
    float light_intensity_scale_ = 1.0f;
    bool enable_sun_shadow_ = false;
    float sun_shadow_strength_ = 0.0f;
    bool use_forward_plus_ = true;
    shs::LightCullingMode culling_mode_ = shs::LightCullingMode::Tiled;
    uint32_t light_tile_size_ = kDefaultTileSize;
    uint32_t cluster_z_slices_ = kDefaultClusterZSlices;
    shs::RenderPathLightGridRuntimeLayout light_grid_layout_{};
    shs::ShadowCompositionSettings shadow_settings_ = shs::make_default_shadow_composition_settings();
    VulkanCullerBackend vulkan_culler_backend_ = VulkanCullerBackend::GpuCompute;
    bool profile_depth_prepass_enabled_ = true;
    bool enable_depth_prepass_ = true;
    bool enable_light_culling_ = true;
    bool enable_scene_pass_ = true;
    bool frame_gbuffer_pass_executed_ = false;
    bool frame_ssao_pass_executed_ = false;
    bool frame_deferred_lighting_pass_executed_ = false;
    bool frame_motion_blur_pass_executed_ = false;
    bool frame_depth_of_field_pass_executed_ = false;
    bool frame_taa_pass_executed_ = false;
    bool frame_deferred_emulated_scene_pass_ = false;
    bool deferred_emulation_warning_emitted_ = false;
    FramebufferDebugPreset framebuffer_debug_preset_ = FramebufferDebugPreset::FinalComposite;
    bool semantic_debug_enabled_ = false;
    shs::PassSemantic active_semantic_debug_ = shs::PassSemantic::Unknown;
    size_t semantic_debug_index_ = 0u;
    std::vector<shs::PassSemantic> semantic_debug_targets_{};
    uint64_t cull_debug_total_refs_ = 0;
    uint32_t cull_debug_non_empty_lists_ = 0;
    uint32_t cull_debug_list_count_ = 0;
    uint32_t cull_debug_max_list_size_ = 0;
    uint32_t barrier_edge_count_ = 0u;
    uint32_t barrier_memory_edge_count_ = 0u;
    uint32_t barrier_layout_edge_count_ = 0u;
    uint32_t barrier_alias_class_count_ = 0u;
    uint32_t barrier_alias_slot_count_ = 0u;
    uint32_t frame_graph_barrier_edges_emitted_ = 0u;
    uint32_t frame_graph_barrier_fallback_count_ = 0u;
    shs::RenderPathExecutor render_path_executor_{};
    shs::PassFactoryRegistry pass_contract_registry_{};
    shs::PassFactoryRegistry pass_contract_registry_sw_{};
    shs::RenderPathPassDispatcher<FramePassExecutionContext> frame_pass_dispatcher_{};
    bool pass_dispatch_warning_emitted_ = false;
    double dispatch_total_cpu_ms_ = 0.0;
    double dispatch_slowest_pass_cpu_ms_ = 0.0;
    std::string dispatch_slowest_pass_id_{};
    std::array<VkQueryPool, kWorkerPoolRingSize> gpu_pass_query_pools_{};
    std::array<GpuPassTimestampFrameState, kWorkerPoolRingSize> gpu_pass_timestamp_frames_{};
    bool gpu_pass_timestamps_supported_ = false;
    float gpu_timestamp_period_ns_ = 0.0f;
    bool gpu_pass_timestamp_recording_active_ = false;
    uint32_t gpu_pass_timestamp_record_frame_slot_ = 0u;
    uint32_t gpu_pass_query_cursor_ = 0u;
    double gpu_pass_total_ms_ = 0.0;
    double gpu_pass_slowest_ms_ = 0.0;
    std::string gpu_pass_slowest_id_{};
    bool gpu_pass_timing_valid_ = false;
    uint32_t gpu_pass_sample_count_ = 0u;
    uint32_t gpu_pass_rejected_sample_count_ = 0u;
    std::string gpu_pass_timing_state_{"disabled"};
    PhaseFBenchmarkConfig phase_f_config_{};
    std::ofstream phase_f_metrics_stream_{};
    std::vector<size_t> phase_f_plan_indices_{};
    PhaseFBenchmarkStage phase_f_stage_ = PhaseFBenchmarkStage::Disabled;
    size_t phase_f_active_entry_slot_ = 0u;
    size_t phase_f_active_composition_index_ = 0u;
    size_t phase_f_entries_processed_ = 0u;
    bool phase_f_finished_ = false;
    uint32_t phase_f_stage_frame_counter_ = 0u;
    PhaseFBenchmarkAccumulator phase_f_accumulator_{};
    uint64_t phase_f_rebuild_target_start_ = 0u;
    uint64_t phase_f_rebuild_pipeline_start_ = 0u;
    uint64_t phase_f_swapchain_gen_start_ = 0u;
    bool phase_f_snapshot_request_armed_ = false;
    bool phase_f_snapshot_copy_submitted_ = false;
    bool phase_f_snapshot_completed_ = false;
    bool phase_f_snapshot_failed_ = false;
    std::string phase_f_snapshot_path_{};
    GpuBuffer phase_f_snapshot_readback_buffer_{};
    uint32_t phase_f_snapshot_readback_w_ = 0u;
    uint32_t phase_f_snapshot_readback_h_ = 0u;
    VkFormat phase_f_snapshot_readback_format_ = VK_FORMAT_UNDEFINED;
    PhaseGSoakConfig phase_g_config_{};
    std::ofstream phase_g_metrics_stream_{};
    PhaseGSoakState phase_g_state_{};
    PhaseIParityConfig phase_i_config_{};
    shs::RenderTechniquePreset render_technique_preset_ = shs::RenderTechniquePreset::PBR;
    shs::RenderTechniqueRecipe render_technique_recipe_ = shs::make_builtin_render_technique_recipe(
        shs::RenderTechniquePreset::PBR,
        "render_tech_vk");
    shs::RenderCompositionRecipe active_composition_recipe_ = shs::make_builtin_render_composition_recipe(
        shs::RenderPathPreset::Deferred,
        shs::RenderTechniquePreset::PBR,
        "composition_vk");
    std::vector<shs::RenderCompositionRecipe> composition_cycle_order_{};
    size_t active_composition_index_ = 0u;
    uint32_t shading_variant_ = shs::render_technique_shader_variant(shs::RenderTechniquePreset::PBR);
    float tonemap_exposure_ = 1.40f;
    float tonemap_gamma_ = 2.20f;
    shs::TechniqueMode active_technique_ = shs::TechniqueMode::Deferred;
    bool path_has_ssao_pass_ = false;
    bool path_has_taa_pass_ = false;
    bool path_has_motion_blur_pass_ = false;
    bool path_has_depth_of_field_pass_ = false;
    bool composition_ssao_enabled_ = true;
    bool composition_taa_enabled_ = true;
    bool composition_motion_blur_enabled_ = true;
    bool composition_depth_of_field_enabled_ = true;
    shs::RenderPathTemporalSettings temporal_settings_{};
    shs::RenderPathTemporalFrameState temporal_state_{};
    float technique_switch_accum_sec_ = 0.0f;
    bool auto_cycle_technique_ = false;
    bool use_multithread_recording_ = false;
    FreeCamera camera_{};
    shs::RuntimeInputLatch input_latch_{};
    std::vector<shs::RuntimeInputEvent> pending_input_events_{};
    bool relative_mouse_mode_ = false;
    bool pending_quit_action_ = false;
    std::vector<SDL_Keycode> pending_keydown_actions_{};
    shs::RuntimeState runtime_state_{};
    std::vector<shs::RuntimeAction> runtime_actions_{};
    float time_sec_ = 0.0f;
};
}

int main()
{
    try
    {
        HelloRenderingPathsApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
