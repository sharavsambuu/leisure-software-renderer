#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shs/camera/camera_math.hpp"
#include "shs/geometry/jolt_shapes.hpp"
#include "shs/geometry/scene_shape.hpp"
#include "shs/lighting/light_types.hpp"
#include "shs/scene/scene_elements.hpp"

namespace shs
{
    constexpr uint32_t kLightSelectionCapacity = 8u;

    enum class LightObjectCullMode : uint8_t
    {
        None = 0,
        SphereAabb = 1,
        VolumeAabb = 2
    };

    inline const char* light_object_cull_mode_name(LightObjectCullMode mode)
    {
        switch (mode)
        {
            case LightObjectCullMode::None: return "None";
            case LightObjectCullMode::SphereAabb: return "Sphere";
            case LightObjectCullMode::VolumeAabb: return "Volume";
        }
        return "Unknown";
    }

    inline LightObjectCullMode next_light_object_cull_mode(LightObjectCullMode mode)
    {
        switch (mode)
        {
            case LightObjectCullMode::None: return LightObjectCullMode::SphereAabb;
            case LightObjectCullMode::SphereAabb: return LightObjectCullMode::VolumeAabb;
            case LightObjectCullMode::VolumeAabb: return LightObjectCullMode::None;
        }
        return LightObjectCullMode::None;
    }

    struct LightProperties
    {
        glm::vec3 color{1.0f};
        float intensity = 1.0f;
        glm::vec3 position_ws{0.0f};
        float range = 8.0f;
        glm::vec3 direction_ws{0.0f, -1.0f, 0.0f};
        float inner_angle_rad = glm::radians(16.0f);
        float outer_angle_rad = glm::radians(28.0f);
        glm::vec3 right_ws{1.0f, 0.0f, 0.0f};
        glm::vec3 up_ws{0.0f, 1.0f, 0.0f};
        glm::vec2 rect_half_extents{0.8f, 0.5f};
        float tube_half_length = 1.0f;
        float tube_radius = 0.25f;
        LightAttenuationModel attenuation_model = LightAttenuationModel::Smooth;
        float attenuation_power = 1.0f;
        float attenuation_bias = 0.05f;
        float attenuation_cutoff = 0.0f;
        uint32_t flags = LightFlagsDefault;
    };

    struct LightMotionProfile
    {
        glm::vec3 orbit_center{0.0f};
        glm::vec3 orbit_axis{0.0f, 1.0f, 0.0f};
        glm::vec3 radial_axis{1.0f, 0.0f, 0.0f};
        glm::vec3 aim_center{0.0f};
        float orbit_radius = 8.0f;
        float orbit_speed = 0.5f;
        float orbit_phase = 0.0f;
        float vertical_amplitude = 1.0f;
        float vertical_speed = 1.3f;
        float direction_lead = 0.3f;
        float vertical_aim_bias = -0.1f;
    };

    struct LightContribution
    {
        glm::vec3 diffuse{0.0f};
        glm::vec3 specular{0.0f};
    };

    class ILightModel
    {
    public:
        virtual ~ILightModel() = default;

        virtual LightType type() const noexcept = 0;
        virtual const char* name() const noexcept = 0;
        virtual JPH::ShapeRefC create_volume_shape(const LightProperties& props) const = 0;
        virtual CullingLightGPU pack_for_culling(const LightProperties& props) const = 0;
        virtual glm::mat4 volume_model_matrix(const LightProperties& props) const = 0;
        virtual LightContribution sample(
            const LightProperties& props,
            const glm::vec3& world_pos,
            const glm::vec3& world_normal,
            const glm::vec3& view_dir_ws) const = 0;
    };

    struct LightInstance
    {
        const ILightModel* model = nullptr;
        LightProperties props{};
        LightMotionProfile motion{};
        SceneShape volume{};
        uint32_t mesh_index = 0;
        glm::mat4 volume_model{1.0f};
        CullingLightGPU packed{};
        bool visible = true;
        bool frustum_visible = true;
        bool occluded = false;
    };

    struct LightSelection
    {
        std::array<uint32_t, kLightSelectionCapacity> indices{};
        std::array<float, kLightSelectionCapacity> dist2{};
        uint32_t count = 0;
    };

    inline glm::vec3 safe_forward(const LightProperties& props)
    {
        return normalize_or(props.direction_ws, glm::vec3(0.0f, -1.0f, 0.0f));
    }

    inline void basis_from_forward_and_hint(
        const glm::vec3& forward,
        const glm::vec3& up_hint,
        glm::vec3& out_right,
        glm::vec3& out_up,
        glm::vec3& out_forward)
    {
        out_forward = normalize_or(forward, glm::vec3(0.0f, 0.0f, 1.0f));
        const glm::vec3 up_ref = normalize_or(up_hint, glm::vec3(0.0f, 1.0f, 0.0f));
        out_right = glm::cross(up_ref, out_forward);
        out_right = normalize_or(out_right, right_from_forward(out_forward, up_ref));
        out_up = normalize_or(glm::cross(out_forward, out_right), glm::vec3(0.0f, 1.0f, 0.0f));
        out_right = normalize_or(glm::cross(out_up, out_forward), out_right);
    }

    inline glm::mat4 model_from_basis(
        const glm::vec3& position,
        const glm::vec3& axis_x,
        const glm::vec3& axis_y,
        const glm::vec3& axis_z)
    {
        glm::mat4 model(1.0f);
        model[0] = glm::vec4(axis_x, 0.0f);
        model[1] = glm::vec4(axis_y, 0.0f);
        model[2] = glm::vec4(axis_z, 0.0f);
        model[3] = glm::vec4(position, 1.0f);
        return model;
    }

    inline LocalLightCommon make_light_common(const LightProperties& props)
    {
        LocalLightCommon common{};
        common.position_ws = props.position_ws;
        common.range = std::max(props.range, 0.001f);
        common.color = glm::max(props.color, glm::vec3(0.0f));
        common.intensity = std::max(props.intensity, 0.0f);
        common.flags = props.flags;
        common.attenuation_model = props.attenuation_model;
        common.attenuation_power = std::max(props.attenuation_power, 0.001f);
        common.attenuation_bias = std::max(props.attenuation_bias, 1e-5f);
        common.attenuation_cutoff = std::max(props.attenuation_cutoff, 0.0f);
        return common;
    }

    inline float eval_distance_attenuation(const LightProperties& props, float distance)
    {
        const float range = std::max(props.range, 0.001f);
        if (distance >= range) return 0.0f;

        const float norm = std::clamp(1.0f - distance / range, 0.0f, 1.0f);
        float falloff = 0.0f;
        switch (props.attenuation_model)
        {
            case LightAttenuationModel::Linear:
                falloff = norm;
                break;
            case LightAttenuationModel::Smooth:
                falloff = norm * norm * (3.0f - 2.0f * norm);
                break;
            case LightAttenuationModel::InverseSquare:
            {
                const float denom = std::max(distance * distance, props.attenuation_bias);
                const float inv = 1.0f / denom;
                const float range_norm = range * range;
                falloff = std::min(1.0f, inv * range_norm) * (norm * norm);
                break;
            }
        }

        falloff = std::pow(std::max(falloff, 0.0f), std::max(props.attenuation_power, 0.001f));
        if (props.attenuation_cutoff > 0.0f && falloff < props.attenuation_cutoff) return 0.0f;
        return std::max(falloff, 0.0f);
    }

    inline LightContribution eval_local_light_brdf(
        const LightProperties& props,
        const glm::vec3& L,
        float distance,
        float shaping,
        float spec_power,
        float spec_scale,
        const glm::vec3& world_normal,
        const glm::vec3& view_dir_ws)
    {
        LightContribution out{};
        const float ndotl = std::max(glm::dot(world_normal, L), 0.0f);
        if (ndotl <= 0.0f) return out;

        const float attenuation = eval_distance_attenuation(props, distance) * std::max(shaping, 0.0f);
        if (attenuation <= 0.0f) return out;

        const glm::vec3 radiance = glm::max(props.color, glm::vec3(0.0f)) * std::max(props.intensity, 0.0f) * attenuation;
        const glm::vec3 H = normalize_or(L + view_dir_ws, L);
        const float ndoth = std::max(glm::dot(world_normal, H), 0.0f);
        const float spec = (ndotl > 0.0f) ? (spec_scale * std::pow(ndoth, spec_power)) : 0.0f;

        out.diffuse = radiance * ndotl;
        out.specular = radiance * spec;
        return out;
    }

    inline bool intersect_aabb_aabb(const AABB& a, const AABB& b)
    {
        if (a.maxv.x < b.minv.x || a.minv.x > b.maxv.x) return false;
        if (a.maxv.y < b.minv.y || a.minv.y > b.maxv.y) return false;
        if (a.maxv.z < b.minv.z || a.minv.z > b.maxv.z) return false;
        return true;
    }

    inline bool intersect_sphere_aabb(const Sphere& sphere, const AABB& box)
    {
        const glm::vec3 closest = glm::clamp(sphere.center, box.minv, box.maxv);
        const glm::vec3 d = sphere.center - closest;
        return glm::dot(d, d) <= sphere.radius * sphere.radius;
    }

    inline glm::vec3 closest_point_on_segment(const glm::vec3& p, const glm::vec3& a, const glm::vec3& b)
    {
        const glm::vec3 ab = b - a;
        const float denom = glm::dot(ab, ab);
        if (denom <= 1e-8f) return a;
        const float t = std::clamp(glm::dot(p - a, ab) / denom, 0.0f, 1.0f);
        return a + ab * t;
    }

    inline void add_light_candidate(LightSelection& selection, uint32_t light_idx, float dist2)
    {
        if (selection.count < kLightSelectionCapacity)
        {
            selection.indices[selection.count] = light_idx;
            selection.dist2[selection.count] = dist2;
            ++selection.count;
            return;
        }

        uint32_t farthest = 0;
        float farthest_d2 = selection.dist2[0];
        for (uint32_t i = 1; i < kLightSelectionCapacity; ++i)
        {
            if (selection.dist2[i] > farthest_d2)
            {
                farthest = i;
                farthest_d2 = selection.dist2[i];
            }
        }

        if (dist2 < farthest_d2)
        {
            selection.indices[farthest] = light_idx;
            selection.dist2[farthest] = dist2;
        }
    }

    class PointLightModel final : public ILightModel
    {
    public:
        LightType type() const noexcept override
        {
            return LightType::Point;
        }

        const char* name() const noexcept override
        {
            return "Point";
        }

        JPH::ShapeRefC create_volume_shape(const LightProperties& props) const override
        {
            return jolt::make_point_light_volume(std::max(props.range, 0.1f));
        }

        CullingLightGPU pack_for_culling(const LightProperties& props) const override
        {
            PointLight point{};
            point.common = make_light_common(props);
            return make_point_culling_light(point);
        }

        glm::mat4 volume_model_matrix(const LightProperties& props) const override
        {
            return glm::translate(glm::mat4(1.0f), props.position_ws);
        }

        LightContribution sample(
            const LightProperties& props,
            const glm::vec3& world_pos,
            const glm::vec3& world_normal,
            const glm::vec3& view_dir_ws) const override
        {
            const glm::vec3 to_light = props.position_ws - world_pos;
            const float dist = glm::length(to_light);
            if (dist <= 1e-4f || dist > props.range) return {};

            const glm::vec3 L = to_light / dist;
            return eval_local_light_brdf(props, L, dist, 1.0f, 36.0f, 0.30f, world_normal, view_dir_ws);
        }
    };

    class SpotLightModel final : public ILightModel
    {
    public:
        LightType type() const noexcept override
        {
            return LightType::Spot;
        }

        const char* name() const noexcept override
        {
            return "Spot";
        }

        JPH::ShapeRefC create_volume_shape(const LightProperties& props) const override
        {
            return jolt::make_spot_light_volume(
                std::max(props.range, 0.1f),
                std::clamp(props.outer_angle_rad, 0.05f, glm::half_pi<float>() - 0.01f),
                20);
        }

        CullingLightGPU pack_for_culling(const LightProperties& props) const override
        {
            SpotLight spot{};
            spot.common = make_light_common(props);
            spot.direction_ws = safe_forward(props);
            const float inner = std::clamp(props.inner_angle_rad, 0.02f, glm::half_pi<float>() - 0.02f);
            const float outer = std::clamp(std::max(inner + 0.005f, props.outer_angle_rad), inner + 0.005f, glm::half_pi<float>() - 0.005f);
            spot.inner_angle_rad = inner;
            spot.outer_angle_rad = outer;
            return make_spot_culling_light(spot);
        }

        glm::mat4 volume_model_matrix(const LightProperties& props) const override
        {
            glm::vec3 right{}, up{}, fwd{};
            basis_from_forward_and_hint(safe_forward(props), props.up_ws, right, up, fwd);
            return model_from_basis(props.position_ws, right, up, fwd);
        }

        LightContribution sample(
            const LightProperties& props,
            const glm::vec3& world_pos,
            const glm::vec3& world_normal,
            const glm::vec3& view_dir_ws) const override
        {
            const glm::vec3 to_light = props.position_ws - world_pos;
            const float dist = glm::length(to_light);
            if (dist <= 1e-4f || dist > props.range) return {};

            const glm::vec3 L = to_light / dist;
            const glm::vec3 light_to_surface = -L;
            const glm::vec3 dir = safe_forward(props);

            const float inner = std::clamp(props.inner_angle_rad, 0.02f, glm::half_pi<float>() - 0.02f);
            const float outer = std::clamp(std::max(inner + 0.005f, props.outer_angle_rad), inner + 0.005f, glm::half_pi<float>() - 0.005f);
            const float cos_inner = std::cos(inner);
            const float cos_outer = std::cos(outer);
            const float cos_theta = glm::dot(light_to_surface, dir);
            if (cos_theta <= cos_outer) return {};

            float t = (cos_theta - cos_outer) / std::max(cos_inner - cos_outer, 1e-5f);
            t = std::clamp(t, 0.0f, 1.0f);
            const float shaping = t * t * (3.0f - 2.0f * t);

            return eval_local_light_brdf(props, L, dist, shaping, 34.0f, 0.32f, world_normal, view_dir_ws);
        }
    };

    class RectAreaLightModel final : public ILightModel
    {
    public:
        LightType type() const noexcept override
        {
            return LightType::RectArea;
        }

        const char* name() const noexcept override
        {
            return "Rect";
        }

        JPH::ShapeRefC create_volume_shape(const LightProperties& props) const override
        {
            return jolt::make_rect_area_light_volume(
                glm::max(props.rect_half_extents, glm::vec2(0.1f)),
                std::max(props.range, 0.1f));
        }

        CullingLightGPU pack_for_culling(const LightProperties& props) const override
        {
            glm::vec3 right{}, up{}, fwd{};
            basis_from_forward_and_hint(safe_forward(props), props.up_ws, right, up, fwd);

            RectAreaLight rect{};
            rect.common = make_light_common(props);
            rect.direction_ws = fwd;
            rect.right_ws = right;
            rect.half_extents = glm::max(props.rect_half_extents, glm::vec2(0.1f));
            return make_rect_area_culling_light(rect);
        }

        glm::mat4 volume_model_matrix(const LightProperties& props) const override
        {
            glm::vec3 right{}, up{}, fwd{};
            basis_from_forward_and_hint(safe_forward(props), props.up_ws, right, up, fwd);
            return model_from_basis(props.position_ws, right, up, fwd);
        }

        LightContribution sample(
            const LightProperties& props,
            const glm::vec3& world_pos,
            const glm::vec3& world_normal,
            const glm::vec3& view_dir_ws) const override
        {
            glm::vec3 right{}, up{}, fwd{};
            basis_from_forward_and_hint(safe_forward(props), props.up_ws, right, up, fwd);

            const glm::vec2 half_ext = glm::max(props.rect_half_extents, glm::vec2(0.05f));
            const glm::vec3 d = world_pos - props.position_ws;
            const float ux = std::clamp(glm::dot(d, right), -half_ext.x, half_ext.x);
            const float uy = std::clamp(glm::dot(d, up), -half_ext.y, half_ext.y);
            const glm::vec3 emit_pt = props.position_ws + right * ux + up * uy;

            const glm::vec3 to_light = emit_pt - world_pos;
            const float dist = glm::length(to_light);
            if (dist <= 1e-4f || dist > props.range) return {};

            const glm::vec3 L = to_light / dist;
            const glm::vec3 light_to_surface = -L;
            const float emission_facing = std::max(glm::dot(fwd, light_to_surface), 0.0f);
            if (emission_facing <= 0.0f) return {};

            const float shape_gain = 0.65f + 0.55f * emission_facing;
            return eval_local_light_brdf(props, L, dist, shape_gain, 26.0f, 0.26f, world_normal, view_dir_ws);
        }
    };

    class TubeAreaLightModel final : public ILightModel
    {
    public:
        LightType type() const noexcept override
        {
            return LightType::TubeArea;
        }

        const char* name() const noexcept override
        {
            return "Tube";
        }

        JPH::ShapeRefC create_volume_shape(const LightProperties& props) const override
        {
            return jolt::make_tube_area_light_volume(
                std::max(props.tube_half_length, 0.1f),
                std::max(props.tube_radius, 0.05f));
        }

        CullingLightGPU pack_for_culling(const LightProperties& props) const override
        {
            TubeAreaLight tube{};
            tube.common = make_light_common(props);
            tube.axis_ws = normalize_or(props.right_ws, glm::vec3(1.0f, 0.0f, 0.0f));
            tube.half_length = std::max(props.tube_half_length, 0.1f);
            tube.radius = std::max(props.tube_radius, 0.05f);
            return make_tube_area_culling_light(tube);
        }

        glm::mat4 volume_model_matrix(const LightProperties& props) const override
        {
            const glm::vec3 axis_y = normalize_or(props.right_ws, glm::vec3(1.0f, 0.0f, 0.0f));
            glm::vec3 axis_z = props.direction_ws - axis_y * glm::dot(props.direction_ws, axis_y);
            axis_z = normalize_or(axis_z, glm::vec3(0.0f, 0.0f, 1.0f));
            glm::vec3 axis_x = normalize_or(glm::cross(axis_y, axis_z), glm::vec3(1.0f, 0.0f, 0.0f));
            axis_z = normalize_or(glm::cross(axis_x, axis_y), axis_z);
            return model_from_basis(props.position_ws, axis_x, axis_y, axis_z);
        }

        LightContribution sample(
            const LightProperties& props,
            const glm::vec3& world_pos,
            const glm::vec3& world_normal,
            const glm::vec3& view_dir_ws) const override
        {
            const glm::vec3 axis = normalize_or(props.right_ws, glm::vec3(1.0f, 0.0f, 0.0f));
            const float half_len = std::max(props.tube_half_length, 0.1f);
            const glm::vec3 a = props.position_ws - axis * half_len;
            const glm::vec3 b = props.position_ws + axis * half_len;

            const glm::vec3 emit_pt = closest_point_on_segment(world_pos, a, b);
            const glm::vec3 to_light = emit_pt - world_pos;
            const float dist = glm::length(to_light);
            if (dist <= 1e-4f || dist > props.range) return {};

            const glm::vec3 L = to_light / dist;
            const float radial_softening = std::clamp(1.0f - dist / std::max(props.range, 0.1f), 0.0f, 1.0f);
            const float shaping = 0.75f + 0.35f * radial_softening;
            return eval_local_light_brdf(props, L, dist, shaping, 22.0f, 0.20f, world_normal, view_dir_ws);
        }
    };

    inline void update_light_motion(LightInstance& light, float time_s)
    {
        const LightMotionProfile& motion = light.motion;
        const glm::vec3 orbit_axis = normalize_or(motion.orbit_axis, glm::vec3(0.0f, 1.0f, 0.0f));

        glm::vec3 radial = motion.radial_axis - orbit_axis * glm::dot(motion.radial_axis, orbit_axis);
        radial = normalize_or(radial, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 tangent = normalize_or(glm::cross(orbit_axis, radial), glm::vec3(0.0f, 0.0f, 1.0f));

        const float theta = motion.orbit_phase + motion.orbit_speed * time_s;
        const glm::vec3 orbit_offset =
            radial * std::cos(theta) * motion.orbit_radius +
            tangent * std::sin(theta) * motion.orbit_radius;
        const float bob = motion.vertical_amplitude * std::sin(motion.vertical_speed * time_s + motion.orbit_phase * 1.37f);

        light.props.position_ws = motion.orbit_center + orbit_offset + orbit_axis * bob;

        const glm::vec3 travel = normalize_or(
            -radial * std::sin(theta) + tangent * std::cos(theta),
            tangent);
        const glm::vec3 to_target = normalize_or(motion.aim_center - light.props.position_ws, -orbit_offset);
        const glm::vec3 direction = normalize_or(
            to_target + travel * motion.direction_lead + orbit_axis * motion.vertical_aim_bias,
            to_target);

        glm::vec3 right = normalize_or(travel, glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 up = normalize_or(glm::cross(right, direction), orbit_axis);
        right = normalize_or(glm::cross(direction, up), right);

        light.props.direction_ws = direction;
        light.props.right_ws = right;
        light.props.up_ws = up;
    }

    inline bool light_affects_object(
        const LightInstance& light,
        const AABB& object_aabb,
        LightObjectCullMode mode)
    {
        if (mode == LightObjectCullMode::None) return true;

        if (mode == LightObjectCullMode::SphereAabb)
        {
            Sphere sphere{};
            sphere.center = glm::vec3(light.packed.cull_sphere);
            sphere.radius = std::max(light.packed.cull_sphere.w, 0.0f);
            return intersect_sphere_aabb(sphere, object_aabb);
        }

        AABB light_aabb{};
        light_aabb.minv = glm::vec3(light.packed.cull_aabb_min);
        light_aabb.maxv = glm::vec3(light.packed.cull_aabb_max);
        return intersect_aabb_aabb(light_aabb, object_aabb);
    }

    inline LightSelection collect_object_lights(
        const AABB& object_aabb,
        std::span<const uint32_t> visible_light_scene_indices,
        const SceneElementSet& light_scene,
        const std::vector<LightInstance>& lights,
        LightObjectCullMode cull_mode)
    {
        LightSelection out{};
        const glm::vec3 center = object_aabb.center();

        for (const uint32_t scene_idx : visible_light_scene_indices)
        {
            if (scene_idx >= light_scene.size()) continue;
            const uint32_t light_idx = light_scene[scene_idx].user_index;
            if (light_idx >= lights.size()) continue;
            const LightInstance& light = lights[light_idx];
            if (!light_affects_object(light, object_aabb, cull_mode)) continue;

            const glm::vec3 d = light.props.position_ws - center;
            const float dist2 = glm::dot(d, d);
            add_light_candidate(out, light_idx, dist2);
        }

        return out;
    }

    inline LightSelection collect_object_lights(
        const AABB& object_aabb,
        const std::vector<uint32_t>& visible_light_scene_indices,
        const SceneElementSet& light_scene,
        const std::vector<LightInstance>& lights,
        LightObjectCullMode cull_mode)
    {
        return collect_object_lights(
            object_aabb,
            std::span<const uint32_t>(visible_light_scene_indices),
            light_scene,
            lights,
            cull_mode);
    }
}
