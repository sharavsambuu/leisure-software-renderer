#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_elements.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Culling-д зориулсан scene element container болон
            render-scene (RenderItem) conversion helper.
*/

#if defined(SHS_HAS_JOLT) && ((SHS_HAS_JOLT + 0) == 1)

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "shs/geometry/jolt_adapter.hpp"
#include "shs/geometry/scene_shape.hpp"
#include "shs/scene/scene_types.hpp"

namespace shs
{
    struct SceneElement
    {
        SceneShape geometry{};
        MeshHandle mesh = 0;
        MaterialHandle material = 0;
        std::string name{};
        uint64_t object_id = 0;
        uint32_t user_index = 0;

        bool enabled = true;
        bool visible = true;
        bool frustum_visible = true;
        bool occluded = false;
        bool casts_shadow = true;

        uint32_t stable_id() const noexcept
        {
            return geometry.stable_id;
        }
    };

    inline RenderItem render_item_from_scene_element(const SceneElement& src)
    {
        RenderItem out{};
        out.mesh = src.mesh;
        out.mat = src.material;
        out.object_id = (src.object_id != 0u) ? src.object_id : static_cast<uint64_t>(src.geometry.stable_id);
        out.visible = src.visible;
        out.casts_shadow = src.casts_shadow;

        const glm::mat4 m_shs = jolt::to_glm(src.geometry.transform);
        out.tr.pos = glm::vec3(m_shs[3]);

        glm::vec3 axis_x = glm::vec3(m_shs[0]);
        glm::vec3 axis_y = glm::vec3(m_shs[1]);
        glm::vec3 axis_z = glm::vec3(m_shs[2]);

        glm::vec3 scale{
            glm::length(axis_x),
            glm::length(axis_y),
            glm::length(axis_z)
        };

        if (scale.x <= 1e-6f) scale.x = 1.0f;
        if (scale.y <= 1e-6f) scale.y = 1.0f;
        if (scale.z <= 1e-6f) scale.z = 1.0f;

        axis_x /= scale.x;
        axis_y /= scale.y;
        axis_z /= scale.z;

        glm::mat3 rot_m{};
        rot_m[0] = axis_x;
        rot_m[1] = axis_y;
        rot_m[2] = axis_z;
        if (glm::determinant(rot_m) < 0.0f)
        {
            scale.z = -scale.z;
            rot_m[2] = -rot_m[2];
        }

        out.tr.rot_euler = glm::eulerAngles(glm::normalize(glm::quat_cast(rot_m)));
        out.tr.scl = scale;
        return out;
    }

    class SceneElementSet
    {
    public:
        SceneElement& add(SceneElement element)
        {
            assign_ids(element);
            elements_.push_back(std::move(element));
            return elements_.back();
        }

        void clear() noexcept
        {
            elements_.clear();
        }

        void reserve(size_t count)
        {
            elements_.reserve(count);
        }

        size_t size() const noexcept
        {
            return elements_.size();
        }

        bool empty() const noexcept
        {
            return elements_.empty();
        }

        std::span<SceneElement> elements() noexcept
        {
            return std::span<SceneElement>(elements_.data(), elements_.size());
        }

        std::span<const SceneElement> elements() const noexcept
        {
            return std::span<const SceneElement>(elements_.data(), elements_.size());
        }

        SceneElement& operator[](size_t idx) noexcept
        {
            return elements_[idx];
        }

        const SceneElement& operator[](size_t idx) const noexcept
        {
            return elements_[idx];
        }

        SceneElement* find_by_object_id(uint64_t object_id) noexcept
        {
            for (auto& e : elements_)
            {
                if (e.object_id == object_id) return &e;
            }
            return nullptr;
        }

        const SceneElement* find_by_object_id(uint64_t object_id) const noexcept
        {
            for (const auto& e : elements_)
            {
                if (e.object_id == object_id) return &e;
            }
            return nullptr;
        }

        SceneElement* find_by_stable_id(uint32_t stable_id) noexcept
        {
            for (auto& e : elements_)
            {
                if (e.geometry.stable_id == stable_id) return &e;
            }
            return nullptr;
        }

        const SceneElement* find_by_stable_id(uint32_t stable_id) const noexcept
        {
            for (const auto& e : elements_)
            {
                if (e.geometry.stable_id == stable_id) return &e;
            }
            return nullptr;
        }

        void sync_visible_to_scene(Scene& scene, std::span<const uint32_t> visible_indices) const
        {
            scene.items.clear();
            scene.items.reserve(visible_indices.size());
            for (const uint32_t idx : visible_indices)
            {
                if (idx >= elements_.size()) continue;
                const SceneElement& e = elements_[idx];
                if (!e.enabled) continue;
                scene.items.push_back(render_item_from_scene_element(e));
            }
        }

        void sync_all_to_scene(Scene& scene, bool only_enabled = true) const
        {
            scene.items.clear();
            scene.items.reserve(elements_.size());
            for (const SceneElement& e : elements_)
            {
                if (only_enabled && !e.enabled) continue;
                scene.items.push_back(render_item_from_scene_element(e));
            }
        }

        std::vector<uint32_t> stable_ids() const
        {
            std::vector<uint32_t> out{};
            out.reserve(elements_.size());
            for (const SceneElement& e : elements_)
            {
                out.push_back(e.geometry.stable_id);
            }
            return out;
        }

    private:
        void assign_ids(SceneElement& element)
        {
            if (element.object_id == 0u)
            {
                element.object_id = next_object_id_++;
                if (next_object_id_ == 0u) next_object_id_ = 1u;
            }
            if (element.geometry.stable_id == 0u)
            {
                element.geometry.stable_id = next_stable_id_++;
                if (next_stable_id_ == 0u) next_stable_id_ = 1u;
            }
        }

        std::vector<SceneElement> elements_{};
        uint64_t next_object_id_ = 1u;
        uint32_t next_stable_id_ = 1u;
    };
}

#endif // SHS_HAS_JOLT
