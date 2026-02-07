#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_objects.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн scene модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>
#include <vector>
#include <cstdint>

#include "shs/scene/scene_bindings.hpp"

namespace shs
{
    struct SceneObject
    {
        std::string name{};
        MeshHandle mesh = 0;
        MaterialHandle material = 0;

        Transform tr{};
        bool visible = true;
        bool casts_shadow = true;
        uint64_t object_id = 0;
    };

    class SceneObjectSet
    {
    public:
        SceneObject& add(SceneObject obj)
        {
            if (obj.object_id == 0)
            {
                obj.object_id = stable_object_id(obj.name);
            }
            objects_.push_back(std::move(obj));
            return objects_.back();
        }

        SceneObject* find(const std::string& name)
        {
            for (auto& o : objects_)
            {
                if (o.name == name) return &o;
            }
            return nullptr;
        }

        const SceneObject* find(const std::string& name) const
        {
            for (const auto& o : objects_)
            {
                if (o.name == name) return &o;
            }
            return nullptr;
        }

        void sync_to_scene(Scene& scene) const
        {
            scene.items.clear();
            scene.items.reserve(objects_.size());

            for (const auto& o : objects_)
            {
                RenderItem ri = make_render_item(o.mesh, o.material, o.tr.pos, o.tr.scl, o.tr.rot_euler);
                ri.object_id = o.object_id;
                ri.visible = o.visible;
                ri.casts_shadow = o.casts_shadow;
                scene.items.push_back(ri);
            }
        }

    private:
        static uint64_t stable_object_id(const std::string& name)
        {
            // FNV-1a 64-bit hash for stable cross-frame object keys.
            uint64_t h = 1469598103934665603ull;
            for (unsigned char c : name)
            {
                h ^= (uint64_t)c;
                h *= 1099511628211ull;
            }
            if (h == 0) h = 1;
            return h;
        }

        std::vector<SceneObject> objects_{};
    };
}
