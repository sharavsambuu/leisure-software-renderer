#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_objects.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Рендерлэх объектуудыг (SceneObject) удирдах, хадгалах, өгөгдлийн бүтцийг 
            тодорхойлох болон объект бүрийн тогтмол дугаар (stable_object_id) үүсгэх логик.

    IDENTITY POLICY (step 4.3, engine_domain_separation_migration.md):
    object_id нь тогтвортой танигч (stable identity): 0 хоосон гэсэн үг.
    Нэрээс гаргаж авсан FNV-1a id нь устгаж дахин үүсгэсэн ч мөн адил
    хадгалагдана (deletion/recreation нь identity-г хадгална). Ижил нэрээр
    олдсон тохиолдолд find() эхнийхийг (first-wins) буцаана; remove() эхний
    тааралдлыг устгана. add()-ийн буцаасан reference дараагийн add()/remove()-
    ийн дараа хүчингүй болно (vector-reference invalidation) — хадгалахыг
    хориглоно; renderer projection нь to_render_items()-ийн хуулбар ашиглана.
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
        std::vector<RenderItem> to_render_items() const
        {
            std::vector<RenderItem> out{};
            out.reserve(objects_.size());
            for (const auto& o : objects_)
            {
                RenderItem ri = make_render_item(o.mesh, o.material, o.tr.pos, o.tr.scl, o.tr.rot_euler);
                ri.object_id = o.object_id;
                ri.visible = o.visible;
                ri.casts_shadow = o.casts_shadow;
                out.push_back(ri);
            }
            return out;
        }

        SceneObject& add(SceneObject obj)
        {
            if (obj.object_id == 0)
            {
                obj.object_id = stable_object_id(obj.name);
            }
            objects_.push_back(std::move(obj));
            return objects_.back();
        }

        // Deletion policy (step 4.3): removes the FIRST object with the
        // given name. The object_id is name-derived, so re-adding an equal
        // name later recreates the SAME identity (deletion/recreation
        // preserves identity by construction). Returns false when no object
        // carries the name.
        bool remove(const std::string& name)
        {
            for (auto it = objects_.begin(); it != objects_.end(); ++it)
            {
                if (it->name == name)
                {
                    objects_.erase(it);
                    return true;
                }
            }
            return false;
        }

        // Duplicate-name policy (step 4.3): names are not unique by
        // contract; find() resolves first-wins. This detector makes
        // accidental duplicates visible to hosts that want strictness:
        // it counts repeats after the first occurrence.
        size_t count_duplicate_names() const
        {
            size_t duplicates = 0;
            for (size_t i = 0; i < objects_.size(); ++i)
            {
                for (size_t j = 0; j < i; ++j)
                {
                    if (objects_[j].name == objects_[i].name)
                    {
                        duplicates += 1;
                        break;
                    }
                }
            }
            return duplicates;
        }

        bool has_duplicate_names() const { return count_duplicate_names() != 0; }

        size_t object_count() const { return objects_.size(); }

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

    private:
        static uint64_t stable_object_id(const std::string& name)
        {
            // FNV-1a 64-bit системээр фрэйм хооронд тогтвортой байх объектын дугаар үүсгэх.
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
