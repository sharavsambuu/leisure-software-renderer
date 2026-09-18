#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_registry.hpp
    MODULE: pipeline
    PURPOSE: Named recipe registry for dynamic render path composition.
*/


#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shs/renderpath/planning/render_path_recipe.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace renderpath
    {
    class RenderPathRegistry
    {
    public:
        bool register_recipe(RenderPathRecipe recipe)
        {
            if (recipe.name.empty()) return false;
            recipes_[recipe.name] = std::move(recipe);
            return true;
        }

        bool has_recipe(std::string_view recipe_name) const
        {
            return recipes_.find(std::string(recipe_name)) != recipes_.end();
        }

        const RenderPathRecipe* find_recipe(std::string_view recipe_name) const
        {
            auto it = recipes_.find(std::string(recipe_name));
            if (it == recipes_.end()) return nullptr;
            return &it->second;
        }

        std::vector<std::string> recipe_ids() const
        {
            std::vector<std::string> out{};
            out.reserve(recipes_.size());
            for (const auto& kv : recipes_)
            {
                out.push_back(kv.first);
            }
            return out;
        }

        void clear()
        {
            recipes_.clear();
        }

        // RP-1 (graduation req 4): ONE recipe, not one per substrate. This used
        // to register two backend-forked recipes
        // (`soft_shadow_culling_vk_default` / `soft_shadow_culling_sw_default`);
        // the substrate a pass runs on is now a resolution OUTPUT, so a host that
        // advertises its substrate set gets Vulkan, OpenGL or the host rasterizer
        // out of this single registry entry. Registered under the unified name
        // `soft_shadow_culling`; nothing looked the old two names up.
        void register_default_recipes()
        {
            (void)register_recipe(make_soft_shadow_culling_recipe(SubstratePolicy::DevicePreferred));
        }

    private:
        std::unordered_map<std::string, RenderPathRecipe> recipes_{};
    };

    } // inline namespace renderpath
}
