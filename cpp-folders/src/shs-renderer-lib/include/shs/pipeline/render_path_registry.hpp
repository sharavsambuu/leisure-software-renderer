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

#include "shs/pipeline/render_path_recipe.hpp"

namespace shs
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

        void register_default_recipes()
        {
            (void)register_recipe(make_default_soft_shadow_culling_recipe(RenderBackendType::Vulkan));
            (void)register_recipe(make_default_soft_shadow_culling_recipe(RenderBackendType::Software));
        }

    private:
        std::unordered_map<std::string, RenderPathRecipe> recipes_{};
    };
}
