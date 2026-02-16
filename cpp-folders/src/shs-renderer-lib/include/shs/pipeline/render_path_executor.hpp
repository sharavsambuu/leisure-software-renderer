#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_executor.hpp
    MODULE: pipeline
    PURPOSE: Reusable runtime state for recipe registration, compilation, and path cycling.
*/


#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "shs/core/context.hpp"
#include "shs/frame/technique_mode.hpp"
#include "shs/pipeline/pass_registry.hpp"
#include "shs/pipeline/render_path_compiler.hpp"
#include "shs/pipeline/render_path_barrier_plan.hpp"
#include "shs/pipeline/render_path_presets.hpp"
#include "shs/pipeline/render_path_registry.hpp"
#include "shs/pipeline/render_path_resource_plan.hpp"
#include "shs/rhi/core/backend.hpp"

namespace shs
{
    class RenderPathExecutor
    {
    public:
        void clear()
        {
            registry_.clear();
            recipe_cycle_order_.clear();
            active_recipe_ = RenderPathRecipe{};
            active_plan_ = RenderPathExecutionPlan{};
            active_resource_plan_ = RenderPathResourcePlan{};
            active_barrier_plan_ = RenderPathBarrierPlan{};
            active_plan_valid_ = false;
            active_index_ = 0u;
        }

        bool register_builtin_presets(RenderBackendType backend, std::string_view name_prefix = "path")
        {
            clear();
            return register_builtin_render_path_presets(registry_, backend, &recipe_cycle_order_, name_prefix);
        }

        bool has_recipes() const
        {
            return !recipe_cycle_order_.empty();
        }

        std::size_t recipe_count() const
        {
            return recipe_cycle_order_.size();
        }

        std::size_t active_index() const
        {
            return active_index_;
        }

        const std::vector<std::string>& recipe_cycle_order() const
        {
            return recipe_cycle_order_;
        }

        const RenderPathRecipe& active_recipe() const
        {
            return active_recipe_;
        }

        const RenderPathExecutionPlan& active_plan() const
        {
            return active_plan_;
        }

        bool active_plan_valid() const
        {
            return active_plan_valid_;
        }

        const RenderPathResourcePlan& active_resource_plan() const
        {
            return active_resource_plan_;
        }

        const RenderPathBarrierPlan& active_barrier_plan() const
        {
            return active_barrier_plan_;
        }

        std::size_t find_recipe_index_by_mode(TechniqueMode mode) const
        {
            for (std::size_t i = 0; i < recipe_cycle_order_.size(); ++i)
            {
                const RenderPathRecipe* recipe = registry_.find_recipe(recipe_cycle_order_[i]);
                if (recipe && recipe->technique_mode == mode)
                {
                    return i;
                }
            }
            return 0u;
        }

        bool apply_index(
            std::size_t index,
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr)
        {
            if (recipe_cycle_order_.empty())
            {
                active_recipe_ = RenderPathRecipe{};
                active_plan_ = RenderPathExecutionPlan{};
                active_resource_plan_ = RenderPathResourcePlan{};
                active_barrier_plan_ = RenderPathBarrierPlan{};
                active_plan_valid_ = false;
                active_index_ = 0u;
                return false;
            }

            active_index_ = index % recipe_cycle_order_.size();
            const std::string& recipe_id = recipe_cycle_order_[active_index_];
            const RenderPathRecipe* recipe = registry_.find_recipe(recipe_id);
            if (!recipe)
            {
                active_recipe_ = RenderPathRecipe{};
                active_plan_ = RenderPathExecutionPlan{};
                active_resource_plan_ = RenderPathResourcePlan{};
                active_barrier_plan_ = RenderPathBarrierPlan{};
                active_plan_valid_ = false;
                return false;
            }

            active_recipe_ = *recipe;
            const RenderPathCompiler compiler{};
            active_plan_ = compiler.compile(active_recipe_, ctx, pass_registry);
            active_resource_plan_ = compile_render_path_resource_plan(active_plan_, active_recipe_, pass_registry);
            active_barrier_plan_ = compile_render_path_barrier_plan(active_plan_, active_resource_plan_, pass_registry);
            active_plan_valid_ = active_plan_.valid && active_resource_plan_.valid && active_barrier_plan_.valid;
            return active_plan_valid_;
        }

        bool cycle_next(
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr)
        {
            if (recipe_cycle_order_.empty()) return false;
            return apply_index(active_index_ + 1u, ctx, pass_registry);
        }

        bool apply_recipe(
            const RenderPathRecipe& recipe,
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr)
        {
            active_recipe_ = recipe;
            const RenderPathCompiler compiler{};
            active_plan_ = compiler.compile(active_recipe_, ctx, pass_registry);
            active_resource_plan_ = compile_render_path_resource_plan(active_plan_, active_recipe_, pass_registry);
            active_barrier_plan_ = compile_render_path_barrier_plan(active_plan_, active_resource_plan_, pass_registry);
            active_plan_valid_ = active_plan_.valid && active_resource_plan_.valid && active_barrier_plan_.valid;

            active_index_ = 0u;
            for (std::size_t i = 0; i < recipe_cycle_order_.size(); ++i)
            {
                const RenderPathRecipe* registered = registry_.find_recipe(recipe_cycle_order_[i]);
                if (registered && registered->name == active_recipe_.name)
                {
                    active_index_ = i;
                    break;
                }
            }

            return active_plan_valid_;
        }

    private:
        RenderPathRegistry registry_{};
        std::vector<std::string> recipe_cycle_order_{};
        RenderPathRecipe active_recipe_{};
        RenderPathExecutionPlan active_plan_{};
        RenderPathResourcePlan active_resource_plan_{};
        RenderPathBarrierPlan active_barrier_plan_{};
        bool active_plan_valid_ = false;
        std::size_t active_index_ = 0u;
    };
}
