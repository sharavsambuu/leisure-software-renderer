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
    struct RenderPathResolvedState
    {
        RenderPathRecipe recipe{};
        RenderPathExecutionPlan plan{};
        RenderPathResourcePlan resource_plan{};
        RenderPathBarrierPlan barrier_plan{};
        bool valid = false;
        std::size_t active_index = 0u;
    };

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

        RenderPathResolvedState resolve_index(
            std::size_t index,
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr) const
        {
            RenderPathResolvedState out{};
            if (recipe_cycle_order_.empty())
            {
                return out;
            }

            out.active_index = index % recipe_cycle_order_.size();
            const std::string& recipe_id = recipe_cycle_order_[out.active_index];
            const RenderPathRecipe* recipe = registry_.find_recipe(recipe_id);
            if (!recipe)
            {
                return out;
            }
            return resolve_recipe(*recipe, ctx, pass_registry, out.active_index);
        }

        RenderPathResolvedState resolve_recipe(
            const RenderPathRecipe& recipe,
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr,
            std::size_t active_index_hint = 0u) const
        {
            RenderPathResolvedState out{};
            out.recipe = recipe;
            out.active_index = active_index_hint;

            const RenderPathCompiler compiler{};
            out.plan = compiler.compile(out.recipe, ctx, pass_registry);
            out.resource_plan = compile_render_path_resource_plan(out.plan, out.recipe, pass_registry);
            out.barrier_plan = compile_render_path_barrier_plan(out.plan, out.resource_plan, pass_registry);
            out.valid = out.plan.valid && out.resource_plan.valid && out.barrier_plan.valid;

            if (!recipe_cycle_order_.empty())
            {
                for (std::size_t i = 0; i < recipe_cycle_order_.size(); ++i)
                {
                    const RenderPathRecipe* registered = registry_.find_recipe(recipe_cycle_order_[i]);
                    if (registered && registered->name == out.recipe.name)
                    {
                        out.active_index = i;
                        break;
                    }
                }
            }
            return out;
        }

        bool apply_resolved(const RenderPathResolvedState& state)
        {
            active_recipe_ = state.recipe;
            active_plan_ = state.plan;
            active_resource_plan_ = state.resource_plan;
            active_barrier_plan_ = state.barrier_plan;
            active_plan_valid_ = state.valid;
            active_index_ = state.active_index;
            return active_plan_valid_;
        }

        bool apply_index(
            std::size_t index,
            const Context& ctx,
            const PassFactoryRegistry* pass_registry = nullptr)
        {
            return apply_resolved(resolve_index(index, ctx, pass_registry));
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
            return apply_resolved(resolve_recipe(recipe, ctx, pass_registry));
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
