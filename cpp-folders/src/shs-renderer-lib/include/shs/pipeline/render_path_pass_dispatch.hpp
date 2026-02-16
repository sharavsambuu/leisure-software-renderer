#pragma once

/*
    SHS RENDERER SAN

    FILE: render_path_pass_dispatch.hpp
    MODULE: pipeline
    PURPOSE: Shared pass-chain dispatcher for render-path execution plans.
*/


#include <functional>
#include <chrono>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "shs/pipeline/pass_id.hpp"
#include "shs/pipeline/render_path_compiler.hpp"

namespace shs
{
    struct RenderPathPassDispatchSample
    {
        std::string id{};
        PassId pass_id = PassId::Unknown;
        bool required = false;
        bool handler_found = false;
        bool success = false;
        double cpu_ms = 0.0;
    };

    struct RenderPathPassDispatchResult
    {
        bool ok = true;
        std::size_t executed_count = 0u;
        std::size_t skipped_optional_count = 0u;
        double total_cpu_ms = 0.0;
        double slowest_cpu_ms = 0.0;
        std::string slowest_pass_id{};
        PassId slowest_pass = PassId::Unknown;
        std::vector<RenderPathPassDispatchSample> samples{};
        std::vector<std::string> warnings{};
        std::vector<std::string> errors{};
    };

    template <typename TContext>
    class RenderPathPassDispatcher
    {
    public:
        using Handler = std::function<bool(TContext&, const RenderPathCompiledPass&)>;

        void clear()
        {
            typed_handlers_.clear();
            custom_handlers_.clear();
        }

        bool register_handler(std::string id, Handler handler)
        {
            if (id.empty() || !handler) return false;
            const PassId pid = parse_pass_id(id);
            if (pass_id_is_standard(pid))
            {
                typed_handlers_[pass_id_key(pid)] = std::move(handler);
            }
            else
            {
                custom_handlers_[std::move(id)] = std::move(handler);
            }
            return true;
        }

        bool register_handler(PassId pass_id, Handler handler)
        {
            if (!pass_id_is_standard(pass_id) || !handler) return false;
            typed_handlers_[pass_id_key(pass_id)] = std::move(handler);
            return true;
        }

        bool has_handler(std::string_view id) const
        {
            const PassId pid = parse_pass_id(id);
            if (pass_id_is_standard(pid))
            {
                return typed_handlers_.find(pass_id_key(pid)) != typed_handlers_.end();
            }
            return custom_handlers_.find(std::string(id)) != custom_handlers_.end();
        }

        bool has_handler(PassId pass_id) const
        {
            if (!pass_id_is_standard(pass_id)) return false;
            return typed_handlers_.find(pass_id_key(pass_id)) != typed_handlers_.end();
        }

        bool execute(
            const RenderPathExecutionPlan& plan,
            TContext& context,
            RenderPathPassDispatchResult* out_result = nullptr) const
        {
            RenderPathPassDispatchResult local{};
            RenderPathPassDispatchResult& result = out_result ? *out_result : local;
            result = RenderPathPassDispatchResult{};

            for (const RenderPathCompiledPass& pass : plan.pass_chain)
            {
                const PassId pass_id =
                    pass_id_is_standard(pass.pass_id) ? pass.pass_id : parse_pass_id(pass.id);
                const Handler* handler = nullptr;

                if (pass_id_is_standard(pass_id))
                {
                    auto it = typed_handlers_.find(pass_id_key(pass_id));
                    if (it != typed_handlers_.end()) handler = &it->second;
                }
                else
                {
                    auto it = custom_handlers_.find(pass.id);
                    if (it != custom_handlers_.end()) handler = &it->second;
                }

                if (!handler)
                {
                    RenderPathPassDispatchSample sample{};
                    sample.id = pass.id;
                    sample.pass_id = pass_id;
                    sample.required = pass.required;
                    sample.handler_found = false;
                    sample.success = false;
                    sample.cpu_ms = 0.0;
                    result.samples.push_back(std::move(sample));

                    if (pass.required)
                    {
                        result.errors.push_back(
                            "No pass handler registered for required pass '" + pass.id + "'.");
                        result.ok = false;
                    }
                    else
                    {
                        result.warnings.push_back(
                            "Skipping optional pass '" + pass.id + "' because no handler is registered.");
                        result.skipped_optional_count++;
                    }
                    continue;
                }

                const auto pass_begin = std::chrono::steady_clock::now();
                const bool handled = (*handler)(context, pass);
                const auto pass_end = std::chrono::steady_clock::now();
                const double pass_cpu_ms =
                    std::chrono::duration<double, std::milli>(pass_end - pass_begin).count();
                result.total_cpu_ms += pass_cpu_ms;

                RenderPathPassDispatchSample sample{};
                sample.id = pass.id;
                sample.pass_id = pass_id;
                sample.required = pass.required;
                sample.handler_found = true;
                sample.success = handled;
                sample.cpu_ms = pass_cpu_ms;
                result.samples.push_back(std::move(sample));
                if (pass_cpu_ms >= result.slowest_cpu_ms)
                {
                    result.slowest_cpu_ms = pass_cpu_ms;
                    result.slowest_pass_id = pass.id;
                    result.slowest_pass = pass_id;
                }

                if (!handled)
                {
                    if (pass.required)
                    {
                        result.errors.push_back(
                            "Required pass handler failed for pass '" + pass.id + "'.");
                        result.ok = false;
                    }
                    else
                    {
                        result.warnings.push_back(
                            "Optional pass handler reported failure for pass '" + pass.id + "'.");
                        result.skipped_optional_count++;
                    }
                    continue;
                }

                result.executed_count++;
            }

            return result.ok;
        }

    private:
        static uint16_t pass_id_key(PassId id)
        {
            return static_cast<uint16_t>(id);
        }

        std::unordered_map<uint16_t, Handler> typed_handlers_{};
        std::unordered_map<std::string, Handler> custom_handlers_{};
    };
}
