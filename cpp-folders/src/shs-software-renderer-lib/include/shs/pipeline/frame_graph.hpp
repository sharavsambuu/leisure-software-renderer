#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: frame_graph.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Render pass-уудын I/O contract дээр тулгуурлан гүйцэтгэлийн дараалал
            болон dependency-г шалгах хөнгөн FrameGraph хэрэгжүүлнэ.
*/


#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "shs/pipeline/render_pass.hpp"

namespace shs
{
    struct FrameGraphNode
    {
        IRenderPass* pass = nullptr;
        std::string pass_id{};
        PassIODesc io{};
        size_t original_index = 0;
    };

    struct FrameGraphReport
    {
        bool valid = true;
        std::vector<std::string> errors{};
        std::vector<std::string> warnings{};
    };

    class FrameGraph
    {
    public:
        void clear()
        {
            nodes_.clear();
            execution_order_.clear();
            report_ = FrameGraphReport{};
        }

        void add_node(FrameGraphNode node)
        {
            nodes_.push_back(std::move(node));
        }

        const std::vector<FrameGraphNode>& nodes() const { return nodes_; }
        const std::vector<size_t>& execution_order() const { return execution_order_; }
        const FrameGraphReport& report() const { return report_; }

        const std::vector<IRenderPass*> ordered_passes() const
        {
            std::vector<IRenderPass*> out{};
            out.reserve(execution_order_.size());
            for (size_t i : execution_order_)
            {
                if (i < nodes_.size() && nodes_[i].pass) out.push_back(nodes_[i].pass);
            }
            return out;
        }

        bool compile()
        {
            report_ = FrameGraphReport{};
            execution_order_.clear();
            if (nodes_.empty()) return true;

            const size_t n = nodes_.size();
            std::vector<std::vector<size_t>> edges(n);
            std::vector<int> indegree(n, 0);

            auto add_edge = [&](size_t a, size_t b)
            {
                if (a == b) return;
                auto& e = edges[a];
                if (std::find(e.begin(), e.end(), b) != e.end()) return;
                e.push_back(b);
                indegree[b]++;
            };

            for (size_t i = 0; i < n; ++i)
            {
                const auto& ni = nodes_[i];
                for (size_t j = i + 1; j < n; ++j)
                {
                    const auto& nj = nodes_[j];
                    for (const auto& ri : ni.io.resources)
                    {
                        if (ri.key == 0) continue;
                        const RenderBackendType bi = ni.pass ? ni.pass->preferred_backend() : RenderBackendType::Software;
                        const bool i_read = pass_access_has_read(ri.access);
                        const bool i_write = pass_access_has_write(ri.access);
                        for (const auto& rj : nj.io.resources)
                        {
                            if (rj.key == 0 || rj.key != ri.key) continue;
                            const RenderBackendType bj = nj.pass ? nj.pass->preferred_backend() : RenderBackendType::Software;
                            const bool j_read = pass_access_has_read(rj.access);
                            const bool j_write = pass_access_has_write(rj.access);

                            // Нэг ресурс дээрх write дарааллыг dependency болгон хувиргана.
                            if (i_write && (j_read || j_write))
                            {
                                add_edge(i, j);
                            }
                            else if (j_write && i_read)
                            {
                                // i нь уншихаасаа өмнө j-г бичүүлэх шаардлагатай.
                                add_edge(j, i);
                            }

                            // Hybrid planning sanity checks.
                            if ((i_write || j_write) && !pass_resource_domains_compatible(ri.domain, rj.domain))
                            {
                                report_.warnings.push_back(
                                    "Resource domain mismatch on '" + (ri.name.empty() ? std::string("unnamed") : ri.name)
                                    + "' between passes '" + ni.pass_id + "' (" + pass_resource_domain_name(ri.domain)
                                    + ") and '" + nj.pass_id + "' (" + pass_resource_domain_name(rj.domain) + ")."
                                );
                            }

                            if ((i_write || j_write) && bi != bj)
                            {
                                const bool interop_pair = (ni.pass && ni.pass->is_interop_pass()) || (nj.pass && nj.pass->is_interop_pass());
                                if (!interop_pair)
                                {
                                    report_.warnings.push_back(
                                        "Cross-backend dependency detected for resource '" + (ri.name.empty() ? std::string("unnamed") : ri.name)
                                        + "' between '" + ni.pass_id + "' (" + render_backend_type_name(bi)
                                        + ") and '" + nj.pass_id + "' (" + render_backend_type_name(bj)
                                        + "). Consider explicit interop/copy pass."
                                    );
                                }
                            }
                        }
                    }
                }
            }

            std::vector<size_t> zero_nodes{};
            zero_nodes.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                if (indegree[i] == 0) zero_nodes.push_back(i);
            }
            std::sort(zero_nodes.begin(), zero_nodes.end(), [&](size_t a, size_t b)
            {
                return nodes_[a].original_index < nodes_[b].original_index;
            });

            std::queue<size_t> q{};
            for (size_t i : zero_nodes) q.push(i);
            while (!q.empty())
            {
                const size_t v = q.front();
                q.pop();
                execution_order_.push_back(v);
                for (size_t to : edges[v])
                {
                    indegree[to]--;
                    if (indegree[to] == 0) q.push(to);
                }
            }

            if (execution_order_.size() != n)
            {
                report_.valid = false;
                report_.errors.push_back("FrameGraph cycle detected in pass resource dependencies.");
                execution_order_.clear();
                for (size_t i = 0; i < n; ++i) execution_order_.push_back(i);
                return false;
            }

            // Reordered warning: dependency contract order differs from add_pass order.
            bool reordered = false;
            for (size_t i = 0; i < n; ++i)
            {
                if (execution_order_[i] != i)
                {
                    reordered = true;
                    break;
                }
            }
            if (reordered)
            {
                report_.warnings.push_back("FrameGraph reordered passes to satisfy resource dependencies.");
            }

            return report_.valid;
        }

    private:
        std::vector<FrameGraphNode> nodes_{};
        std::vector<size_t> execution_order_{};
        FrameGraphReport report_{};
    };
}
