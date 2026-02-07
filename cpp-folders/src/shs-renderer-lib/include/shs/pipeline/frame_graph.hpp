#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: frame_graph.hpp
    МОДУЛЬ: pipeline
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн pipeline модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <functional>
#include <string>
#include <vector>

namespace shs
{
    struct FrameGraphPass
    {
        std::string name{};
        std::function<void()> execute{};
    };

    class FrameGraph
    {
    public:
        void add_pass(FrameGraphPass pass)
        {
            passes_.push_back(std::move(pass));
        }

        void clear()
        {
            passes_.clear();
        }

        void execute_all()
        {
            for (auto& p : passes_)
            {
                if (p.execute) p.execute();
            }
        }

    private:
        std::vector<FrameGraphPass> passes_{};
    };
}

