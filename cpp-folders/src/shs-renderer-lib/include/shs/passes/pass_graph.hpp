/*

    shs/passes/pass_graph.hpp

    PASS INTERFACE + PASS GRAPH (тушаалын дараалалтай executor)

    ЗОРИЛГО:
    - common pass.execute интерфэйсийг маш энгийнээр эхлүүлэх.
    - Одоохондоо dependency solver хэрэггүй: нэмсэн дарааллаар нь ажиллуулна.
    - Дараа нь хүсвэл name/tag, enable/disable, профайлер нэмнэ.

*/

#pragma once

#include <vector>
#include <memory>
#include <string>

#include "shs/passes/pass_context.hpp"

namespace shs
{
    // ---------------------------------------------
    // IPass: хамгийн минимал интерфэйс
    // ---------------------------------------------
    struct IPass
    {
        virtual ~IPass() = default;

        // Pass нэр (debug/log-д хэрэгтэй)
        virtual const char* name() const = 0;

        // Гол ажиллуулах функц
        virtual void execute(PassContext& pc) = 0;
    };

    // ---------------------------------------------
    // PassGraph: дараалсан executor
    // ---------------------------------------------
    class PassGraph
    {
    public:
        PassGraph() = default;

        // Unique ownership (ихэнх demo-д хамгийн цэвэр)
        template<typename TPass, typename... Args>
        TPass& add(Args&&... args)
        {
            auto p = std::make_unique<TPass>(std::forward<Args>(args)...);
            TPass& ref = *p;
            m_passes.emplace_back(std::move(p));
            return ref;
        }

        // Raw pointer (хэрэв гаднаас owned бол)
        void add_external(IPass* p)
        {
            m_external.push_back(p);
        }

        // Кадрын бүх pass-уудыг ажиллуулах
        void execute_all(PassContext& pc)
        {
            // owned passes
            for (auto& up : m_passes)
            {
                up->execute(pc);
            }
            // external passes
            for (auto* p : m_external)
            {
                p->execute(pc);
            }
        }

        void clear()
        {
            m_passes.clear();
            m_external.clear();
        }

        int count_owned() const { return (int)m_passes.size(); }
        int count_external() const { return (int)m_external.size(); }

    private:
        std::vector<std::unique_ptr<IPass>> m_passes;
        std::vector<IPass*> m_external;
    };

} // namespace shs
