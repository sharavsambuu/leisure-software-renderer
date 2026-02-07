// File: src/shs-renderer-lib/include/shs/gfx/rt_registry.hpp
#pragma once
/*
    SHS RENDERER LIB - RT REGISTRY

    ЗОРИЛГО:
    - Pipeline дотор "RT allocation"-ыг хожим төвлөрүүлэх суурь
    - Одоогоор demo талд байгаа бодит RT pointer-уудыг handle -> ptr map болгоно

    - Энд юу ч allocate хийхгүй (түр). Зөвхөн mapping.
*/

#include <cstdint>
#include <unordered_map>

#include "shs/gfx/rt_handle.hpp"

namespace shs
{
    class RTRegistry
    {
    public:
        void reset()
        {
            next_id_ = 1;
            map_.clear();
        }

        // Register an existing RT pointer from demo code
        template<typename THandle>
        THandle reg(void* ptr)
        {
            THandle h{};
            h.id = next_id_++;
            map_[h.id] = ptr;
            return h;
        }

        template<typename THandle>
        void* get(THandle h) const
        {
            auto it = map_.find(h.id);
            return (it == map_.end()) ? nullptr : it->second;
        }

    private:
        uint32_t next_id_ = 1;
        std::unordered_map<uint32_t, void*> map_;
    };
}

