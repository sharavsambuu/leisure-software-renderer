#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: vk_frame_ownership.hpp
    МОДУЛЬ: rhi/drivers/vulkan
    ЗОРИЛГО: Vulkan frame-slot ownership (ring) болон per-slot descriptor allocation
            загварыг reusable байдлаар төвлөрүүлэх helper.
*/


#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
    inline uint32_t vk_frame_slot(uint64_t frame_index, uint32_t slot_count)
    {
        if (slot_count == 0u) return 0u;
        return static_cast<uint32_t>(frame_index % static_cast<uint64_t>(slot_count));
    }

    template <typename T, size_t SlotCount>
    class VkFrameRing final
    {
        static_assert(SlotCount > 0, "VkFrameRing requires SlotCount > 0");

    public:
        static constexpr size_t slot_count_v = SlotCount;

        uint32_t slot_index(uint64_t frame_index) const
        {
            return vk_frame_slot(frame_index, static_cast<uint32_t>(SlotCount));
        }

        bool valid_slot(uint32_t slot) const
        {
            return slot < static_cast<uint32_t>(SlotCount);
        }

        T& at_slot(uint32_t slot)
        {
            if (!valid_slot(slot))
            {
                throw std::out_of_range("VkFrameRing slot out of range");
            }
            return slots_[slot];
        }

        const T& at_slot(uint32_t slot) const
        {
            if (!valid_slot(slot))
            {
                throw std::out_of_range("VkFrameRing slot out of range");
            }
            return slots_[slot];
        }

        T& at_frame(uint64_t frame_index)
        {
            return slots_[slot_index(frame_index)];
        }

        const T& at_frame(uint64_t frame_index) const
        {
            return slots_[slot_index(frame_index)];
        }

        T& operator[](size_t idx) { return slots_[idx]; }
        const T& operator[](size_t idx) const { return slots_[idx]; }

        auto begin() noexcept { return slots_.begin(); }
        auto end() noexcept { return slots_.end(); }
        auto begin() const noexcept { return slots_.begin(); }
        auto end() const noexcept { return slots_.end(); }

        template <typename Fn>
        void for_each(Fn&& fn)
        {
            for (uint32_t i = 0; i < static_cast<uint32_t>(SlotCount); ++i)
            {
                fn(i, slots_[i]);
            }
        }

        template <typename Fn>
        void for_each(Fn&& fn) const
        {
            for (uint32_t i = 0; i < static_cast<uint32_t>(SlotCount); ++i)
            {
                fn(i, slots_[i]);
            }
        }

    private:
        std::array<T, SlotCount> slots_{};
    };

#ifdef SHS_HAS_VULKAN
    template <size_t SlotCount>
    bool vk_allocate_descriptor_set_ring(
        VkDevice device,
        VkDescriptorPool descriptor_pool,
        VkDescriptorSetLayout set_layout,
        std::array<VkDescriptorSet, SlotCount>& out_sets)
    {
        static_assert(SlotCount > 0, "Descriptor set ring requires SlotCount > 0");
        if (device == VK_NULL_HANDLE || descriptor_pool == VK_NULL_HANDLE || set_layout == VK_NULL_HANDLE)
        {
            return false;
        }

        std::array<VkDescriptorSetLayout, SlotCount> layouts{};
        layouts.fill(set_layout);

        VkDescriptorSetAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        ai.descriptorPool = descriptor_pool;
        ai.descriptorSetCount = static_cast<uint32_t>(SlotCount);
        ai.pSetLayouts = layouts.data();
        return vkAllocateDescriptorSets(device, &ai, out_sets.data()) == VK_SUCCESS;
    }
#endif
}

