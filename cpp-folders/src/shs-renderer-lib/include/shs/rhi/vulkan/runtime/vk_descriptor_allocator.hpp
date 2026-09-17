#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>

#ifdef SHS_HAS_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace shs
{
#ifdef SHS_HAS_VULKAN

    class VulkanDescriptorAllocator
    {
    public:
        struct PoolSizes
        {
            std::vector<std::pair<VkDescriptorType, float>> sizes =
            {
                {VK_DESCRIPTOR_TYPE_SAMPLER, 0.5f},
                {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4.0f},
                {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 4.0f},
                {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1.0f},
                {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1.0f},
                {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1.0f},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2.0f},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2.0f},
                {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1.0f},
                {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1.0f},
                {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 0.5f}
            };
        };

        void init(VkDevice device)
        {
            device_ = device;
        }

        void cleanup()
        {
            for (auto p : free_pools_) vkDestroyDescriptorPool(device_, p, nullptr);
            for (auto p : used_pools_) vkDestroyDescriptorPool(device_, p, nullptr);
            free_pools_.clear();
            used_pools_.clear();
        }

        void reset_pools()
        {
            for (auto p : used_pools_)
            {
                vkResetDescriptorPool(device_, p, 0);
                free_pools_.push_back(p);
            }
            used_pools_.clear();
            current_pool_ = VK_NULL_HANDLE;
        }

        bool allocate(VkDescriptorSet* out_set, VkDescriptorSetLayout layout)
        {
            if (current_pool_ == VK_NULL_HANDLE)
            {
                current_pool_ = grab_pool();
                used_pools_.push_back(current_pool_);
            }

            VkDescriptorSetAllocateInfo ai{};
            ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool = current_pool_;
            ai.descriptorSetCount = 1;
            ai.pSetLayouts = &layout;

            VkResult res = vkAllocateDescriptorSets(device_, &ai, out_set);
            if (res == VK_ERROR_FRAGMENTED_POOL || res == VK_ERROR_OUT_OF_POOL_MEMORY)
            {
                current_pool_ = grab_pool();
                used_pools_.push_back(current_pool_);

                ai.descriptorPool = current_pool_;
                res = vkAllocateDescriptorSets(device_, &ai, out_set);
            }

            return res == VK_SUCCESS;
        }

    private:
        VkDevice device_ = VK_NULL_HANDLE;
        VkDescriptorPool current_pool_ = VK_NULL_HANDLE;
        PoolSizes descriptor_sizes_;
        std::vector<VkDescriptorPool> used_pools_;
        std::vector<VkDescriptorPool> free_pools_;

        VkDescriptorPool create_pool(uint32_t count, VkDescriptorPoolCreateFlags flags)
        {
            std::vector<VkDescriptorPoolSize> sizes;
            sizes.reserve(descriptor_sizes_.sizes.size());
            for (auto& p : descriptor_sizes_.sizes)
            {
                sizes.push_back({p.first, static_cast<uint32_t>(p.second * static_cast<float>(count))});
            }

            VkDescriptorPoolCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            ci.flags = flags;
            ci.maxSets = count;
            ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
            ci.pPoolSizes = sizes.data();

            VkDescriptorPool pool;
            if (vkCreateDescriptorPool(device_, &ci, nullptr, &pool) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateDescriptorPool failed in VulkanDescriptorAllocator");
            }
            return pool;
        }

        VkDescriptorPool grab_pool()
        {
            if (!free_pools_.empty())
            {
                VkDescriptorPool pool = free_pools_.back();
                free_pools_.pop_back();
                return pool;
            }
            else
            {
                return create_pool(1000, 0); // pool max sets
            }
        }
    };

#endif
}
