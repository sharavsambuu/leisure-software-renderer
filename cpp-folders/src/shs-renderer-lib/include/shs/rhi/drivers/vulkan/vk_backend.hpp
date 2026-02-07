#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: vk_backend.hpp
    МОДУЛЬ: rhi/drivers/vulkan
    ЗОРИЛГО: Vulkan backend-ийн суурь класс.
            Одоогийн байдлаар lifecycle ба capability contract-уудыг хангана.
            Минимал Vulkan runtime + swapchain замыг хангана.
*/


#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include "shs/rhi/core/backend.hpp"
#include "shs/rhi/drivers/vulkan/vk_component_notes.hpp"

struct SDL_Window;

#ifdef SHS_HAS_VULKAN
#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#endif

namespace shs
{
    class VulkanRenderBackend final : public IRenderBackend
    {
    public:
        struct InitDesc
        {
            SDL_Window* window = nullptr;
            int width = 0;
            int height = 0;
            bool enable_validation = false;
            const char* app_name = "shs-renderer-lib";
        };

        struct FrameInfo
        {
#ifdef SHS_HAS_VULKAN
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            VkFramebuffer framebuffer = VK_NULL_HANDLE;
            VkRenderPass render_pass = VK_NULL_HANDLE;
            VkExtent2D extent{};
            VkFormat format = VK_FORMAT_UNDEFINED;
            uint32_t image_index = 0;
#endif
        };

        ~VulkanRenderBackend() override { shutdown(); }

        RenderBackendType type() const override { return RenderBackendType::Vulkan; }
        BackendCapabilities capabilities() const override
        {
            BackendCapabilities c = capabilities_;
            if (!capabilities_ready_)
            {
                c.queues.graphics_count = 1;
                c.features.validation_layers = false;
                c.features.push_constants = true;
                c.features.multithread_command_recording = true;
                c.limits.max_frames_in_flight = 2;
                c.limits.max_color_attachments = 1;
                c.limits.max_descriptor_sets_per_pipeline = 1;
                c.limits.max_push_constant_bytes = 128;
                c.supports_offscreen = true;
            }
#ifdef SHS_HAS_VULKAN
            c.features.validation_layers = layer_supported("VK_LAYER_KHRONOS_validation");
#endif
            return c;
        }

        void begin_frame(Context& ctx, const RenderBackendFrameInfo& frame) override
        {
            (void)ctx;
            if (frame.width > 0 && frame.height > 0)
            {
                request_resize(frame.width, frame.height);
            }
            (void)ensure_initialized();
        }

        void end_frame(Context& ctx, const RenderBackendFrameInfo& frame) override
        {
            (void)ctx;
            (void)frame;
        }

        void on_resize(Context& ctx, int w, int h) override
        {
            (void)ctx;
            request_resize(w, h);
        }

        bool ready() const { return initialized_; }

        bool init_sdl(const InitDesc& desc)
        {
#ifdef SHS_HAS_VULKAN
            shutdown();
            if (!desc.window) return false;
            window_ = desc.window;
            enable_validation_ = desc.enable_validation;
            requested_width_ = desc.width;
            requested_height_ = desc.height;
            app_name_ = desc.app_name ? desc.app_name : "shs-renderer-lib";
            layers_.clear();
            resize_pending_ = false;
            swapchain_needs_rebuild_ = false;
            device_lost_ = false;
            current_frame_ = 0;
            capabilities_ready_ = false;
            init_attempted_ = false;
            return ensure_initialized();
#else
            (void)desc;
            return false;
#endif
        }

        void request_resize(int w, int h)
        {
#ifdef SHS_HAS_VULKAN
            if (w <= 0 || h <= 0)
            {
                resize_pending_ = true;
                return;
            }
            requested_width_ = w;
            requested_height_ = h;
            resize_pending_ = true;
#else
            (void)w;
            (void)h;
#endif
        }

        bool begin_frame(Context& ctx, const RenderBackendFrameInfo& frame, FrameInfo& out)
        {
#ifdef SHS_HAS_VULKAN
            (void)ctx;
            if (!ensure_initialized()) return false;
            if (device_lost_) return false;
            if (resize_pending_ || swapchain_needs_rebuild_)
            {
                if (!recreate_swapchain()) return false;
            }
            if (swapchain_ == VK_NULL_HANDLE) return false;

            const uint32_t cur = current_frame_ % kMaxFramesInFlight;
            const VkResult wait_res = vkWaitForFences(device_, 1, &inflight_fences_[cur], VK_TRUE, UINT64_MAX);
            if (wait_res == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return false;
            }
            if (wait_res != VK_SUCCESS) return false;

            uint32_t image_index = 0;
            VkResult res = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, image_available_[cur], VK_NULL_HANDLE, &image_index);
            if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_SURFACE_LOST_KHR)
            {
                swapchain_needs_rebuild_ = true;
                return false;
            }
            if (res == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return false;
            }
            if (res != VK_SUCCESS)
            {
                return false;
            }

            if (image_index >= cmd_bufs_.size() || image_index >= framebuffers_.size())
            {
                swapchain_needs_rebuild_ = true;
                return false;
            }
            if (images_in_flight_.size() == images_.size() && images_in_flight_[image_index] != VK_NULL_HANDLE)
            {
                const VkResult wait_img = vkWaitForFences(device_, 1, &images_in_flight_[image_index], VK_TRUE, UINT64_MAX);
                if (wait_img == VK_ERROR_DEVICE_LOST)
                {
                    device_lost_ = true;
                    return false;
                }
                if (wait_img != VK_SUCCESS) return false;
            }
            if (images_in_flight_.size() == images_.size())
            {
                images_in_flight_[image_index] = inflight_fences_[cur];
            }

            const VkResult reset_fence = vkResetFences(device_, 1, &inflight_fences_[cur]);
            if (reset_fence == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return false;
            }
            if (reset_fence != VK_SUCCESS) return false;

            VkCommandBuffer cb = cmd_bufs_[image_index];
            const VkResult reset_cb = vkResetCommandBuffer(cb, 0);
            if (reset_cb == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return false;
            }
            if (reset_cb != VK_SUCCESS) return false;

            out.cmd = cb;
            out.framebuffer = framebuffers_[image_index];
            out.render_pass = render_pass_;
            out.extent = extent_;
            out.format = swapchain_format_;
            out.image_index = image_index;
            (void)frame;
            return true;
#else
            (void)ctx;
            (void)frame;
            (void)out;
            return false;
#endif
        }

        void end_frame(const FrameInfo& info)
        {
#ifdef SHS_HAS_VULKAN
            if (device_lost_) return;
            const uint32_t cur = current_frame_ % kMaxFramesInFlight;

            VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
            VkSubmitInfo si{};
            si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &image_available_[cur];
            si.pWaitDstStageMask = &wait_stage;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &info.cmd;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &render_finished_[cur];
            const VkResult submit_res = vkQueueSubmit(graphics_q_, 1, &si, inflight_fences_[cur]);
            if (submit_res == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return;
            }
            if (submit_res != VK_SUCCESS) return;

            VkPresentInfoKHR pi{};
            pi.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
            pi.waitSemaphoreCount = 1;
            pi.pWaitSemaphores = &render_finished_[cur];
            pi.swapchainCount = 1;
            pi.pSwapchains = &swapchain_;
            pi.pImageIndices = &info.image_index;
            const VkResult pres = vkQueuePresentKHR(present_q_, &pi);
            if (pres == VK_ERROR_OUT_OF_DATE_KHR || pres == VK_SUBOPTIMAL_KHR || pres == VK_ERROR_SURFACE_LOST_KHR)
            {
                swapchain_needs_rebuild_ = true;
            }
            else if (pres == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return;
            }
            else if (pres != VK_SUCCESS)
            {
                return;
            }
            ++current_frame_;
#else
            (void)info;
#endif
        }

#ifdef SHS_HAS_VULKAN
        VkDevice device() const { return device_; }
        VkPhysicalDevice physical_device() const { return gpu_; }
        VkQueue graphics_queue() const { return graphics_q_; }
        VkQueue present_queue() const { return present_q_; }
        uint32_t graphics_queue_family_index() const { return qf_.graphics.value_or(0u); }
        VkRenderPass render_pass() const { return render_pass_; }
        VkExtent2D swapchain_extent() const { return extent_; }
        VkFormat swapchain_format() const { return swapchain_format_; }
        VkFormat depth_format() const { return depth_format_; }
        VkImageUsageFlags swapchain_usage_flags() const { return swapchain_usage_; }
        uint64_t swapchain_generation() const { return swapchain_generation_; }
        bool has_depth_attachment() const { return depth_view_ != VK_NULL_HANDLE; }
        VkImage swapchain_image(uint32_t image_index) const
        {
            if (image_index >= images_.size()) return VK_NULL_HANDLE;
            return images_[image_index];
        }
#endif

    private:
#ifdef SHS_HAS_VULKAN
        static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
            VkDebugUtilsMessageSeverityFlagBitsEXT severity,
            VkDebugUtilsMessageTypeFlagsEXT type,
            const VkDebugUtilsMessengerCallbackDataEXT* data,
            void* user)
        {
            (void)severity;
            (void)type;
            (void)user;
            if (data && data->pMessage)
            {
                std::fprintf(stderr, "[vulkan] %s\n", data->pMessage);
            }
            return VK_FALSE;
        }

        static bool layer_supported(const char* name)
        {
            uint32_t count = 0;
            vkEnumerateInstanceLayerProperties(&count, nullptr);
            std::vector<VkLayerProperties> layers(count);
            vkEnumerateInstanceLayerProperties(&count, layers.data());
            for (const auto& l : layers)
            {
                if (std::strcmp(l.layerName, name) == 0) return true;
            }
            return false;
        }

        static bool extension_supported(const char* name)
        {
            uint32_t count = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> exts(count);
            vkEnumerateInstanceExtensionProperties(nullptr, &count, exts.data());
            for (const auto& e : exts)
            {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        }

        struct QueueFamilies
        {
            std::optional<uint32_t> graphics{};
            std::optional<uint32_t> present{};
            bool ok() const { return graphics.has_value() && present.has_value(); }
        };

        struct SwapchainSupport
        {
            VkSurfaceCapabilitiesKHR caps{};
            std::vector<VkSurfaceFormatKHR> formats{};
            std::vector<VkPresentModeKHR> modes{};
        };

        bool ensure_initialized()
        {
            if (init_attempted_) return initialized_;
            init_attempted_ = true;
            if (!window_) return false;

            if (!create_instance()) { shutdown(); return false; }
            if (!SDL_Vulkan_CreateSurface(window_, instance_, &surface_)) { shutdown(); return false; }
            if (!pick_physical_device()) { shutdown(); return false; }
            if (!create_device_and_queues()) { shutdown(); return false; }
            if (!create_swapchain()) { shutdown(); return false; }
            if (!create_render_pass()) { shutdown(); return false; }
            if (!create_depth_resources()) { shutdown(); return false; }
            if (!create_framebuffers()) { shutdown(); return false; }
            if (!create_command_pool_and_buffers()) { shutdown(); return false; }
            if (!create_sync_objects()) { shutdown(); return false; }
            refresh_capabilities();
            device_lost_ = false;
            initialized_ = true;
            return true;
        }

        bool create_instance()
        {
            unsigned int ext_count = 0;
            if (!SDL_Vulkan_GetInstanceExtensions(window_, &ext_count, nullptr)) return false;
            std::vector<const char*> exts(ext_count);
            if (!SDL_Vulkan_GetInstanceExtensions(window_, &ext_count, exts.data())) return false;

            auto add_instance_ext_if_supported = [&](const char* ext_name) {
                if (!ext_name) return false;
                for (const char* existing : exts)
                {
                    if (existing && std::strcmp(existing, ext_name) == 0) return true;
                }
                if (!extension_supported(ext_name)) return false;
                exts.push_back(ext_name);
                return true;
            };

            const char* validation_layer = "VK_LAYER_KHRONOS_validation";
            if (enable_validation_ && layer_supported(validation_layer))
            {
                layers_.push_back(validation_layer);
            }

            if (enable_validation_)
            {
                add_instance_ext_if_supported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            VkApplicationInfo app{};
            app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app.pApplicationName = app_name_.c_str();
            app.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
            app.pEngineName = "shs";
            app.engineVersion = VK_MAKE_VERSION(0, 1, 0);
            app.apiVersion = VK_API_VERSION_1_1;

            VkInstanceCreateInfo ci{};
            ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            ci.pApplicationInfo = &app;
            ci.enabledLayerCount = static_cast<uint32_t>(layers_.size());
            ci.ppEnabledLayerNames = layers_.empty() ? nullptr : layers_.data();
            // MoltenVK portability drivers require both extension + enumerate flag.
            constexpr const char* kPortabilityEnumExt = "VK_KHR_portability_enumeration";
            if (add_instance_ext_if_supported(kPortabilityEnumExt))
            {
#ifdef VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
                ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#else
                ci.flags |= static_cast<VkInstanceCreateFlags>(0x00000001u);
#endif
            }
            ci.enabledExtensionCount = static_cast<uint32_t>(exts.size());
            ci.ppEnabledExtensionNames = exts.empty() ? nullptr : exts.data();

            VkDebugUtilsMessengerCreateInfoEXT dbg{};
            if (enable_validation_ && !layers_.empty())
            {
                dbg.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                dbg.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                dbg.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                dbg.pfnUserCallback = debug_callback;
                ci.pNext = &dbg;
            }

            if (vkCreateInstance(&ci, nullptr, &instance_) != VK_SUCCESS) return false;

            if (enable_validation_ && extension_supported(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
            {
                auto create_fn = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT");
                if (create_fn)
                {
                    VkDebugUtilsMessengerCreateInfoEXT dbg_create{};
                    dbg_create.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
                    dbg_create.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
                    dbg_create.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
                    dbg_create.pfnUserCallback = debug_callback;
                    create_fn(instance_, &dbg_create, nullptr, &debug_messenger_);
                }
            }
            return true;
        }

        QueueFamilies find_queue_families(VkPhysicalDevice gpu)
        {
            QueueFamilies out{};
            uint32_t n = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(gpu, &n, nullptr);
            std::vector<VkQueueFamilyProperties> props(n);
            vkGetPhysicalDeviceQueueFamilyProperties(gpu, &n, props.data());
            for (uint32_t i = 0; i < n; ++i)
            {
                if (props[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) out.graphics = i;
                VkBool32 present = VK_FALSE;
                vkGetPhysicalDeviceSurfaceSupportKHR(gpu, i, surface_, &present);
                if (present) out.present = i;
                if (out.ok()) break;
            }
            return out;
        }

        SwapchainSupport query_swapchain_support(VkPhysicalDevice gpu)
        {
            SwapchainSupport out{};
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu, surface_, &out.caps);
            uint32_t nf = 0;
            vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface_, &nf, nullptr);
            if (nf)
            {
                out.formats.resize(nf);
                vkGetPhysicalDeviceSurfaceFormatsKHR(gpu, surface_, &nf, out.formats.data());
            }
            uint32_t nm = 0;
            vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface_, &nm, nullptr);
            if (nm)
            {
                out.modes.resize(nm);
                vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface_, &nm, out.modes.data());
            }
            return out;
        }

        bool device_extension_supported(VkPhysicalDevice gpu, const char* name)
        {
            uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> exts(count);
            vkEnumerateDeviceExtensionProperties(gpu, nullptr, &count, exts.data());
            for (const auto& e : exts)
            {
                if (std::strcmp(e.extensionName, name) == 0) return true;
            }
            return false;
        }

        bool pick_physical_device()
        {
            uint32_t gpu_count = 0;
            vkEnumeratePhysicalDevices(instance_, &gpu_count, nullptr);
            if (gpu_count == 0) return false;
            std::vector<VkPhysicalDevice> gpus(gpu_count);
            vkEnumeratePhysicalDevices(instance_, &gpu_count, gpus.data());
            for (VkPhysicalDevice gpu : gpus)
            {
                QueueFamilies qf = find_queue_families(gpu);
                SwapchainSupport sc = query_swapchain_support(gpu);
                if (!qf.ok() || sc.formats.empty() || sc.modes.empty()) continue;
                if (!device_extension_supported(gpu, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) continue;
                gpu_ = gpu;
                qf_ = qf;
                return true;
            }
            return false;
        }

        bool create_device_and_queues()
        {
            if (!gpu_) return false;
            float qprio = 1.0f;
            std::vector<uint32_t> fams{*qf_.graphics};
            if (*qf_.present != *qf_.graphics) fams.push_back(*qf_.present);
            std::vector<VkDeviceQueueCreateInfo> qcis{};
            for (uint32_t fam : fams)
            {
                VkDeviceQueueCreateInfo qci{};
                qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                qci.queueFamilyIndex = fam;
                qci.queueCount = 1;
                qci.pQueuePriorities = &qprio;
                qcis.push_back(qci);
            }

            std::vector<const char*> device_exts{};
            device_exts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            constexpr const char* kPortabilitySubsetExt = "VK_KHR_portability_subset";
            if (device_extension_supported(gpu_, kPortabilitySubsetExt))
            {
                device_exts.push_back(kPortabilitySubsetExt);
            }

            VkDeviceCreateInfo dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            dci.queueCreateInfoCount = static_cast<uint32_t>(qcis.size());
            dci.pQueueCreateInfos = qcis.data();
            dci.enabledExtensionCount = static_cast<uint32_t>(device_exts.size());
            dci.ppEnabledExtensionNames = device_exts.data();

            if (vkCreateDevice(gpu_, &dci, nullptr, &device_) != VK_SUCCESS) return false;
            vkGetDeviceQueue(device_, *qf_.graphics, 0, &graphics_q_);
            vkGetDeviceQueue(device_, *qf_.present, 0, &present_q_);
            return true;
        }

        void refresh_capabilities()
        {
            BackendCapabilities c{};
            c.limits.max_frames_in_flight = kMaxFramesInFlight;
            c.supports_offscreen = true;
            c.supports_present = surface_ != VK_NULL_HANDLE;
            c.features.validation_layers = layer_supported("VK_LAYER_KHRONOS_validation");
            c.features.push_constants = true;
            c.features.multithread_command_recording = true;
            c.features.timeline_semaphore = false;
            c.features.descriptor_indexing = false;
            c.features.dynamic_rendering = false;
            c.features.async_compute = false;
            c.limits.max_color_attachments = 1;
            c.limits.max_descriptor_sets_per_pipeline = 1;
            c.limits.max_push_constant_bytes = 128;

            if (gpu_ != VK_NULL_HANDLE)
            {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(gpu_, &props);
                c.limits.max_color_attachments = std::max<uint32_t>(1u, props.limits.maxColorAttachments);
                c.limits.max_descriptor_sets_per_pipeline = std::max<uint32_t>(1u, props.limits.maxBoundDescriptorSets);
                c.limits.max_push_constant_bytes = std::max<uint32_t>(1u, props.limits.maxPushConstantsSize);
                c.limits.min_uniform_buffer_offset_alignment = std::max<uint32_t>(1u, static_cast<uint32_t>(props.limits.minUniformBufferOffsetAlignment));

                uint32_t qcount = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &qcount, nullptr);
                std::vector<VkQueueFamilyProperties> qprops(qcount);
                vkGetPhysicalDeviceQueueFamilyProperties(gpu_, &qcount, qprops.data());

                bool has_dedicated_compute = false;
                for (uint32_t i = 0; i < qcount; ++i)
                {
                    const VkQueueFamilyProperties& qp = qprops[i];
                    if (qp.queueCount == 0) continue;
                    if ((qp.queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
                    {
                        c.queues.graphics_count += qp.queueCount;
                    }
                    if ((qp.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0)
                    {
                        c.queues.compute_count += qp.queueCount;
                        if ((qp.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)
                        {
                            has_dedicated_compute = true;
                        }
                    }
                    if ((qp.queueFlags & VK_QUEUE_TRANSFER_BIT) != 0)
                    {
                        c.queues.transfer_count += qp.queueCount;
                    }
                    VkBool32 present = VK_FALSE;
                    if (surface_ != VK_NULL_HANDLE)
                    {
                        vkGetPhysicalDeviceSurfaceSupportKHR(gpu_, i, surface_, &present);
                    }
                    if (present == VK_TRUE)
                    {
                        c.queues.present_count += qp.queueCount;
                    }
                }
                c.features.async_compute = has_dedicated_compute;
            }

            capabilities_ = c;
            capabilities_ready_ = true;
        }

        VkSurfaceFormatKHR choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats)
        {
            for (const auto& f : formats)
            {
                if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return f;
            }
            return formats.empty() ? VkSurfaceFormatKHR{} : formats[0];
        }

        VkPresentModeKHR choose_present_mode(const std::vector<VkPresentModeKHR>& modes)
        {
            const char* present_mode_env = std::getenv("SHS_VK_PRESENT_MODE");
            const bool prefer_mailbox =
                present_mode_env &&
                (std::strcmp(present_mode_env, "mailbox") == 0 ||
                 std::strcmp(present_mode_env, "MAILBOX") == 0);

            if (prefer_mailbox)
            {
                for (const auto& m : modes)
                {
                    if (m == VK_PRESENT_MODE_MAILBOX_KHR) return m;
                }
            }

            for (const auto& m : modes)
            {
                if (m == VK_PRESENT_MODE_FIFO_KHR) return m;
            }
            return modes.empty() ? VK_PRESENT_MODE_FIFO_KHR : modes[0];
        }

        VkExtent2D choose_extent(const VkSurfaceCapabilitiesKHR& caps)
        {
            if (caps.currentExtent.width != UINT32_MAX) return caps.currentExtent;

            int w = requested_width_;
            int h = requested_height_;
            if (w <= 0 || h <= 0)
            {
                int dw = 0;
                int dh = 0;
                SDL_Vulkan_GetDrawableSize(window_, &dw, &dh);
                w = dw;
                h = dh;
            }
            VkExtent2D out{};
            out.width = std::clamp((uint32_t)w, caps.minImageExtent.width, caps.maxImageExtent.width);
            out.height = std::clamp((uint32_t)h, caps.minImageExtent.height, caps.maxImageExtent.height);
            return out;
        }

        bool has_stencil_component(VkFormat fmt) const
        {
            return fmt == VK_FORMAT_D32_SFLOAT_S8_UINT || fmt == VK_FORMAT_D24_UNORM_S8_UINT;
        }

        VkFormat choose_depth_format() const
        {
            const VkFormat candidates[] = {
                VK_FORMAT_D32_SFLOAT,
                VK_FORMAT_D32_SFLOAT_S8_UINT,
                VK_FORMAT_D24_UNORM_S8_UINT
            };
            for (VkFormat fmt : candidates)
            {
                VkFormatProperties props{};
                vkGetPhysicalDeviceFormatProperties(gpu_, fmt, &props);
                if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0)
                {
                    return fmt;
                }
            }
            return VK_FORMAT_UNDEFINED;
        }

        bool create_swapchain()
        {
            SwapchainSupport sc = query_swapchain_support(gpu_);
            if (sc.formats.empty() || sc.modes.empty()) return false;
            VkSurfaceFormatKHR sf = choose_surface_format(sc.formats);
            VkPresentModeKHR pm = choose_present_mode(sc.modes);
            VkExtent2D extent = choose_extent(sc.caps);
            if (extent.width == 0 || extent.height == 0) return false;

            uint32_t img_count = sc.caps.minImageCount + 1;
            if (sc.caps.maxImageCount > 0 && img_count > sc.caps.maxImageCount) img_count = sc.caps.maxImageCount;

            const uint32_t qidx[] = {*qf_.graphics, *qf_.present};
            VkSwapchainCreateInfoKHR sci{};
            sci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            sci.surface = surface_;
            sci.minImageCount = img_count;
            sci.imageFormat = sf.format;
            sci.imageColorSpace = sf.colorSpace;
            sci.imageExtent = extent;
            sci.imageArrayLayers = 1;
            VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            if ((sc.caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0)
            {
                usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            }
            sci.imageUsage = usage;
            if (*qf_.graphics != *qf_.present)
            {
                sci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                sci.queueFamilyIndexCount = 2;
                sci.pQueueFamilyIndices = qidx;
            }
            else
            {
                sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            }
            sci.preTransform = sc.caps.currentTransform;
            sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
            sci.presentMode = pm;
            sci.clipped = VK_TRUE;
            sci.oldSwapchain = VK_NULL_HANDLE;
            if (vkCreateSwapchainKHR(device_, &sci, nullptr, &swapchain_) != VK_SUCCESS) return false;

            extent_ = extent;
            swapchain_format_ = sf.format;
            swapchain_usage_ = usage;
            depth_format_ = choose_depth_format();

            uint32_t nimg = 0;
            if (vkGetSwapchainImagesKHR(device_, swapchain_, &nimg, nullptr) != VK_SUCCESS || nimg == 0)
            {
                return false;
            }
            images_.resize(nimg);
            if (vkGetSwapchainImagesKHR(device_, swapchain_, &nimg, images_.data()) != VK_SUCCESS)
            {
                return false;
            }

            views_.resize(images_.size());
            for (size_t i = 0; i < images_.size(); ++i)
            {
                VkImageViewCreateInfo iv{};
                iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                iv.image = images_[i];
                iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
                iv.format = swapchain_format_;
                iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                iv.subresourceRange.levelCount = 1;
                iv.subresourceRange.layerCount = 1;
                if (vkCreateImageView(device_, &iv, nullptr, &views_[i]) != VK_SUCCESS) return false;
            }
            images_in_flight_.assign(images_.size(), VK_NULL_HANDLE);
            ++swapchain_generation_;
            return true;
        }

        bool create_render_pass()
        {
            VkAttachmentDescription color{};
            color.format = swapchain_format_;
            color.samples = VK_SAMPLE_COUNT_1_BIT;
            color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            VkAttachmentReference color_ref{};
            color_ref.attachment = 0;
            color_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription depth{};
            depth.format = depth_format_;
            depth.samples = VK_SAMPLE_COUNT_1_BIT;
            depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference depth_ref{};
            depth_ref.attachment = 1;
            depth_ref.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkSubpassDescription sub{};
            sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            sub.colorAttachmentCount = 1;
            sub.pColorAttachments = &color_ref;
            if (depth_format_ != VK_FORMAT_UNDEFINED)
            {
                sub.pDepthStencilAttachment = &depth_ref;
            }

            VkSubpassDependency dep{};
            dep.srcSubpass = VK_SUBPASS_EXTERNAL;
            dep.dstSubpass = 0;
            dep.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dep.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            std::array<VkAttachmentDescription, 2> attachments{};
            attachments[0] = color;
            attachments[1] = depth;

            VkRenderPassCreateInfo rp{};
            rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            rp.attachmentCount = (depth_format_ != VK_FORMAT_UNDEFINED) ? 2u : 1u;
            rp.pAttachments = attachments.data();
            rp.subpassCount = 1;
            rp.pSubpasses = &sub;
            rp.dependencyCount = 1;
            rp.pDependencies = &dep;
            if (vkCreateRenderPass(device_, &rp, nullptr, &render_pass_) != VK_SUCCESS) return false;
            return true;
        }

        bool create_framebuffers()
        {
            framebuffers_.resize(views_.size());
            for (size_t i = 0; i < views_.size(); ++i)
            {
                std::array<VkImageView, 2> att{};
                att[0] = views_[i];
                uint32_t att_count = 1;
                if (depth_view_ != VK_NULL_HANDLE)
                {
                    att[1] = depth_view_;
                    att_count = 2;
                }
                VkFramebufferCreateInfo fb{};
                fb.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                fb.renderPass = render_pass_;
                fb.attachmentCount = att_count;
                fb.pAttachments = att.data();
                fb.width = extent_.width;
                fb.height = extent_.height;
                fb.layers = 1;
                if (vkCreateFramebuffer(device_, &fb, nullptr, &framebuffers_[i]) != VK_SUCCESS) return false;
            }
            return true;
        }

        bool create_depth_resources()
        {
            if (depth_format_ == VK_FORMAT_UNDEFINED) return true;
            if (extent_.width == 0 || extent_.height == 0) return false;

            VkImageCreateInfo ii{};
            ii.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            ii.imageType = VK_IMAGE_TYPE_2D;
            ii.extent.width = extent_.width;
            ii.extent.height = extent_.height;
            ii.extent.depth = 1;
            ii.mipLevels = 1;
            ii.arrayLayers = 1;
            ii.format = depth_format_;
            ii.tiling = VK_IMAGE_TILING_OPTIMAL;
            ii.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            ii.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            ii.samples = VK_SAMPLE_COUNT_1_BIT;
            ii.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateImage(device_, &ii, nullptr, &depth_image_) != VK_SUCCESS) return false;

            VkMemoryRequirements req{};
            vkGetImageMemoryRequirements(device_, depth_image_, &req);
            VkPhysicalDeviceMemoryProperties mp{};
            vkGetPhysicalDeviceMemoryProperties(gpu_, &mp);
            uint32_t memory_type = UINT32_MAX;
            for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            {
                const bool type_ok = (req.memoryTypeBits & (1u << i)) != 0;
                const bool prop_ok = (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
                if (type_ok && prop_ok)
                {
                    memory_type = i;
                    break;
                }
            }
            if (memory_type == UINT32_MAX) return false;

            VkMemoryAllocateInfo mai{};
            mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = memory_type;
            if (vkAllocateMemory(device_, &mai, nullptr, &depth_memory_) != VK_SUCCESS) return false;
            if (vkBindImageMemory(device_, depth_image_, depth_memory_, 0) != VK_SUCCESS) return false;

            VkImageViewCreateInfo iv{};
            iv.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            iv.image = depth_image_;
            iv.viewType = VK_IMAGE_VIEW_TYPE_2D;
            iv.format = depth_format_;
            iv.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            if (has_stencil_component(depth_format_)) iv.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            iv.subresourceRange.baseMipLevel = 0;
            iv.subresourceRange.levelCount = 1;
            iv.subresourceRange.baseArrayLayer = 0;
            iv.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device_, &iv, nullptr, &depth_view_) != VK_SUCCESS) return false;
            return true;
        }

        bool create_command_pool_and_buffers()
        {
            if (cmd_pool_ == VK_NULL_HANDLE)
            {
                VkCommandPoolCreateInfo cp{};
                cp.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
                cp.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
                cp.queueFamilyIndex = *qf_.graphics;
                if (vkCreateCommandPool(device_, &cp, nullptr, &cmd_pool_) != VK_SUCCESS) return false;
            }

            if (!cmd_bufs_.empty())
            {
                vkFreeCommandBuffers(device_, cmd_pool_, static_cast<uint32_t>(cmd_bufs_.size()), cmd_bufs_.data());
                cmd_bufs_.clear();
            }

            cmd_bufs_.resize(framebuffers_.size());
            VkCommandBufferAllocateInfo cba{};
            cba.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            cba.commandPool = cmd_pool_;
            cba.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cba.commandBufferCount = static_cast<uint32_t>(cmd_bufs_.size());
            if (vkAllocateCommandBuffers(device_, &cba, cmd_bufs_.data()) != VK_SUCCESS) return false;
            return true;
        }

        bool create_sync_objects()
        {
            VkSemaphoreCreateInfo sem{};
            sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            VkFenceCreateInfo fe{};
            fe.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fe.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
            {
                if (image_available_[i] == VK_NULL_HANDLE &&
                    vkCreateSemaphore(device_, &sem, nullptr, &image_available_[i]) != VK_SUCCESS)
                    return false;
                if (render_finished_[i] == VK_NULL_HANDLE &&
                    vkCreateSemaphore(device_, &sem, nullptr, &render_finished_[i]) != VK_SUCCESS)
                    return false;
                if (inflight_fences_[i] == VK_NULL_HANDLE &&
                    vkCreateFence(device_, &fe, nullptr, &inflight_fences_[i]) != VK_SUCCESS)
                    return false;
            }
            return true;
        }

        void destroy_swapchain_objects()
        {
            for (auto fb : framebuffers_) vkDestroyFramebuffer(device_, fb, nullptr);
            framebuffers_.clear();
            if (render_pass_ != VK_NULL_HANDLE)
            {
                vkDestroyRenderPass(device_, render_pass_, nullptr);
                render_pass_ = VK_NULL_HANDLE;
            }
            for (auto iv : views_) vkDestroyImageView(device_, iv, nullptr);
            views_.clear();
            images_.clear();
            images_in_flight_.clear();
            if (depth_view_ != VK_NULL_HANDLE)
            {
                vkDestroyImageView(device_, depth_view_, nullptr);
                depth_view_ = VK_NULL_HANDLE;
            }
            if (depth_image_ != VK_NULL_HANDLE)
            {
                vkDestroyImage(device_, depth_image_, nullptr);
                depth_image_ = VK_NULL_HANDLE;
            }
            if (depth_memory_ != VK_NULL_HANDLE)
            {
                vkFreeMemory(device_, depth_memory_, nullptr);
                depth_memory_ = VK_NULL_HANDLE;
            }
            depth_format_ = VK_FORMAT_UNDEFINED;
            swapchain_usage_ = 0;
            if (swapchain_ != VK_NULL_HANDLE)
            {
                vkDestroySwapchainKHR(device_, swapchain_, nullptr);
                swapchain_ = VK_NULL_HANDLE;
            }
        }

        bool recreate_swapchain()
        {
            if (device_ == VK_NULL_HANDLE) return false;
            int w = 0;
            int h = 0;
            SDL_Vulkan_GetDrawableSize(window_, &w, &h);
            if (w <= 0 || h <= 0) return false;
            const VkResult idle_res = vkDeviceWaitIdle(device_);
            if (idle_res == VK_ERROR_DEVICE_LOST)
            {
                device_lost_ = true;
                return false;
            }
            if (idle_res != VK_SUCCESS) return false;
            destroy_swapchain_objects();
            if (!create_swapchain()) return false;
            if (!create_render_pass()) return false;
            if (!create_depth_resources()) return false;
            if (!create_framebuffers()) return false;
            if (!create_command_pool_and_buffers()) return false;
            resize_pending_ = false;
            swapchain_needs_rebuild_ = false;
            return true;
        }

        void shutdown()
        {
            if (!init_attempted_ && !initialized_) return;
            if (device_ != VK_NULL_HANDLE) (void)vkDeviceWaitIdle(device_);
            destroy_swapchain_objects();
            if (cmd_pool_ != VK_NULL_HANDLE)
            {
                vkDestroyCommandPool(device_, cmd_pool_, nullptr);
                cmd_pool_ = VK_NULL_HANDLE;
            }
            for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
            {
                if (image_available_[i] != VK_NULL_HANDLE) vkDestroySemaphore(device_, image_available_[i], nullptr);
                if (render_finished_[i] != VK_NULL_HANDLE) vkDestroySemaphore(device_, render_finished_[i], nullptr);
                if (inflight_fences_[i] != VK_NULL_HANDLE) vkDestroyFence(device_, inflight_fences_[i], nullptr);
                image_available_[i] = VK_NULL_HANDLE;
                render_finished_[i] = VK_NULL_HANDLE;
                inflight_fences_[i] = VK_NULL_HANDLE;
            }
            if (device_ != VK_NULL_HANDLE)
            {
                vkDestroyDevice(device_, nullptr);
                device_ = VK_NULL_HANDLE;
            }
            if (surface_ != VK_NULL_HANDLE)
            {
                vkDestroySurfaceKHR(instance_, surface_, nullptr);
                surface_ = VK_NULL_HANDLE;
            }
            if (debug_messenger_ != VK_NULL_HANDLE)
            {
                auto destroy_fn = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT");
                if (destroy_fn) destroy_fn(instance_, debug_messenger_, nullptr);
                debug_messenger_ = VK_NULL_HANDLE;
            }
            if (instance_ != VK_NULL_HANDLE)
            {
                vkDestroyInstance(instance_, nullptr);
                instance_ = VK_NULL_HANDLE;
            }
            qf_ = QueueFamilies{};
            gpu_ = VK_NULL_HANDLE;
            graphics_q_ = VK_NULL_HANDLE;
            present_q_ = VK_NULL_HANDLE;
            requested_width_ = 0;
            requested_height_ = 0;
            window_ = nullptr;
            layers_.clear();
            initialized_ = false;
            init_attempted_ = false;
            resize_pending_ = false;
            swapchain_needs_rebuild_ = false;
            device_lost_ = false;
            current_frame_ = 0;
            swapchain_generation_ = 0;
            depth_format_ = VK_FORMAT_UNDEFINED;
            swapchain_usage_ = 0;
            capabilities_ = BackendCapabilities{};
            capabilities_ready_ = false;
        }
#else
        bool ensure_initialized() { return false; }
        void shutdown()
        {
            initialized_ = false;
            init_attempted_ = false;
            capabilities_ = BackendCapabilities{};
            capabilities_ready_ = false;
        }
#endif

    private:
        bool initialized_ = false;
        bool init_attempted_ = false;
        BackendCapabilities capabilities_{};
        bool capabilities_ready_ = false;
#ifdef SHS_HAS_VULKAN
        static constexpr uint32_t kMaxFramesInFlight = 2;

        SDL_Window* window_ = nullptr;
        bool enable_validation_ = false;
        bool resize_pending_ = false;
        bool swapchain_needs_rebuild_ = false;
        bool device_lost_ = false;
        int requested_width_ = 0;
        int requested_height_ = 0;
        std::string app_name_ = "shs-renderer-lib";
        std::vector<const char*> layers_{};

        VkInstance instance_ = VK_NULL_HANDLE;
        VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
        VkSurfaceKHR surface_ = VK_NULL_HANDLE;
        VkPhysicalDevice gpu_ = VK_NULL_HANDLE;
        QueueFamilies qf_{};
        VkDevice device_ = VK_NULL_HANDLE;
        VkQueue graphics_q_ = VK_NULL_HANDLE;
        VkQueue present_q_ = VK_NULL_HANDLE;
        VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
        VkFormat swapchain_format_ = VK_FORMAT_UNDEFINED;
        VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
        VkImageUsageFlags swapchain_usage_ = 0;
        VkExtent2D extent_{};
        std::vector<VkImage> images_{};
        std::vector<VkImageView> views_{};
        VkImage depth_image_ = VK_NULL_HANDLE;
        VkDeviceMemory depth_memory_ = VK_NULL_HANDLE;
        VkImageView depth_view_ = VK_NULL_HANDLE;
        VkRenderPass render_pass_ = VK_NULL_HANDLE;
        std::vector<VkFramebuffer> framebuffers_{};
        VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> cmd_bufs_{};
        std::vector<VkFence> images_in_flight_{};
        VkSemaphore image_available_[kMaxFramesInFlight]{VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkSemaphore render_finished_[kMaxFramesInFlight]{VK_NULL_HANDLE, VK_NULL_HANDLE};
        VkFence inflight_fences_[kMaxFramesInFlight]{VK_NULL_HANDLE, VK_NULL_HANDLE};
        uint64_t current_frame_ = 0;
        uint64_t swapchain_generation_ = 0;
#endif
    };
}
