#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: scene_types.hpp
    МОДУЛЬ: scene
    ЗОРИЛГО: Орчин үеийн (modern) пасс болон пайплайн (pipeline) API-уудад зориулсан
            үндсэн рендеринг ба ертөнцийн (scene) гэрээ болон төрлүүд.
*/


#include <cstdint>
#include <vector>
#include <glm/glm.hpp>

namespace shs
{
    // Орчин үеийн пасс болон пайплайнуудын ашиглах ертөнцийн үндсэн бүтэц.
    class ISkyModel;
    class ResourceRegistry;
    struct LightSet;

    // ------------------------------------------
    // Хөнгөн handle-ууд (demo бүр өөрийн asset системтэй байж болно)
    // ------------------------------------------
    using MeshHandle     = uint32_t;
    using MaterialHandle = uint32_t;

    // ------------------------------------------
    // Transform
    // ------------------------------------------
    struct Transform
    {
        glm::vec3 pos{0.0f};
        glm::vec3 rot_euler{0.0f}; // radians, demo-д тааруулж өөрчилж болно
        glm::vec3 scl{1.0f};
    };

    // ------------------------------------------
    // Камер
    // ------------------------------------------
    struct Camera
    {
        glm::vec3 pos{0.0f, 0.0f, -3.0f};
        glm::vec3 target{0.0f, 0.0f, 0.0f};
        glm::vec3 up{0.0f, 1.0f, 0.0f};

        float fov_y_radians = glm::radians(60.0f);
        float znear = 0.1f;
        float zfar  = 200.0f;

        // Pass-ууд шууд ашиглах камерын матрицууд.
        glm::mat4 view{1.0f};
        glm::mat4 proj{1.0f};
        glm::mat4 viewproj{1.0f};

        // Motion blur / velocity-д хэрэгтэй
        glm::mat4 prev_viewproj{1.0f};
    };

    // ------------------------------------------
    // Нарны гэрэл (Directional)
    // ------------------------------------------
    struct DirectionalLight
    {
        glm::vec3 dir_ws = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.2f));
        glm::vec3 color  = glm::vec3(1.0f);
        float intensity  = 5.0f;

        // Shadow pass-д хэрэгтэй
        glm::mat4 light_viewproj{1.0f};
    };

    // ------------------------------------------
    // Scene render item
    // ------------------------------------------
    struct RenderItem
    {
        Transform tr{};
        MeshHandle mesh = 0;
        MaterialHandle mat = 0;
        uint64_t object_id = 0;

        bool casts_shadow = true;
        bool visible = true;
    };

    // ------------------------------------------
    // Нийт scene
    // ------------------------------------------
    struct Scene
    {
        Camera cam{};
        DirectionalLight sun{};
        std::vector<RenderItem> items{};
        // Optional local light set (point/spot/area) used by light-culling passes.
        const LightSet* local_lights = nullptr;

        // Skybox-т хэрэгтэй handle (demo-д өөрийнхөөрөө ашиглана)
        uint32_t skybox_tex = 0;
        const ISkyModel* sky = nullptr;
        const ResourceRegistry* resources = nullptr;
    };
}
