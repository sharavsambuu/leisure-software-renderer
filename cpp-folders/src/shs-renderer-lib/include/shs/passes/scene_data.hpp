/*

    shs/passes/scene_data.hpp

    SCENE DATA (pass-уудын нийтлэг scene input)

    ЗОРИЛГО:
    - Demo бүр pass бүрт өөр өөр parameter дамжуулахгүй.
    - Scene-ийн үндсэн өгөгдөл (camera/sun/objects/material refs) нэг газар байна.
    - Heavy include-ийг багасгахын тулд handle/pointer маягаар холбоно.

*/

#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace shs
{
    // ---------------------------------------------
    // Forward declarations: renderer core types
    // ---------------------------------------------
    struct Mesh;
    struct Texture2D;

    // ---------------------------------------------
    // Camera / Sun
    // ---------------------------------------------
    struct CameraData
    {
        glm::vec3 pos_ws   = {0.0f, 1.5f, -5.0f};
        glm::vec3 fwd_ws   = {0.0f, 0.0f,  1.0f};
        glm::vec3 up_ws    = {0.0f, 1.0f,  0.0f};

        float fov_y_rad    = glm::radians(60.0f);
        float znear        = 0.1f;
        float zfar         = 1000.0f;

        glm::mat4 view     = glm::mat4(1.0f);
        glm::mat4 proj     = glm::mat4(1.0f);
        glm::mat4 viewproj = glm::mat4(1.0f);
        glm::mat4 prev_viewproj = glm::mat4(1.0f);

        inline void rebuild_view()
        {
            view = glm::lookAt(pos_ws, pos_ws + fwd_ws, up_ws);
        }

        inline void rebuild_proj(float aspect)
        {
            proj = glm::perspective(fov_y_rad, aspect, znear, zfar);
        }

        inline void rebuild_viewproj()
        {
            viewproj = proj * view;
        }

        inline void begin_frame()
        {
            prev_viewproj = viewproj;
        }
    };

    struct SunData
    {
        glm::vec3 dir_ws   = glm::normalize(glm::vec3(-0.35f, -1.0f, -0.25f)); // нар "доош" чиглэнэ
        glm::vec3 color    = {1.0f, 1.0f, 1.0f};
        float     intensity = 5.0f;
    };

    // ---------------------------------------------
    // Materials (PBR minimal)
    // ---------------------------------------------
    struct MaterialPBR
    {
        // Base params
        glm::vec3 base_color = {1.0f, 1.0f, 1.0f};
        float metallic       = 0.0f;
        float roughness      = 0.6f;
        float ao             = 1.0f;

        // Texture refs (optional)
        const Texture2D* base_color_tex = nullptr;
        const Texture2D* normal_tex     = nullptr;
        const Texture2D* mr_tex         = nullptr; // metallic-roughness (эсвэл roughness-metallic)
        const Texture2D* ao_tex         = nullptr;
        const Texture2D* emissive_tex   = nullptr;

        glm::vec3 emissive_color = {0.0f, 0.0f, 0.0f};
        float     emissive_intensity = 0.0f;

        // Flags / conventions
        bool mr_is_roughness_in_g = true; // pipeline-д тааруулах
        bool normal_y_flip        = false;
    };

    // ---------------------------------------------
    // Render item (mesh + material + transform)
    // ---------------------------------------------
    struct RenderItem
    {
        const Mesh* mesh = nullptr;
        MaterialPBR mat  = {};

        glm::mat4 model = glm::mat4(1.0f);

        // Optional: object id / layer / cast-shadow flags
        uint32_t object_id   = 0;
        bool cast_shadow     = true;
        bool receive_shadow  = true;
    };

    // ---------------------------------------------
    // Environment (skybox / IBL)
    // ---------------------------------------------
    struct EnvironmentData
    {
        // Skybox (LDR cubemap)
        const Texture2D* sky_cubemap = nullptr;

        // IBL (diffuse irradiance, prefiltered specular, BRDF LUT)
        const Texture2D* ibl_irradiance = nullptr;
        const Texture2D* ibl_prefilter  = nullptr;
        const Texture2D* ibl_brdf_lut   = nullptr;

        float sky_intensity = 1.0f;
    };

    // ---------------------------------------------
    // SceneData: pass-уудын унших "нэг цэг"
    // ---------------------------------------------
    struct SceneData
    {
        CameraData camera;
        SunData    sun;
        EnvironmentData env;

        std::vector<RenderItem> items;

        // Common toggles
        bool enable_skybox   = true;
        bool enable_ibl      = true;
        bool enable_shadows  = true;

        inline void clear_items()
        {
            items.clear();
        }
    };

} // namespace shs
