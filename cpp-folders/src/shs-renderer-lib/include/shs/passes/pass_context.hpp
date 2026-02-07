/*

    shs/passes/pass_context.hpp

    PASS CONTEXT (нэг кадрын нийтлэг төлөв / параметрүүд)

    ЗОРИЛГО:
    - Pass бүр өөр өөр параметр авдаг байдлыг зогсооно.
    - Бүх pass-д хэрэгтэй нийтлэг зүйлсийг нэг газар төвлөрүүлнэ.
    - Demo бүрийн main() нь цэвэрхэн болно: 
        update(ctx) -> graph.execute(pc).

*/

#pragma once

#include <cstdint>

namespace shs
{
    // ---------------------------------------------
    // Forward declarations (хүнд include-ээс зайлсхийх)
    // ---------------------------------------------
    struct Context;

    struct PassContext
    {
        // --- App / Engine context ---
        Context* ctx = nullptr;          // Цонх, input, timing, device гэх мэт (demo-гийн үндсэн context)

        // --- Frame timing ---
        uint64_t frame_index = 0;        // Кадрын дугаар
        float    dt          = 0.0f;     // delta time (сек)

        // --- Camera ---
        // Энд матриц/векторын төрлийг оруулж болно (glm::mat4 гэх мэт),
        // гэхдээ одоогоор include тэсрэхээс хамгаалж void* хийж байна.
        // Дараа нь glm-г төв math.hpp дээр нэгтгээд хатуу төрөл болгоно.
        const void* view          = nullptr; // (const glm::mat4*)
        const void* proj          = nullptr; // (const glm::mat4*)
        const void* viewproj      = nullptr; // (const glm::mat4*)
        const void* prev_viewproj = nullptr; // (const glm::mat4*)

        const void* cam_pos_ws    = nullptr; // (const glm::vec3*)
        const void* sun_dir_ws    = nullptr; // (const glm::vec3*)

        // --- Common post params ---
        float exposure = 1.0f;           // Tonemap exposure
        float gamma    = 2.2f;           // Display gamma

        // --- Shared resource hubs (дараагийн header-уудтай холбоно) ---
        void* resources = nullptr;       // (shs::RendererResources*) - resource_handles.hpp дээр тодорхойлно
        void* scene     = nullptr;       // (shs::SceneData*)         - scene_data.hpp дээр тодорхойлно

        // --- Debug knobs (optional) ---
        int debug_view = 0;              // 0=final, 1=shadow, 2=depth, ... гэх мэт
    };
} // namespace shs
