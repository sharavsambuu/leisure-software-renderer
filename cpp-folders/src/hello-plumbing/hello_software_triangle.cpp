#define SDL_MAIN_HANDLED

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include <shs/core/context.hpp>
#include <shs/gfx/rt_types.hpp>
#include <shs/platform/platform_input.hpp>
#include <shs/platform/platform_runtime.hpp>
#include <shs/platform/sdl/sdl_runtime.hpp>
#include <shs/render/rasterizer.hpp>
#include <shs/resources/mesh.hpp>
#include <shs/rhi/backend/backend_factory.hpp>
#include <shs/shader/program.hpp>
#include <shs/shader/types.hpp>

namespace
{
constexpr int kWindowW = 960;
constexpr int kWindowH = 640;
constexpr int kSurfaceW = 960;
constexpr int kSurfaceH = 640;

void upload_hdr_to_rgba8(std::vector<uint8_t>& rgba, const shs::RT_ColorHDR& hdr)
{
    rgba.resize(static_cast<size_t>(hdr.w) * static_cast<size_t>(hdr.h) * 4u);
    for (int y_screen = 0; y_screen < hdr.h; ++y_screen)
    {
        const int y_canvas = hdr.h - 1 - y_screen;
        uint8_t* row = rgba.data() + static_cast<size_t>(y_screen) * static_cast<size_t>(hdr.w) * 4u;
        for (int x = 0; x < hdr.w; ++x)
        {
            const shs::ColorF c = hdr.color.at(x, y_canvas);
            const float r = std::clamp(c.r, 0.0f, 1.0f);
            const float g = std::clamp(c.g, 0.0f, 1.0f);
            const float b = std::clamp(c.b, 0.0f, 1.0f);
            const int i = x * 4;
            row[i + 0] = static_cast<uint8_t>(std::lround(r * 255.0f));
            row[i + 1] = static_cast<uint8_t>(std::lround(g * 255.0f));
            row[i + 2] = static_cast<uint8_t>(std::lround(b * 255.0f));
            row[i + 3] = 255u;
        }
    }
}

class HelloSoftwareTriangleApp
{
public:
    void run()
    {
        init_runtime();
        init_backend();
        init_triangle_pipeline();
        main_loop();
    }

private:
    void init_runtime()
    {
        shs::WindowDesc win{};
        win.title = "HelloSoftwareTriangle";
        win.width = kWindowW;
        win.height = kWindowH;

        shs::SurfaceDesc surface{};
        surface.width = kSurfaceW;
        surface.height = kSurfaceH;

        runtime_ = std::make_unique<shs::SdlRuntime>(win, surface);
        if (!runtime_ || !runtime_->valid())
        {
            throw std::runtime_error("SdlRuntime init failed");
        }
    }

    void init_backend()
    {
        shs::RenderBackendCreateResult created = shs::create_render_backend(shs::RenderBackendType::Software);
        if (!created.note.empty())
        {
            std::fprintf(stderr, "[shs] %s\n", created.note.c_str());
        }
        if (!created.backend)
        {
            throw std::runtime_error("Backend factory did not return backend");
        }

        keep_.push_back(std::move(created.backend));
        for (auto& aux : created.auxiliary_backends)
        {
            if (aux) keep_.push_back(std::move(aux));
        }
        for (auto& b : keep_)
        {
            ctx_.register_backend(b.get());
        }

        sw_ = ctx_.backend(shs::RenderBackendType::Software);
        if (!sw_)
        {
            throw std::runtime_error("Software backend is not registered");
        }
        ctx_.set_primary_backend(sw_);
        std::fprintf(stderr, "[shs] active backend: %s\n", ctx_.active_backend_name());
    }

    void init_triangle_pipeline()
    {
        triangle_.positions = {
            {0.0f, 0.6f, 0.0f},
            {0.6f, -0.6f, 0.0f},
            {-0.6f, -0.6f, 0.0f}
        };
        // MeshData-д тусдаа vertex color талбар байхгүй тул demo color-ийг normals-д шингээж дамжуулна.
        triangle_.normals = {
            {1.0f, 0.2f, 0.2f},
            {0.2f, 1.0f, 0.2f},
            {0.2f, 0.4f, 1.0f}
        };
        triangle_.indices = {0, 1, 2};

        program_.vs = [](const shs::ShaderVertex& vin, const shs::ShaderUniforms&) -> shs::VertexOut {
            shs::VertexOut o{};
            o.clip = glm::vec4(vin.position, 1.0f);
            o.world_pos = vin.position;
            o.normal_ws = glm::vec3(0.0f, 0.0f, 1.0f);
            shs::set_varying(o, shs::VaryingSemantic::Color0, glm::vec4(vin.normal, 1.0f));
            return o;
        };
        program_.fs = [](const shs::FragmentIn& fin, const shs::ShaderUniforms&) -> shs::FragmentOut {
            const glm::vec4 c = shs::get_varying(fin, shs::VaryingSemantic::Color0, glm::vec4(1.0f));
            shs::FragmentOut out{};
            out.color = shs::ColorF{
                std::clamp(c.r, 0.0f, 1.0f),
                std::clamp(c.g, 0.0f, 1.0f),
                std::clamp(c.b, 0.0f, 1.0f),
                1.0f
            };
            return out;
        };

        rast_cfg_.cull_mode = shs::RasterizerCullMode::None;
        rast_cfg_.front_face_ccw = true;
    }

    void main_loop()
    {
        bool running = true;
        while (running)
        {
            shs::PlatformInputState input{};
            running = runtime_->pump_input(input);
            if (!running || input.quit) break;
            draw_frame();
        }
    }

    void draw_frame()
    {
        shs::RenderBackendFrameInfo frame{};
        frame.frame_index = ctx_.frame_index;
        frame.width = color_hdr_.w;
        frame.height = color_hdr_.h;
        sw_->begin_frame(ctx_, frame);

        color_hdr_.clear({0.04f, 0.05f, 0.09f, 1.0f});

        shs::RasterizerTarget target{};
        target.hdr = &color_hdr_;
        const shs::RasterizerStats stats = shs::rasterize_mesh(triangle_, program_, uniforms_, target, rast_cfg_);
        ctx_.debug.tri_input = stats.tri_input;
        ctx_.debug.tri_after_clip = stats.tri_after_clip;
        ctx_.debug.tri_raster = stats.tri_raster;

        upload_hdr_to_rgba8(rgba_staging_, color_hdr_);
        runtime_->upload_rgba8(rgba_staging_.data(), color_hdr_.w, color_hdr_.h, color_hdr_.w * 4);
        runtime_->present();

        sw_->end_frame(ctx_, frame);
        ++ctx_.frame_index;
    }

private:
    shs::Context ctx_{};
    std::vector<std::unique_ptr<shs::IRenderBackend>> keep_{};
    shs::IRenderBackend* sw_ = nullptr;
    std::unique_ptr<shs::SdlRuntime> runtime_{};

    shs::MeshData triangle_{};
    shs::ShaderProgram program_{};
    shs::ShaderUniforms uniforms_{};
    shs::RasterizerConfig rast_cfg_{};

    shs::RT_ColorHDR color_hdr_{kSurfaceW, kSurfaceH};
    std::vector<uint8_t> rgba_staging_{};
};
}

int main()
{
    try
    {
        HelloSoftwareTriangleApp app{};
        app.run();
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "Fatal: %s\n", e.what());
        return 1;
    }
}
