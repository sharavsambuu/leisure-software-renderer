/*
    3D Software Renderer - Flat Shading (XSIMD + Tiled + Perspective Correct Depth)

    ЗОРИЛГО:
    XSIMD санг ашиглан SIMD-ээр Flat Shading демог хурдасгаж харьцуулж харах, бараг 2 дахин шахуу хурдан
    - Flat shading + Z-buffer
    - Tiled binning (tile бүр дээр tri жагсаалт)
    - XSIMD ашиглан contiguous SIMD (load/compare/select/store) хийж хурд авах
    - Scatter / per-lane scalar loop-оос зайлсхийх (гол speedup)

    ГОЛ САНАА
    - Z update: batch-аар хуучин z-г load -> шинэ z-г compare -> select -> store (contiguous)
    - Color update: RGBA-г uint32 болгон pack хийгээд batch-аар select/store (contiguous)
    - Tile-row бүрт 1 job: job submit overhead багасна (300 job/frame биш)
    - Edge function row хэсгийг урьдчилж бодож ашиглана (B*y + C)

    
    - Perspective Correct Depth: энд depth утга нь ndc.z биш, харин -(w0*invW0 + w1*invW1 + w2*invW2) хэлбэртэй.
*/

#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <queue>
#include <iostream>
#include <vector>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <thread>

// Гадаад сангууд
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <xsimd/xsimd.hpp>

#include "shs_renderer.hpp"

// Тохиргоо
#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480
#define TILE_SIZE         32

// ==========================================
// DATA STRUCTURES
// ==========================================

struct TriProcessed {
    // Edge Coefficients (Ax + By + C)
    float A0, B0, C0;
    float A1, B1, C1;
    float A2, B2, C2;

    // Perspective Correct Depth: 1/w хадгална
    float inv_w0, inv_w1, inv_w2;

    float inv_area;
    int min_x, min_y, max_x, max_y;

    // Flat shading өнгө
    shs::Color color;
};

struct TileBin {
    std::vector<int> tri_indices;
};

// ==========================================
// SCENE CLASSES
// ==========================================

// Using standardized shs::Viewer
using Viewer = shs::Viewer;

// Using standardized shs::ModelGeometry
using ModelGeometry = shs::ModelGeometry;

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 position, glm::vec3 scale) {
        this->position       = position;
        this->scale          = scale;
        this->geometry       = new ModelGeometry("./obj/monkey/monkey.rawobj");
        this->rotation_angle = 0.0f;
    }
    ~MonkeyObject() { delete geometry; }

    glm::mat4 get_world_matrix() override {
        glm::mat4 T = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
        return T * R * S;
    }

    void update(float delta_time) override {
        this->rotation_angle += 30.0f * delta_time;
        if (this->rotation_angle >= 360.0f) this->rotation_angle -= 360.0f;
    }
    void render() override {}

    ModelGeometry *geometry = nullptr;
    glm::vec3 scale         = glm::vec3(1.0f);
    glm::vec3 position      = glm::vec3(0.0f);
    float rotation_angle    = 0.0f;
};

class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::RT_ColorDepth *rt, Viewer *viewer) {
        this->rt     = rt;
        this->viewer = viewer;

        float step = 15.0f;
        for (int i=0; i<2; ++i)
            for (int j=0; j<2; ++j)
                scene_objects.push_back(new MonkeyObject(glm::vec3(i*step - 7.5f, 0.0f, j*step + 20.0f), glm::vec3(5.0f)));
    }
    ~HelloScene() {
        for (auto *obj : scene_objects) delete obj;
    }

    void process() override {}

    std::vector<shs::AbstractObject3D *> scene_objects;
    shs::RT_ColorDepth *rt     = nullptr;
    Viewer             *viewer = nullptr;

    // Гэрлийн чиглэл (world -> view болгож хэрэглэнэ)
    glm::vec3 light_direction = glm::vec3(1.0f, 0.3f, 1.0f);
};

// ==========================================
// RENDERER SYSTEM (XSIMD + TILING + PC DEPTH)
// ==========================================

class RendererSystemXSIMD : public shs::AbstractSystem
{
public:
    RendererSystemXSIMD(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *jobs)
        : scene(scene), jobs(jobs)
    {
        tiles_x = (CANVAS_WIDTH  + TILE_SIZE - 1) / TILE_SIZE;
        tiles_y = (CANVAS_HEIGHT + TILE_SIZE - 1) / TILE_SIZE;

        tile_bins.resize(tiles_x * tiles_y);
        processed_tris.reserve(100000);
    }

    void process(float /*delta_time*/) override
    {
        // 1) CLEAR (Color+Depth)
        scene->rt->clear(shs::Color::black());

        // Bin болон tri list-ээ цэвэрлэх
        for (auto &bin : tile_bins) bin.tri_indices.clear();
        processed_tris.clear();

        const glm::mat4 V = scene->viewer->camera->view_matrix;
        const glm::mat4 P = scene->viewer->camera->projection_matrix;

        // Light dir: world -> view
        glm::vec3 light_dir_world = glm::normalize(scene->light_direction);
        glm::vec3 light_dir_view  = glm::normalize(glm::vec3(V * glm::vec4(light_dir_world, 0.0f)));

        const int screen_w = CANVAS_WIDTH;
        const int screen_h = CANVAS_HEIGHT;

        // 2) GEOMETRY PROCESSING + BINNING
        for (shs::AbstractObject3D *obj : scene->scene_objects)
        {
            MonkeyObject *monkey = dynamic_cast<MonkeyObject*>(obj);
            if (!monkey) continue;

            glm::mat4 MV   = V * monkey->get_world_matrix();
            glm::mat3 NMat = glm::transpose(glm::inverse(glm::mat3(MV)));

            const auto& verts = monkey->geometry->triangles;
            const auto& norms = monkey->geometry->normals;

            for (size_t i = 0; i < verts.size(); i += 3)
            {
                // Vertex shader (clip)
                glm::vec4 c0 = P * (MV * glm::vec4(verts[i],   1.0f));
                glm::vec4 c1 = P * (MV * glm::vec4(verts[i+1], 1.0f));
                glm::vec4 c2 = P * (MV * glm::vec4(verts[i+2], 1.0f));

                // Near plane cull (энгийн)
                if (c0.w <= 0.1f || c1.w <= 0.1f || c2.w <= 0.1f) continue;

                // Clip -> Screen
                auto to_screen = [&](const glm::vec4& c) {
                    glm::vec3 ndc = glm::vec3(c) / c.w;
                    return glm::vec3(
                        (ndc.x + 1.0f) * 0.5f * (screen_w - 1),
                        (1.0f - ndc.y) * 0.5f * (screen_h - 1),
                        ndc.z
                    );
                };

                glm::vec3 s0 = to_screen(c0);
                glm::vec3 s1 = to_screen(c1);
                glm::vec3 s2 = to_screen(c2);

                // Backface cull (screen space area)
                float area = (s1.x - s0.x) * (s2.y - s0.y) - (s2.x - s0.x) * (s1.y - s0.y);
                if (area <= 0.0f) continue;
                float inv_area = 1.0f / area;

                // Flat shading (face normal)
                glm::vec3 face_normal = glm::normalize(NMat * norms[i] + NMat * norms[i+1] + NMat * norms[i+2]);
                float diffuse         = std::max(0.0f, glm::dot(face_normal, light_dir_view));
                float total           = std::min(1.0f, 0.15f + diffuse);
                uint8_t c_val         = (uint8_t)(total * 255.0f);

                // Bounding box
                int min_x = std::max(0, (int)std::min({s0.x, s1.x, s2.x}));
                int min_y = std::max(0, (int)std::min({s0.y, s1.y, s2.y}));
                int max_x = std::min(screen_w, (int)std::max({s0.x, s1.x, s2.x}) + 1);
                int max_y = std::min(screen_h, (int)std::max({s0.y, s1.y, s2.y}) + 1);
                if (min_x >= max_x || min_y >= max_y) continue;

                // TriProcessed build
                TriProcessed tri;
                tri.color   = {c_val, c_val, c_val, 255};
                tri.min_x   = min_x; tri.min_y = min_y;
                tri.max_x   = max_x; tri.max_y = max_y;
                tri.inv_area = inv_area;

                // PC depth: 1/w хадгална
                tri.inv_w0 = 1.0f / c0.w;
                tri.inv_w1 = 1.0f / c1.w;
                tri.inv_w2 = 1.0f / c2.w;

                // Edge functions: Ax + By + C
                tri.A0 = s0.y - s1.y; tri.B0 = s1.x - s0.x; tri.C0 = s0.x * s1.y - s0.y * s1.x;
                tri.A1 = s1.y - s2.y; tri.B1 = s2.x - s1.x; tri.C1 = s1.x * s2.y - s1.y * s2.x;
                tri.A2 = s2.y - s0.y; tri.B2 = s0.x - s2.x; tri.C2 = s2.x * s0.y - s2.y * s0.x;

                int tri_idx = (int)processed_tris.size();
                processed_tris.push_back(tri);

                // Binning: tri BB -> tile indices
                int t_min_x = min_x / TILE_SIZE;
                int t_max_x = (max_x - 1) / TILE_SIZE;
                int t_min_y = min_y / TILE_SIZE;
                int t_max_y = (max_y - 1) / TILE_SIZE;

                for (int ty = t_min_y; ty <= t_max_y; ++ty) {
                    for (int tx = t_min_x; tx <= t_max_x; ++tx) {
                        tile_bins[ty * tiles_x + tx].tri_indices.push_back(tri_idx);
                    }
                }
            }
        }

        // 3) RASTER STAGE
        // Tile-row бүрийг нэг job болгоно (job overhead багасна)
        shs::Job::WaitGroup wg;
        wg.reset();
        wg.add(tiles_y);

        for (int ty = 0; ty < tiles_y; ++ty) {
            jobs->submit({
                [this, ty, &wg]() {
                    for (int tx = 0; tx < tiles_x; ++tx) {
                        this->rasterize_tile_xsimd(tx, ty);
                    }
                    wg.done();
                },
                shs::Job::PRIORITY_NORMAL
            });
        }
        wg.wait();
    }

private:
    void rasterize_tile_xsimd(int tx, int ty)
    {
        namespace xs = xsimd;
        using bf = xs::batch<float>;
        using bu = xs::batch<uint32_t>;

        // Tile bounds (screen space, y-down)
        int x_base     = tx * TILE_SIZE;
        int y_base     = ty * TILE_SIZE;
        int x_end_tile = std::min(x_base + TILE_SIZE, CANVAS_WIDTH);
        int y_end_tile = std::min(y_base + TILE_SIZE, CANVAS_HEIGHT);

        const auto& bin = tile_bins[ty * tiles_x + tx];
        if (bin.tri_indices.empty()) return;

        // Буферүүд (Canvas is bottom-up; row_offset дээр y-г хөрвүүлнэ)
        shs::Color* c_buf_raw = scene->rt->color.buffer().raw();
        float*      z_buf_raw = scene->rt->depth.buffer().raw();

        const int width  = CANVAS_WIDTH;
        const int height = CANVAS_HEIGHT;

        const int LANES  = bf::size;

        // iota = [0.5..] — pixel center offset
        static const bf iota_f = [](){
            alignas(64) float tmp[64];
            for (int i = 0; i < 64; ++i) tmp[i] = (float)i + 0.5f;
            return bf::load_aligned(tmp);
        }();

        static_assert(sizeof(shs::Color) == 4, "shs::Color must be 4 bytes to pack to u32");

        auto pack_rgba_u32 = [](const shs::Color& c) -> uint32_t {
            return (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
        };

        // triangle бүрээр давтаж, tri const-уудыг регистрт барина
        for (int idx : bin.tri_indices)
        {
            const TriProcessed& tri = processed_tris[idx];

            // BB ∩ Tile
            int ix0 = std::max(x_base, tri.min_x);
            int ix1 = std::min(x_end_tile, tri.max_x);
            int iy0 = std::max(y_base, tri.min_y);
            int iy1 = std::min(y_end_tile, tri.max_y);
            if (ix0 >= ix1 || iy0 >= iy1) continue;

            // Register cache (scalar -> batch)
            bf A0(tri.A0), B0(tri.B0), C0(tri.C0);
            bf A1(tri.A1), B1(tri.B1), C1(tri.C1);
            bf A2(tri.A2), B2(tri.B2), C2(tri.C2);

            bf invArea(tri.inv_area);
            bf invW0(tri.inv_w0), invW1(tri.inv_w1), invW2(tri.inv_w2);

            const uint32_t packed = pack_rgba_u32(tri.color);
            bu color_vec(packed);

            for (int y = iy0; y < iy1; ++y)
            {
                bf y_vec((float)y + 0.5f);

                // Row constants: (B*y + C)
                bf row_E0 = B0 * y_vec + C0;
                bf row_E1 = B1 * y_vec + C1;
                bf row_E2 = B2 * y_vec + C2;

                // Canvas y хөрвүүлэлт (screen y-down -> canvas y-up)
                int canvas_y   = (height - 1) - y;
                int row_offset = canvas_y * width;

                int x = ix0;

                // SIMD: contiguous load/store
                for (; x + LANES <= ix1; x += LANES)
                {
                    bf x_vec = bf((float)x) + iota_f;

                    // Edge functions (3)
                    bf E0 = A0 * x_vec + row_E0;
                    bf E1 = A1 * x_vec + row_E1;
                    bf E2 = A2 * x_vec + row_E2;

                    auto inside = (E0 >= 0.0f) & (E1 >= 0.0f) & (E2 >= 0.0f);
                    if (!xs::any(inside)) continue;

                    // Barycentric (w mapping-тэй ижил)
                    bf w0 = E1 * invArea;
                    bf w1 = E2 * invArea;
                    bf w2 = E0 * invArea;

                    // Perspective Correct Depth proxy
                    bf z_new = -(w0 * invW0 + w1 * invW1 + w2 * invW2);

                    float* z_ptr = z_buf_raw + row_offset + x;

                    // Z compare/select/store (contiguous)
                    bf z_old = bf::load_unaligned(z_ptr);
                    auto pass = inside & (z_new < z_old);
                    if (!xs::any(pass)) continue;

                    bf z_out = xs::select(pass, z_new, z_old);
                    z_out.store_unaligned(z_ptr);

                    // Color compare/select/store (contiguous, u32)
                    uint32_t* c_ptr = reinterpret_cast<uint32_t*>(c_buf_raw + row_offset + x);

                    bu c_old = bu::load_unaligned(c_ptr);

                    // pass mask-ыг integer bitmask болгож bitwise blend хийнэ
                    bu m = xs::bitwise_cast<bu>(pass);
                    bu c_out = (m & color_vec) | (~m & c_old);
                    c_out.store_unaligned(c_ptr);
                }

                // Tail scalar
                for (; x < ix1; ++x)
                {
                    float fx = (float)x + 0.5f;
                    float fy = (float)y + 0.5f;

                    float e0 = tri.A0 * fx + tri.B0 * fy + tri.C0;
                    float e1 = tri.A1 * fx + tri.B1 * fy + tri.C1;
                    float e2 = tri.A2 * fx + tri.B2 * fy + tri.C2;

                    if (e0 >= 0 && e1 >= 0 && e2 >= 0)
                    {
                        float w0s = e1 * tri.inv_area;
                        float w1s = e2 * tri.inv_area;
                        float w2s = e0 * tri.inv_area;

                        float z = -(w0s * tri.inv_w0 + w1s * tri.inv_w1 + w2s * tri.inv_w2);

                        int pixel_idx = row_offset + x;
                        if (z < z_buf_raw[pixel_idx])
                        {
                            z_buf_raw[pixel_idx] = z;
                            c_buf_raw[pixel_idx] = tri.color;
                        }
                    }
                }
            }
        }
    }

    HelloScene                          *scene = nullptr;
    shs::Job::ThreadedPriorityJobSystem *jobs  = nullptr;

    int tiles_x = 0;
    int tiles_y = 0;

    std::vector<TileBin>      tile_bins;
    std::vector<TriProcessed> processed_tris;
};

// ==========================================
// SYSTEM WRAPPERS
// ==========================================

class LogicSystem : public shs::AbstractSystem {
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override {
        scene->viewer->update();
        for (auto *obj : scene->scene_objects) obj->update(delta_time);
    }
private:
    HelloScene *scene = nullptr;
};

class SystemProcessor {
public:
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *jobs) {
        cmd_proc     = new shs::CommandProcessor();
        logic_sys    = new LogicSystem(scene);
        renderer_sys = new RendererSystemXSIMD(scene, jobs);
    }
    ~SystemProcessor() { delete cmd_proc; delete logic_sys; delete renderer_sys; }

    void process(float dt) { 
        cmd_proc->process(); 
        logic_sys->process(dt); 
    }
    void render(float dt)  { 
        renderer_sys->process(dt); 
    }

    shs::CommandProcessor *cmd_proc     = nullptr;
    LogicSystem           *logic_sys    = nullptr;
    RendererSystemXSIMD   *renderer_sys = nullptr;
};

static void copy_canvas_to_sdl(SDL_Surface *surface, const shs::Canvas &canvas) {
    uint32_t* dst_pixels         = (uint32_t*)surface->pixels;
    const shs::Color* src_pixels = canvas.buffer().raw();
    int w = canvas.get_width();
    int h = canvas.get_height();
    for (int y = 0; y < h; ++y) {
        int src_y = (h - 1) - y;
        const shs::Color* src_row = src_pixels + src_y * w;
        uint32_t* dst_row = dst_pixels + y * w;
        for (int x = 0; x < w; ++x) {
            shs::Color c = src_row[x];
            dst_row[x] = (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
        }
    }
}

// ==========================================
// MAIN
// ==========================================

int main(int argc, char* argv[])
{
    // XSIMD аль ISA-г сонгосныг шалгах (AVX2 бол batch<float>::size=8 байх ёстой)
    std::cout << "XSIMD arch: " << xsimd::default_arch::name()
              << " | batch<float>::size=" << xsimd::batch<float>::size << "\n";

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window *window = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::RT_ColorDepth *rt      = new shs::RT_ColorDepth(CANVAS_WIDTH, CANVAS_HEIGHT, 0.1f, 1000.0f);
    SDL_Surface *screen_surface = rt->color.create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, screen_surface);

    // Threads: бүх цөмийг ашиглаж болно, SDL дээр лаг бага байлгахын тулд бага зэрэг үлдээв
    int cores = (int)std::thread::hardware_concurrency();
    if (cores == 0) cores  = 4;
    if (cores > 2 ) cores -= 2;
    if (cores < 1 ) cores  = 1;

    shs::Job::ThreadedPriorityJobSystem *jobs = new shs::Job::ThreadedPriorityJobSystem(cores);

    Viewer          *viewer = new Viewer(glm::vec3(0.0f, 10.0f, -50.0f), 150.0f, CANVAS_WIDTH, CANVAS_HEIGHT);
    HelloScene      *scene  = new HelloScene(rt, viewer);
    SystemProcessor *sys    = new SystemProcessor(scene, jobs);

    bool exit = false;
    SDL_Event e;

    Uint32 last_time = SDL_GetTicks();
    int frames = 0;
    float fps_timer = 0.0f;

    while (!exit) {
        Uint32 current_time = SDL_GetTicks();
        float dt = (current_time - last_time) / 1000.0f;
        last_time = current_time;

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) exit = true;
            if (e.type == SDL_KEYDOWN) {
                switch(e.key.keysym.sym) {
                    case SDLK_ESCAPE: exit = true; break;
                    case SDLK_w: sys->cmd_proc->add_command(new shs::MoveForwardCommand (viewer->position, viewer->get_direction_vector(), viewer->speed, dt)); break;
                    case SDLK_s: sys->cmd_proc->add_command(new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, dt)); break;
                    case SDLK_a: sys->cmd_proc->add_command(new shs::MoveLeftCommand    (viewer->position, viewer->get_right_vector()    , viewer->speed, dt)); break;
                    case SDLK_d: sys->cmd_proc->add_command(new shs::MoveRightCommand   (viewer->position, viewer->get_right_vector()    , viewer->speed, dt)); break;
                }
            }
        }

        sys->process(dt);
        sys->render(dt);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        copy_canvas_to_sdl(screen_surface, rt->color);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        frames++;
        fps_timer += dt;
        if (fps_timer >= 1.0f) {
            std::string title = "Flat Shading + XSIMD | FPS: " + std::to_string(frames) +
                                " | Threads: " + std::to_string(cores) +
                                " | Tile: " + std::to_string((int)TILE_SIZE);
            SDL_SetWindowTitle(window, title.c_str());
            frames = 0;
            fps_timer = 0.0f;
        }
    }

    delete sys;
    delete scene;
    delete viewer;
    delete rt;
    delete jobs;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(screen_surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
