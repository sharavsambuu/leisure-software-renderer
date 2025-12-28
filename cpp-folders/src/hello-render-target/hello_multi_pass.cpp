/*
    3D Software Renderer - MULTI-PASS PING-PONG (Outline + Fog) + 9 Monkeys (3x3)
    - Pass 0: Render scene into RT_ColorDepth (Color + Depth)
    - Pass 1: Outline (depth-based)  : rt_scene.color + rt_scene.depth -> ping
    - Pass 2: Fog (depth-based)      : ping + rt_scene.depth -> pong
    - Present: pong -> SDL


    NOTE : (My coordinate system convention):
    - Screen space origin: top-left (SDL)
    - shs::Canvas origin : bottom-left
    - Depth is stored in CANVAS coordinates (bottom-left) to avoid the inverting bug.
*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>

// External Libraries
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

#define WINDOW_WIDTH      840
#define WINDOW_HEIGHT     720
#define CANVAS_WIDTH      840
#define CANVAS_HEIGHT     720
#define MOUSE_SENSITIVITY 0.2f
#define THREAD_COUNT      20
#define TILE_SIZE_X       40
#define TILE_SIZE_Y       40

// ===============================
// 9 MONKEYS (3x3) CONFIG
// ===============================
#define GRID_X            3
#define GRID_Z            3
#define MONKEY_SCALE      3.2f
#define SPACING_X         7.5f
#define SPACING_Z         9.0f
#define START_Z           10.0f
#define BASE_Y            0.0f

// ===============================
// OUTLINE PASS CONFIG
// ===============================
//#define EDGE_THRESHOLD    0.55f
//#define EDGE_STRENGTH     0.45f
//#define EDGE_RADIUS       1
static const int   EDGE_RADIUS    = 1;       // 1 or 2
static const float EDGE_THRESHOLD = 0.35f;   // smaller = more edges
static const float EDGE_STRENGTH  = 0.55f;   // 0..1 (higher = darker lines)

// ===============================
// FOG PASS CONFIG
// ===============================
//#define FOG_START_Z       18.0f
//#define FOG_END_Z         55.0f
//static const shs::Color FOG_COLOR = shs::Color{ 22, 24, 30, 255 };
static const shs::Color FOG_COLOR = shs::Color{ 28, 30, 38, 255 };
static const float FOG_START_Z    = 14.0f;
static const float FOG_END_Z      = 55.0f;
static const float FOG_POWER      = 1.25f;   // 1..2 (higher = stronger near end)

// Outline

// ==========================================
// SMALL HELPERS
// ==========================================

static inline int clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float clampf(float v, float lo, float hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float smoothstep01(float t)
{
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline shs::Color lerp_color(const shs::Color& a, const shs::Color& b, float t)
{
    t = clampf(t, 0.0f, 1.0f);
    float ia = 1.0f - t;

    return shs::Color{
        (uint8_t)(ia * a.r + t * b.r),
        (uint8_t)(ia * a.g + t * b.g),
        (uint8_t)(ia * a.b + t * b.b),
        255
    };
}

static inline shs::Color monkey_color_from_i(int i)
{
    const int m = i % 6;
    if (m == 0) return shs::Color{  60, 100, 200, 255 };
    if (m == 1) return shs::Color{ 200,  90,  80, 255 };
    if (m == 2) return shs::Color{  80, 200, 120, 255 };
    if (m == 3) return shs::Color{ 210, 180,  80, 255 };
    if (m == 4) return shs::Color{ 180,  90, 210, 255 };
    return             shs::Color{  80, 180, 200, 255 };
}

// ==========================================
// UNIFORMS & SHADERS (Blinn-Phong)
// ==========================================

struct Uniforms {
    glm::mat4  mvp;
    glm::mat4  model;
    glm::mat4  view;
    glm::vec3  light_dir;
    glm::vec3  camera_pos;
    shs::Color color;
};

shs::Varyings blinn_phong_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;
    out.position  = u.mvp * glm::vec4(aPos, 1.0f);
    out.world_pos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal    = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv        = glm::vec2(0.0f);

    glm::vec4 view_pos = u.view * u.model * glm::vec4(aPos, 1.0f);
    out.view_z = view_pos.z; // my convention: forward is +z

    return out;
}

shs::Color blinn_phong_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    glm::vec3 norm     = glm::normalize(in.normal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - in.world_pos);

    float ambientStrength = 0.15f;
    glm::vec3 ambient     = ambientStrength * glm::vec3(1.0f);

    float diff        = glm::max(glm::dot(norm, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f);

    glm::vec3 halfwayDir   = glm::normalize(lightDir + viewDir);
    float specularStrength = 0.5f;
    float shininess        = 64.0f;

    float spec         = glm::pow(glm::max(glm::dot(norm, halfwayDir), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f);

    glm::vec3 objectColorVec = glm::vec3(u.color.r, u.color.g, u.color.b) / 255.0f;
    glm::vec3 result         = (ambient + diffuse + specular) * objectColorVec;

    result = glm::clamp(result, 0.0f, 1.0f);

    return shs::Color{
        (uint8_t)(result.r * 255),
        (uint8_t)(result.g * 255),
        (uint8_t)(result.b * 255),
        255
    };
}

// ==========================================
// GEOMETRY LOADER
// ==========================================

class ModelGeometry
{
public:
    ModelGeometry(std::string model_path)
    {
        unsigned int flags =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_FlipUVs |
            aiProcess_JoinIdenticalVertices;

        const aiScene *scene = importer.ReadFile(model_path.c_str(), flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Model load error: " << importer.GetErrorString() << std::endl;
            return;
        }

        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                if (mesh->mFaces[j].mNumIndices == 3) {
                    for(int k=0; k<3; k++) {
                        aiVector3D v = mesh->mVertices[mesh->mFaces[j].mIndices[k]];
                        triangles.push_back(glm::vec3(v.x, v.y, v.z));

                        if (mesh->HasNormals()) {
                            aiVector3D n = mesh->mNormals[mesh->mFaces[j].mIndices[k]];
                            normals.push_back(glm::vec3(n.x, n.y, n.z));
                        } else {
                            normals.push_back(glm::vec3(0, 0, 1));
                        }
                    }
                }
            }
        }
    }

    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;

private:
    Assimp::Importer importer;
};

// ==========================================
// VIEWER 
// ==========================================

class Viewer
{
public:
    Viewer(glm::vec3 position, float speed)
    {
        this->position              = position;
        this->speed                 = speed;
        this->camera                = new shs::Camera3D();
        this->camera->position      = this->position;
        this->camera->width         = float(CANVAS_WIDTH);
        this->camera->height        = float(CANVAS_HEIGHT);
        this->camera->field_of_view = 60.0f;
        this->camera->z_near        = 0.1f;
        this->camera->z_far         = 1000.0f;
        this->horizontal_angle      = 0.0f;
        this->vertical_angle        = 0.0f;
        update();
    }
    ~Viewer() { delete camera; }

    void update()
    {
        this->camera->position         = this->position;
        this->camera->horizontal_angle = this->horizontal_angle;
        this->camera->vertical_angle   = this->vertical_angle;
        this->camera->update();
    }

    glm::vec3 get_direction_vector() { return this->camera->direction_vector; }
    glm::vec3 get_right_vector()     { return this->camera->right_vector; }

    shs::Camera3D *camera;
    glm::vec3      position;
    float          horizontal_angle;
    float          vertical_angle;
    float          speed;
};

// ==========================================
// 9 MONKEY OBJECTS (independent tween/bob + some rotate)
// ==========================================

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(ModelGeometry* geom, glm::vec3 base_pos, shs::Color color, int idx)
    {
        this->geometry      = geom;
        this->base_position = base_pos;
        this->position      = base_pos;
        this->scale         = glm::vec3(MONKEY_SCALE);
        this->color         = color;

        this->rotate_enabled  = (idx % 2 == 0);
        this->rotate_speed_deg = 20.0f + 12.0f * float(idx % 4);

        this->bob_speed = 0.6f + 0.25f * float(idx);
        this->bob_amp   = 0.8f + 0.15f * float(idx % 3);
        this->phase     = 1.37f * float(idx);

        this->time_accum = 0.0f;
        this->rotation_angle = 0.0f;
    }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float delta_time) override
    {
        time_accum += delta_time;

        float y = std::sin(time_accum * bob_speed + phase) * bob_amp;
        this->position.y = base_position.y + y;

        if (rotate_enabled) {
            rotation_angle += rotate_speed_deg * delta_time;
            if (rotation_angle > 360.0f) rotation_angle -= 360.0f;
        }
    }

    void render() override {}

    ModelGeometry *geometry;
    glm::vec3      scale;
    glm::vec3      base_position;
    glm::vec3      position;
    shs::Color     color;

    bool  rotate_enabled;
    float rotate_speed_deg;

    float time_accum;
    float bob_speed;
    float bob_amp;
    float phase;

    float rotation_angle;
};

// ==========================================
// SCENE
// ==========================================

class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(Viewer *viewer)
    {
        this->viewer = viewer;
        this->light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));

        this->shared_monkey_geometry = new ModelGeometry("./obj/monkey/monkey.rawobj");

        int idx = 0;
        for (int gz = 0; gz < GRID_Z; gz++) {
            for (int gx = 0; gx < GRID_X; gx++) {

                float x = (float(gx) - float(GRID_X - 1) * 0.5f) * SPACING_X;
                float z = START_Z + float(gz) * SPACING_Z;

                this->scene_objects.push_back(new MonkeyObject(
                    this->shared_monkey_geometry,
                    glm::vec3(x, BASE_Y, z),
                    monkey_color_from_i(idx),
                    idx
                ));

                idx++;
            }
        }
    }

    ~HelloScene() {
        for (auto *obj : this->scene_objects) delete obj;
        delete this->shared_monkey_geometry;
    }

    void process() override {}

    std::vector<shs::AbstractObject3D *> scene_objects;
    Viewer                              *viewer;
    glm::vec3                            light_direction;
    ModelGeometry*                       shared_monkey_geometry;
};

// ==========================================
// DEMO-SPECIFIC RENDER TARGET (Color+Depth) 
// Expected fields:
//   struct RT_ColorDepth { shs::Canvas color; shs::ZBuffer depth; ... clear(...) ... }
// ==========================================
/*
    REQUIRED (in header):
    namespace shs {
        struct RT_ColorDepth {
            shs::Canvas  color;
            shs::ZBuffer depth;
            RT_ColorDepth(int w,int h,float zn,float zf, shs::Color clear);
            inline void clear(shs::Color c);
            inline int width() const;
            inline int height() const;
        };
    }
*/

// ==========================================
// TILED RASTERIZER (writes depth in CANVAS coords to avoid inversion)
// ==========================================

static void draw_triangle_tile_color_depth(
    shs::RT_ColorDepth &rt,
    const std::vector<glm::vec3> &vertices,
    const std::vector<glm::vec3> &normals,
    std::function<shs::Varyings(const glm::vec3&, const glm::vec3&)> vertex_shader,
    std::function<shs::Color(const shs::Varyings&)> fragment_shader,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    int w = rt.color.get_width();
    int h = rt.color.get_height();

    shs::Varyings vout[3];
    glm::vec3 screen_coords[3];

    for (int i = 0; i < 3; i++) {
        vout[i] = vertex_shader(vertices[i], normals[i]);
        screen_coords[i] = shs::Canvas::clip_to_screen(vout[i].position, w, h);
    }

    glm::vec2 bboxmin(tile_max.x, tile_max.y);
    glm::vec2 bboxmax(tile_min.x, tile_min.y);
    std::vector<glm::vec2> v2d = {
        glm::vec2(screen_coords[0]),
        glm::vec2(screen_coords[1]),
        glm::vec2(screen_coords[2])
    };

    for (int i = 0; i < 3; i++) {
        bboxmin = glm::max(glm::vec2(tile_min), glm::min(bboxmin, v2d[i]));
        bboxmax = glm::min(glm::vec2(tile_max), glm::max(bboxmax, v2d[i]));
    }

    if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

    float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) - (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
    if (area <= 0) return;

    for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
        for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

            float z = bc.x * vout[0].view_z + bc.y * vout[1].view_z + bc.z * vout[2].view_z;

            // screen py -> canvas y
            int cy = (h - 1) - py;

            if (rt.depth.test_and_set_depth(px, cy, z)) {

                shs::Varyings interpolated;
                interpolated.normal    = glm::normalize(bc.x * vout[0].normal    + bc.y * vout[1].normal    + bc.z * vout[2].normal);
                interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                interpolated.view_z    = z;

                rt.color.draw_pixel_screen_space(px, py, fragment_shader(interpolated));
            }
        }
    }
}

// ==========================================
// PASS 1: OUTLINE (depth-based)  rt_scene -> ping
// ==========================================

static void outline_pass(
    const shs::Canvas& src,
    const shs::ZBuffer& depth,
    shs::Canvas& dst,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    int W = src.get_width();
    int H = src.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wg.add(1);
            job_system->submit({[&, tx, ty]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {

                        shs::Color c = src.get_color_at(x, y);
                        float d0 = depth.get_depth_at(x, y);

                        if (d0 == std::numeric_limits<float>::max()) {
                            dst.draw_pixel(x, y, c);
                            continue;
                        }

                        float max_delta = 0.0f;
                        for (int oy = -EDGE_RADIUS; oy <= EDGE_RADIUS; oy++) {
                            for (int ox = -EDGE_RADIUS; ox <= EDGE_RADIUS; ox++) {
                                if (ox == 0 && oy == 0) continue;
                                int sx = clampi(x + ox, 0, W - 1);
                                int sy = clampi(y + oy, 0, H - 1);
                                float d1 = depth.get_depth_at(sx, sy);
                                if (d1 == std::numeric_limits<float>::max()) continue;
                                max_delta = std::max(max_delta, std::abs(d1 - d0));
                            }
                        }

                        float edge = (max_delta > EDGE_THRESHOLD) ? 1.0f : 0.0f;
                        float k = 1.0f - edge * EDGE_STRENGTH;

                        shs::Color out = shs::Color{
                            (uint8_t)clampi(int(float(c.r) * k), 0, 255),
                            (uint8_t)clampi(int(float(c.g) * k), 0, 255),
                            (uint8_t)clampi(int(float(c.b) * k), 0, 255),
                            255
                        };

                        dst.draw_pixel(x, y, out);
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

// ==========================================
// PASS 2: FOG (depth-based)  ping + depth -> pong
// ==========================================

static void fog_pass(
    const shs::Canvas& src,
    const shs::ZBuffer& depth,
    shs::Canvas& dst,
    shs::Color fog_color,
    float fog_start,
    float fog_end,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    int W = src.get_width();
    int H = src.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wg.add(1);
            job_system->submit({[&, tx, ty]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {

                        shs::Color c = src.get_color_at(x, y);
                        float d = depth.get_depth_at(x, y);

                        if (d == std::numeric_limits<float>::max()) {
                            dst.draw_pixel(x, y, c);
                            continue;
                        }

                        float t = (d - fog_start) / (fog_end - fog_start);
                        t = smoothstep01(t);

                        dst.draw_pixel(x, y, lerp_color(c, fog_color, t));
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

// ==========================================
// RENDERER SYSTEM (Threaded)
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, shs::RT_ColorDepth *rt)
        : scene(scene), job_system(job_sys), rt(rt)
    {}

    void process(float delta_time) override
    {
        (void)delta_time;

        rt->clear(shs::Color{20, 20, 25, 255});

        glm::mat4 view = scene->viewer->camera->view_matrix;
        glm::mat4 proj = scene->viewer->camera->projection_matrix;

        int w = rt->color.get_width();
        int h = rt->color.get_height();

        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wait_group.reset();

        for(int ty = 0; ty < rows; ty++) {
            for(int tx = 0; tx < cols; tx++) {

                wait_group.add(1);

                job_system->submit({[this, tx, ty, w, h, view, proj]() {

                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, w) - 1,
                                     std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                    for (shs::AbstractObject3D *object : scene->scene_objects)
                    {
                        MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                        if (!monkey) continue;

                        Uniforms uniforms;
                        uniforms.model      = monkey->get_world_matrix();
                        uniforms.view       = view;
                        uniforms.mvp        = proj * view * uniforms.model;
                        uniforms.light_dir  = scene->light_direction;
                        uniforms.camera_pos = scene->viewer->position;
                        uniforms.color      = monkey->color;

                        const auto& verts = monkey->geometry->triangles;
                        const auto& norms = monkey->geometry->normals;

                        for (size_t i = 0; i < verts.size(); i += 3)
                        {
                            std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                            std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                            draw_triangle_tile_color_depth(
                                *rt,
                                tri_verts,
                                tri_norms,
                                [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                                    return blinn_phong_vertex_shader(p, n, uniforms);
                                },
                                [&uniforms](const shs::Varyings& v) {
                                    return blinn_phong_fragment_shader(v, uniforms);
                                },
                                t_min, t_max
                            );
                        }
                    }

                    wait_group.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wait_group.wait();
    }

private:
    HelloScene                          *scene;
    shs::Job::ThreadedPriorityJobSystem *job_system;
    shs::RT_ColorDepth                  *rt;
    shs::Job::WaitGroup                  wait_group;
};

// ==========================================
// LOGIC SYSTEM
// ==========================================

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {
        scene->viewer->update();
        for (auto *obj : scene->scene_objects)
            obj->update(delta_time);
    }
private:
    HelloScene *scene;
};

// ==========================================
// SYSTEM PROCESSOR 
// ==========================================

class SystemProcessor
{
public:
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, shs::RT_ColorDepth *rt)
    {
        command_processor = new shs::CommandProcessor();
        renderer_system   = new RendererSystem(scene, job_sys, rt);
        logic_system      = new LogicSystem(scene);
    }
    ~SystemProcessor()
    {
        delete command_processor;
        delete renderer_system;
        delete logic_system;
    }
    void process(float delta_time)
    {
        command_processor->process();
        logic_system->process(delta_time);
    }
    void render(float delta_time)
    {
        renderer_system->process(delta_time);
    }

    shs::CommandProcessor *command_processor;
    LogicSystem           *logic_system;
    RendererSystem        *renderer_system;
};

// ==========================================
// MAIN
// ==========================================

int main(int argc, char* argv[])
{
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas *screen_canvas  = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *screen_surface = screen_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, screen_surface);

    Viewer     *viewer      = new Viewer(glm::vec3(0.0f, 6.0f, -28.0f), 50.0f);
    HelloScene *scene       = new HelloScene(viewer);

    // Scene render target (Color+Depth)
    shs::RT_ColorDepth rt_scene(CANVAS_WIDTH, CANVAS_HEIGHT, viewer->camera->z_near, viewer->camera->z_far, shs::Color{20,20,25,255});

    // Ping-pong canvases (only color)
    shs::Canvas ping(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas pong(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    SystemProcessor *sys = new SystemProcessor(scene, job_system, &rt_scene);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();

    bool is_dragging = false;

    shs::Job::WaitGroup wg_outline;
    shs::Job::WaitGroup wg_fog;

    while (!exit)
    {
        Uint32 current_tick = SDL_GetTicks();
        float delta_time = (current_tick - last_tick) / 1000.0f;
        last_tick = current_tick;

        while (SDL_PollEvent(&event_data))
        {
            if (event_data.type == SDL_QUIT) exit = true;

            if (event_data.type == SDL_MOUSEBUTTONDOWN) {
                if (event_data.button.button == SDL_BUTTON_LEFT) is_dragging = true;
            }
            if (event_data.type == SDL_MOUSEBUTTONUP) {
                if (event_data.button.button == SDL_BUTTON_LEFT) is_dragging = false;
            }

            if (event_data.type == SDL_MOUSEMOTION)
            {
                if (is_dragging) {
                    viewer->horizontal_angle += event_data.motion.xrel * MOUSE_SENSITIVITY;
                    viewer->vertical_angle   -= event_data.motion.yrel * MOUSE_SENSITIVITY;

                    if (viewer->vertical_angle >  89.0f) viewer->vertical_angle =  89.0f;
                    if (viewer->vertical_angle < -89.0f) viewer->vertical_angle = -89.0f;
                }
            }

            if (event_data.type == SDL_KEYDOWN) {
                if(event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;

                if(event_data.key.keysym.sym == SDLK_w) sys->command_processor->add_command(new shs::MoveForwardCommand (viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_s) sys->command_processor->add_command(new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_a) sys->command_processor->add_command(new shs::MoveLeftCommand    (viewer->position, viewer->get_right_vector()    , viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_d) sys->command_processor->add_command(new shs::MoveRightCommand   (viewer->position, viewer->get_right_vector()    , viewer->speed, delta_time));
            }
        }

        // Logic + Render scene into rt_scene
        sys->process(delta_time);
        sys->render(delta_time);

        // PASS 1: Outline (rt_scene -> ping)
        outline_pass(rt_scene.color, rt_scene.depth, ping, job_system, wg_outline);

        // PASS 2: Fog (ping -> pong) using rt_scene.depth
        fog_pass(ping, rt_scene.depth, pong, FOG_COLOR, FOG_START_Z, FOG_END_Z, job_system, wg_fog);

        // Present pong
        screen_canvas->buffer() = pong.buffer();
        shs::Canvas::copy_to_SDLSurface(screen_surface, screen_canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    delete sys;
    delete scene;
    delete viewer;
    delete screen_canvas;
    delete job_system;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(screen_surface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
