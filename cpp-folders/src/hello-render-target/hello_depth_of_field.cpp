/*
    3D Software Renderer - Blinn-Phong + Auto-Focus Depth of Field (CPU Post Process)
    - Render scene into RenderTarget (offscreen) with real depth (view_z)
    - Build blurred image using separable Gaussian blur (job system accelerated)
    - Auto-focus by sampling center depth window (median)
    - Composite sharp vs blurred based on circle-of-confusion from depth (job system accelerated)
    - Present final color buffer to SDL screen

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

#define WINDOW_WIDTH      800
#define WINDOW_HEIGHT     600
#define CANVAS_WIDTH      380
#define CANVAS_HEIGHT     280
#define MOUSE_SENSITIVITY 0.2f
#define THREAD_COUNT      20
#define TILE_SIZE_X       80
#define TILE_SIZE_Y       80

// ===============================
// MONKEY CONFIG
// ===============================
#define MONKEY_COUNT        9   
#define MONKEY_SCALE        3.2f
#define MONKEY_SPACING_Z    8.0f
#define MONKEY_SPACING_X    6.5f
#define MONKEY_START_Z      6.0f
#define MONKEY_CENTER_X     0.0f

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

struct Uniforms {
    glm::mat4  mvp;
    glm::mat4  model;
    glm::mat4  view;
    glm::vec3  light_dir;
    glm::vec3  camera_pos;
    shs::Color color;
};

/*
    VERTEX SHADER (Blinn-Phong)
    - out.position : clip-space
    - out.world_pos: world-space
    - out.normal   : world-space normal
    - out.view_z   : view-space depth (for ZBuffer + DoF)
*/
shs::Varyings blinn_phong_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;
    out.position  = u.mvp * glm::vec4(aPos, 1.0f);
    out.world_pos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal    = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv        = glm::vec2(0.0f);

    glm::vec4 view_pos = u.view * u.model * glm::vec4(aPos, 1.0f);
    out.view_z = view_pos.z;

    return out;
}

shs::Color blinn_phong_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    glm::vec3 norm     = glm::normalize(in.normal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - in.world_pos);

    float ambientStrength = 0.15f;
    glm::vec3 ambient     = ambientStrength * glm::vec3(1.0f, 1.0f, 1.0f);

    float diff        = glm::max(glm::dot(norm, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f, 1.0f, 1.0f);

    glm::vec3 halfwayDir   = glm::normalize(lightDir + viewDir);
    float specularStrength = 0.5f;
    float shininess        = 64.0f;

    float spec         = glm::pow(glm::max(glm::dot(norm, halfwayDir), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f, 1.0f, 1.0f);

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

static inline shs::Color color_from_rgbaf(float r, float g, float b, float a)
{
    r = std::min(255.0f, std::max(0.0f, r));
    g = std::min(255.0f, std::max(0.0f, g));
    b = std::min(255.0f, std::max(0.0f, b));
    a = std::min(255.0f, std::max(0.0f, a));
    return shs::Color{ (uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a };
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
// GAUSSIAN BLUR (JOB SYSTEM)
// ==========================================

static void gaussian_blur_pass(
    const shs::Canvas& src,
    shs::Canvas& dst,
    bool horizontal,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wait_group)
{
    const float w0 = 0.06136f;
    const float w1 = 0.24477f;
    const float w2 = 0.38774f;

    int W = src.get_width();
    int H = src.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wait_group.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wait_group.add(1);

            job_system->submit({[&, tx, ty]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                auto sample = [&](int sx, int sy) -> shs::Color {
                    sx = clampi(sx, 0, W - 1);
                    sy = clampi(sy, 0, H - 1);
                    return src.get_color_at(sx, sy);
                };

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {

                        float r = 0, g = 0, b = 0, a = 0;

                        if (horizontal) {
                            shs::Color c0 = sample(x - 2, y);
                            shs::Color c1 = sample(x - 1, y);
                            shs::Color c2 = sample(x,     y);
                            shs::Color c3 = sample(x + 1, y);
                            shs::Color c4 = sample(x + 2, y);

                            r = w0*c0.r + w1*c1.r + w2*c2.r + w1*c3.r + w0*c4.r;
                            g = w0*c0.g + w1*c1.g + w2*c2.g + w1*c3.g + w0*c4.g;
                            b = w0*c0.b + w1*c1.b + w2*c2.b + w1*c3.b + w0*c4.b;
                            a = w0*c0.a + w1*c1.a + w2*c2.a + w1*c3.a + w0*c4.a;
                        } else {
                            shs::Color c0 = sample(x, y - 2);
                            shs::Color c1 = sample(x, y - 1);
                            shs::Color c2 = sample(x, y);
                            shs::Color c3 = sample(x, y + 1);
                            shs::Color c4 = sample(x, y + 2);

                            r = w0*c0.r + w1*c1.r + w2*c2.r + w1*c3.r + w0*c4.r;
                            g = w0*c0.g + w1*c1.g + w2*c2.g + w1*c3.g + w0*c4.g;
                            b = w0*c0.b + w1*c1.b + w2*c2.b + w1*c3.b + w0*c4.b;
                            a = w0*c0.a + w1*c1.a + w2*c2.a + w1*c3.a + w0*c4.a;
                        }

                        dst.draw_pixel(x, y, color_from_rgbaf(r, g, b, a));
                    }
                }

                wait_group.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wait_group.wait();
}

// ==========================================
// AUTOFOCUS + DOF COMPOSITE (JOB SYSTEM)
// ==========================================

static float autofocus_depth_median_center(
    const shs::ZBuffer& zbuf,
    int cx, int cy,
    int radius_px)
{
    std::vector<float> samples;
    samples.reserve((size_t)(2 * radius_px + 1) * (size_t)(2 * radius_px + 1));

    for (int dy = -radius_px; dy <= radius_px; ++dy) {
        for (int dx = -radius_px; dx <= radius_px; ++dx) {
            int x = cx + dx;
            int y = cy + dy;

            float d = zbuf.get_depth_at(x, y);
            if (d == std::numeric_limits<float>::max()) continue;
            samples.push_back(d);
        }
    }

    if (samples.empty()) {
        float d = zbuf.get_depth_at(cx, cy);
        if (d == std::numeric_limits<float>::max()) return 15.0f;
        return d;
    }

    size_t mid = samples.size() / 2;
    std::nth_element(samples.begin(), samples.begin() + mid, samples.end());
    return samples[mid];
}

static void dof_composite_pass(
    const shs::Canvas& sharp,
    const shs::Canvas& blur,
    const shs::ZBuffer& zbuf,
    shs::Canvas& out,
    float focus_depth,
    float range,
    float max_blur,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wait_group)
{
    int W = sharp.get_width();
    int H = sharp.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wait_group.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wait_group.add(1);

            job_system->submit({[&, tx, ty]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {

                        float d = zbuf.get_depth_at(x, y);
                        if (d == std::numeric_limits<float>::max()) {
                            d = focus_depth + range;
                        }

                        float coc = std::abs(d - focus_depth) / range;
                        float t   = smoothstep01(coc);
                        t = clampf(t * max_blur, 0.0f, 1.0f);

                        shs::Color c_sharp = sharp.get_color_at(x, y);
                        shs::Color c_blur  = blur.get_color_at(x, y);

                        out.draw_pixel(x, y, lerp_color(c_sharp, c_blur, t));
                    }
                }

                wait_group.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wait_group.wait();
}

// ==========================================
// SCENE & OBJECT CLASSES
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

class ModelGeometry
{
public:
    ModelGeometry(std::string model_path)
    {
        unsigned int flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices;
        const aiScene *scene = this->importer.ReadFile(model_path.c_str(), flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Model load error: " << this->importer.GetErrorString() << std::endl;
            return;
        }

        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                if (mesh->mFaces[j].mNumIndices == 3) {
                    for(int k=0; k<3; k++) {
                        aiVector3D v = mesh->mVertices[mesh->mFaces[j].mIndices[k]];
                        this->triangles.push_back(glm::vec3(v.x, v.y, v.z));
                        if (mesh->HasNormals()) {
                            aiVector3D n = mesh->mNormals[mesh->mFaces[j].mIndices[k]];
                            this->normals.push_back(glm::vec3(n.x, n.y, n.z));
                        } else {
                            this->normals.push_back(glm::vec3(0, 0, 1));
                        }
                    }
                }
            }
        }
    }
    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;
    Assimp::Importer importer;
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(ModelGeometry* shared_geom, glm::vec3 position, glm::vec3 scale, shs::Color color, bool rotate_enabled, float rotate_speed_deg)
    {
        this->geometry        = shared_geom;
        this->position        = position;
        this->scale           = scale;
        this->color           = color;
        this->rotate_enabled  = rotate_enabled;
        this->rotate_speed_deg = rotate_speed_deg;
        this->rotation_angle  = 0.0f;
    }
    ~MonkeyObject() {}

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float delta_time) override
    {
        if (this->rotate_enabled) {
            this->rotation_angle += this->rotate_speed_deg * delta_time;
            if (this->rotation_angle > 360.0f) this->rotation_angle -= 360.0f;
        }
    }

    void render() override {}

    ModelGeometry *geometry;
    glm::vec3      scale;
    glm::vec3      position;
    shs::Color     color;

    bool  rotate_enabled;
    float rotate_speed_deg;
    float rotation_angle;
};

class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(Viewer *viewer)
    {
        this->viewer = viewer;
        this->light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));

        this->shared_monkey_geometry = new ModelGeometry("./assets/obj/monkey/monkey.rawobj");

        const int N = MONKEY_COUNT;

        for (int i = 0; i < N; i++)
        {
            float z = MONKEY_START_Z + (float)i * MONKEY_SPACING_Z;

            // x pattern: center, right, left, right, left...
            float x = MONKEY_CENTER_X;
            if (i % 2 == 1) x += MONKEY_SPACING_X;
            if (i % 4 == 3) x -= 2.0f * MONKEY_SPACING_X;

            bool rotate_enabled = ((i % 3) != 0);        // 2/3 rotate
            float rotate_speed  = 25.0f + 10.0f * (i % 4);

            this->scene_objects.push_back(new MonkeyObject(
                this->shared_monkey_geometry,
                glm::vec3(x, 0.0f, z),
                glm::vec3(MONKEY_SCALE),
                monkey_color_from_i(i),
                rotate_enabled,
                rotate_speed
            ));
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

    ModelGeometry* shared_monkey_geometry;
};

// ==========================================
// RENDERER SYSTEM (THREADED RENDERING -> RenderTarget)
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, shs::RT_ColorDepth *target)
        : scene(scene), job_system(job_sys), target(target)
    {}

    static void draw_triangle_tile(
        shs::Canvas &canvas,
        shs::ZBuffer &z_buffer,
        const std::vector<glm::vec3> &vertices,
        const std::vector<glm::vec3> &normals,
        std::function<shs::Varyings(const glm::vec3&, const glm::vec3&)> vertex_shader,
        std::function<shs::Color(const shs::Varyings&)> fragment_shader,
        glm::ivec2 tile_min, glm::ivec2 tile_max)
    {
        shs::Varyings vout[3];
        glm::vec3 screen_coords[3];

        for (int i = 0; i < 3; i++) {
            vout[i] = vertex_shader(vertices[i], normals[i]);
            screen_coords[i] = shs::Canvas::clip_to_screen(vout[i].position, canvas.get_width(), canvas.get_height());
        }

        glm::vec2 bboxmin(tile_max.x, tile_max.y);
        glm::vec2 bboxmax(tile_min.x, tile_min.y);
        std::vector<glm::vec2> v2d = { glm::vec2(screen_coords[0]), glm::vec2(screen_coords[1]), glm::vec2(screen_coords[2]) };

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

                if (z_buffer.test_and_set_depth(px, py, z)) {

                    shs::Varyings interpolated;
                    interpolated.normal    = glm::normalize(bc.x * vout[0].normal    + bc.y * vout[1].normal    + bc.z * vout[2].normal);
                    interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                    interpolated.view_z    = z;

                    canvas.draw_pixel_screen_space(px, py, fragment_shader(interpolated));
                }
            }
        }
    }

    void process(float delta_time) override
    {
        (void)delta_time;

        this->target->clear(shs::Color{20, 20, 25, 255});

        glm::mat4 view = this->scene->viewer->camera->view_matrix;
        glm::mat4 proj = this->scene->viewer->camera->projection_matrix;

        int w = this->target->color.get_width();
        int h = this->target->color.get_height();

        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wait_group.reset();

        for(int ty = 0; ty < rows; ty++) {
            for(int tx = 0; tx < cols; tx++) {

                wait_group.add(1);

                job_system->submit({[this, tx, ty, w, h, view, proj]() {

                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, w) - 1, std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                    for (shs::AbstractObject3D *object : this->scene->scene_objects)
                    {
                        MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                        if (!monkey) continue;

                        Uniforms uniforms;
                        uniforms.model      = monkey->get_world_matrix();
                        uniforms.view       = view;
                        uniforms.mvp        = proj * view * uniforms.model;
                        uniforms.light_dir  = this->scene->light_direction;
                        uniforms.camera_pos = this->scene->viewer->position;
                        uniforms.color      = monkey->color;

                        const auto& verts = monkey->geometry->triangles;
                        const auto& norms = monkey->geometry->normals;

                        for (size_t i = 0; i < verts.size(); i += 3)
                        {
                            std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                            std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                            draw_triangle_tile(
                                this->target->color,
                                this->target->depth,
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
    shs::RT_ColorDepth                  *target;
    shs::Job::WaitGroup                  wait_group;
};

// ==========================================
// LOGIC & MAIN LOOP
// ==========================================

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {
        this->scene->viewer->update();
        for (auto *obj : this->scene->scene_objects)
            obj->update(delta_time);
    }
private:
    HelloScene *scene;
};

class SystemProcessor
{
public:
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, shs::RT_ColorDepth *target)
    {
        this->command_processor = new shs::CommandProcessor();
        this->renderer_system   = new RendererSystem(scene, job_sys, target);
        this->logic_system      = new LogicSystem(scene);
    }
    ~SystemProcessor()
    {
        delete this->command_processor;
        delete this->renderer_system;
        delete this->logic_system;
    }
    void process(float delta_time)
    {
        this->command_processor->process();
        this->logic_system->process(delta_time);
    }
    void render(float delta_time)
    {
        this->renderer_system->process(delta_time);
    }

    shs::CommandProcessor *command_processor;
    LogicSystem           *logic_system;
    RendererSystem        *renderer_system;
};

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

    Viewer     *viewer      = new Viewer(glm::vec3(0.0f, 5.0f, -20.0f), 50.0f);
    HelloScene *hello_scene = new HelloScene(viewer);

    shs::RT_ColorDepth ping(CANVAS_WIDTH, CANVAS_HEIGHT, viewer->camera->z_near, viewer->camera->z_far, shs::Color{20,20,25,255});
    shs::RT_ColorDepth pong(CANVAS_WIDTH, CANVAS_HEIGHT, viewer->camera->z_near, viewer->camera->z_far, shs::Color{20,20,25,255});

    shs::Canvas sharp_copy(CANVAS_WIDTH, CANVAS_HEIGHT);

    SystemProcessor *sys = new SystemProcessor(hello_scene, job_system, &ping);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();

    bool is_dragging = false;

    // Blur + DoF params
    const bool ENABLE_DOF       = true;
    const int  BLUR_ITERATIONS  = 3;
    const int  AUTOFOCUS_RADIUS = 6;
    float dof_range             = 24.0f;
    float dof_maxblur           = 0.6f;

    shs::Job::WaitGroup blur_wait_group;
    shs::Job::WaitGroup dof_wait_group;

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

        sys->process(delta_time);
        sys->render(delta_time);

        if (ENABLE_DOF)
        {
            sharp_copy.buffer() = ping.color.buffer();
            pong.color.buffer() = sharp_copy.buffer();

            for (int i = 0; i < BLUR_ITERATIONS; i++)
            {
                gaussian_blur_pass(pong.color, ping.color, true,  job_system, blur_wait_group);
                gaussian_blur_pass(ping.color, pong.color, false, job_system, blur_wait_group);
            }

            int cx = CANVAS_WIDTH  / 2;
            int cy = CANVAS_HEIGHT / 2;
            float focus_depth = autofocus_depth_median_center(ping.depth, cx, cy, AUTOFOCUS_RADIUS);

            dof_composite_pass(
                sharp_copy,
                pong.color,
                ping.depth,
                ping.color,
                focus_depth,
                dof_range,
                dof_maxblur,
                job_system,
                dof_wait_group
            );
        }

        screen_canvas->buffer() = ping.color.buffer();

        shs::Canvas::copy_to_SDLSurface(screen_surface, screen_canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    delete sys;
    delete hello_scene;
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
