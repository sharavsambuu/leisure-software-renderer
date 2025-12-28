/*
    3D Software Renderer - PER-OBJECT MOTION BLUR (Velocity Buffer, CPU Post Process)
    - Render scene into RT_ColorDepthVelocity (Color + Depth + Velocity)
    - Velocity is computed per-pixel from current_clip - prev_clip (stored in Varyings)
    - Post-process: sample backward along velocity in Canvas space

    References:
    - https://john-chapman-graphics.blogspot.com/2013/01/per-object-motion-blur.html
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

#define WINDOW_WIDTH      740
#define WINDOW_HEIGHT     520
#define CANVAS_WIDTH      740
#define CANVAS_HEIGHT     520
#define MOUSE_SENSITIVITY 0.2f
#define THREAD_COUNT      20
#define TILE_SIZE_X       40
#define TILE_SIZE_Y       40

// ===============================
// MONKEY CONFIG (3x3 = 9)
// ===============================
#define MONKEY_COUNT_X      3
#define MONKEY_COUNT_Y      3
#define MONKEY_COUNT        9
#define MONKEY_SCALE        3.2f
#define MONKEY_SPACING_X    8.0f
#define MONKEY_SPACING_Z    8.0f
#define MONKEY_START_Z      8.0f
#define MONKEY_BASE_Y       0.0f

// ===============================
// MOTION BLUR CONFIG
// ===============================
#define BLUR_MULTIPLIER     0.85f
#define MAX_BLUR_SAMPLES    12
#define MIN_VEL2_THRESHOLD  0.25f

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

struct Uniforms {
    glm::mat4  mvp;
    glm::mat4  prev_mvp;
    glm::mat4  model;
    glm::mat4  view;
    glm::vec3  light_dir;
    glm::vec3  camera_pos;
    shs::Color color;
    glm::vec2  viewport_size;
};

using FragOutput = std::pair<shs::Color, glm::vec2>;

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

static inline shs::Color color_from_rgbaf(float r, float g, float b, float a)
{
    r = std::min(255.0f, std::max(0.0f, r));
    g = std::min(255.0f, std::max(0.0f, g));
    b = std::min(255.0f, std::max(0.0f, b));
    a = std::min(255.0f, std::max(0.0f, a));
    return shs::Color{ (uint8_t)r, (uint8_t)g, (uint8_t)b, (uint8_t)a };
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

/*
    VERTEX SHADER (Blinn-Phong + Velocity)
    - out.position      : current clip-space
    - out.prev_position : previous clip-space
    - out.world_pos     : current world pos (for lighting)
    - out.normal        : world normal
    - out.view_z        : view depth for ZBuffer
*/
static shs::Varyings velocity_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;
    out.position       = u.mvp * glm::vec4(aPos, 1.0f);
    out.prev_position  = u.prev_mvp * glm::vec4(aPos, 1.0f);

    out.world_pos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal    = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv        = glm::vec2(0.0f);

    glm::vec4 view_pos = u.view * u.model * glm::vec4(aPos, 1.0f);
    out.view_z = view_pos.z;

    return out;
}

static FragOutput velocity_fragment_shader(const shs::Varyings& in, const Uniforms& u)
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

    shs::Color final_color = {
        (uint8_t)(result.r * 255),
        (uint8_t)(result.g * 255),
        (uint8_t)(result.b * 255),
        255
    };

    glm::vec2 current_ndc = glm::vec2(in.position)      / in.position.w;
    glm::vec2 prev_ndc    = glm::vec2(in.prev_position) / in.prev_position.w;

    glm::vec2 velocity_ndc    = current_ndc - prev_ndc;
    glm::vec2 velocity_pixels = velocity_ndc * 0.5f * u.viewport_size; // +Y up (Canvas convention)

    return { final_color, velocity_pixels };
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
    MonkeyObject(ModelGeometry* shared_geom, glm::vec3 base_position, glm::vec3 scale, shs::Color color, bool rotate_enabled, float rotate_speed_deg, int idx)
    {
        this->geometry          = shared_geom;
        this->scale             = scale;
        this->color             = color;

        this->base_position     = base_position;
        this->position          = base_position;
        this->prev_position     = base_position;

        this->rotate_enabled    = rotate_enabled;
        this->rotate_speed_deg  = rotate_speed_deg;

        this->rotation_angle    = 0.0f;
        this->prev_rotation     = 0.0f;

        // Tween-like bobbing (smooth):
        // - different speed
        // - different phase offset
        // - different amplitude
        this->time_accum        = 0.0f + (float)idx * 0.77f;
        //this->bob_speed         = 0.95f + 0.35f * (float)(idx % 5) + 0.15f * (float)(idx % 3);
        this->bob_speed         = 1.5f + 0.6f * (float)(idx % 5);
        //this->bob_amp           = 0.80f + 0.25f * (float)(idx % 4);
        this->bob_amp           = 2.0f + 0.8f * (float)(idx % 3);
    }

    ~MonkeyObject() {}

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), this->scale);
        return t * r * s;
    }

    glm::mat4 get_prev_world_matrix()
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->prev_position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->prev_rotation), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), this->scale);
        return t * r * s;
    }

    void update(float delta_time) override
    {
        // Save previous state (for velocity)
        this->prev_position = this->position;
        this->prev_rotation = this->rotation_angle;

        this->time_accum += delta_time;

        // Smooth tween-like vertical motion (sin is already smooth)
        float y = this->base_position.y + std::sin(this->time_accum * this->bob_speed) * this->bob_amp;
        this->position.y = y;

        if (this->rotate_enabled) {
            this->rotation_angle += this->rotate_speed_deg * delta_time;
            if (this->rotation_angle > 360.0f) this->rotation_angle -= 360.0f;
        }
    }

    void render() override {}

    ModelGeometry *geometry;
    glm::vec3      scale;

    glm::vec3      base_position;
    glm::vec3      position;
    glm::vec3      prev_position;

    shs::Color     color;

    bool  rotate_enabled;
    float rotate_speed_deg;

    float rotation_angle;
    float prev_rotation;

    float time_accum;
    float bob_speed;
    float bob_amp;
};

class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(Viewer *viewer)
    {
        this->viewer = viewer;
        this->light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));

        this->shared_monkey_geometry = new ModelGeometry("./obj/monkey/monkey.rawobj");

        int idx = 0;
        for (int yy = 0; yy < MONKEY_COUNT_Y; yy++)
        {
            for (int xx = 0; xx < MONKEY_COUNT_X; xx++)
            {
                float x = (xx - 1) * MONKEY_SPACING_X; // -1,0,1
                float z = MONKEY_START_Z + (yy * MONKEY_SPACING_Z);
                float y = MONKEY_BASE_Y;

                bool rotate_enabled = ((idx % 3) != 0);
                //float rotate_speed  = 20.0f + 15.0f * (idx % 4);
                float rotate_speed = 60.0f + 40.0f * (idx % 4);

                this->scene_objects.push_back(new MonkeyObject(
                    this->shared_monkey_geometry,
                    glm::vec3(x, y, z),
                    glm::vec3(MONKEY_SCALE),
                    monkey_color_from_i(idx),
                    rotate_enabled,
                    rotate_speed,
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

    ModelGeometry* shared_monkey_geometry;
};

// ==========================================
// DEMO-SPECIFIC RASTERIZER (Velocity RT)
// ==========================================

static void draw_triangle_velocity_tile(
    shs::RT_ColorDepthVelocity &rt,
    const std::vector<glm::vec3> &vertices,
    const std::vector<glm::vec3> &normals,
    std::function<shs::Varyings(const glm::vec3&, const glm::vec3&)> vertex_shader,
    std::function<FragOutput(const shs::Varyings&)> fragment_shader,
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

            if (rt.depth.test_and_set_depth(px, py, z)) {

                shs::Varyings interpolated;

                interpolated.position      = bc.x * vout[0].position      + bc.y * vout[1].position      + bc.z * vout[2].position;
                interpolated.prev_position = bc.x * vout[0].prev_position + bc.y * vout[1].prev_position + bc.z * vout[2].prev_position;

                interpolated.normal    = glm::normalize(bc.x * vout[0].normal    + bc.y * vout[1].normal    + bc.z * vout[2].normal);
                interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                interpolated.view_z    = z;

                FragOutput out = fragment_shader(interpolated);

                rt.color.draw_pixel_screen_space(px, py, out.first);

                int canvas_y = (h - 1) - py;
                if (rt.velocity.in_bounds(px, canvas_y)) {
                    rt.velocity.at(px, canvas_y) = out.second;
                }
            }
        }
    }
}

// ==========================================
// POST PROCESS: MOTION BLUR (JOB SYSTEM)
// ==========================================

static void post_process_motion_blur(
    const shs::RT_ColorDepthVelocity& src,
    shs::Canvas& dst,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wait_group)
{
    int W = src.color.get_width();
    int H = src.color.get_height();

    const auto& col = src.color.buffer();
    const auto& vel = src.velocity;
    auto& out = dst.buffer();

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

                        glm::vec2 v = vel.at(x, y) * BLUR_MULTIPLIER;

                        if (glm::dot(v, v) < MIN_VEL2_THRESHOLD) {
                            out.at(x, y) = col.at(x, y);
                            continue;
                        }

                        float speed = glm::length(v);
                        int samples = (int)speed;
                        if (samples < 2) samples = 2;
                        if (samples > MAX_BLUR_SAMPLES) samples = MAX_BLUR_SAMPLES;

                        float r = 0.0f, g = 0.0f, b = 0.0f;
                        float wsum = 0.0f;

                        for (int i = 0; i < samples; i++) {
                            float t = (samples == 1) ? 0.0f : ((float)i / (float)(samples - 1));

                            int sx = (int)(x - v.x * t);
                            int sy = (int)(y - v.y * t);

                            if (!col.in_bounds(sx, sy)) continue;

                            shs::Color c = col.at(sx, sy);
                            r += (float)c.r;
                            g += (float)c.g;
                            b += (float)c.b;
                            wsum += 1.0f;
                        }

                        if (wsum <= 0.0f) {
                            out.at(x, y) = col.at(x, y);
                            continue;
                        }

                        r /= wsum; g /= wsum; b /= wsum;
                        out.at(x, y) = color_from_rgbaf(r, g, b, 255.0f);
                    }
                }

                wait_group.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wait_group.wait();
}

// ==========================================
// RENDERER SYSTEM (THREADED) -> RT_ColorDepthVelocity
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, shs::RT_ColorDepthVelocity *target)
        : scene(scene), job_system(job_sys), target(target)
    {}

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
                        uniforms.model         = monkey->get_world_matrix();
                        uniforms.view          = view;
                        uniforms.mvp           = proj * view * uniforms.model;

                        // previous
                        uniforms.prev_mvp       = proj * prev_view_matrix * monkey->get_prev_world_matrix();

                        uniforms.light_dir     = this->scene->light_direction;
                        uniforms.camera_pos    = this->scene->viewer->position;
                        uniforms.color         = monkey->color;
                        uniforms.viewport_size = glm::vec2((float)w, (float)h);

                        const auto& verts = monkey->geometry->triangles;
                        const auto& norms = monkey->geometry->normals;

                        for (size_t i = 0; i < verts.size(); i += 3)
                        {
                            std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                            std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                            draw_triangle_velocity_tile(
                                *this->target,
                                tri_verts,
                                tri_norms,
                                [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                                    return velocity_vertex_shader(p, n, uniforms);
                                },
                                [&uniforms](const shs::Varyings& v) {
                                    return velocity_fragment_shader(v, uniforms);
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

    inline void set_prev_view(glm::mat4 m) { prev_view_matrix = m; }

private:
    HelloScene                          *scene;
    shs::Job::ThreadedPriorityJobSystem *job_system;
    shs::RT_ColorDepthVelocity          *target;
    shs::Job::WaitGroup                  wait_group;

    glm::mat4 prev_view_matrix = glm::mat4(1.0f);
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
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, shs::RT_ColorDepthVelocity *target)
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

    inline void set_prev_view(glm::mat4 m) { renderer_system->set_prev_view(m); }

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

    Viewer     *viewer      = new Viewer(glm::vec3(0.0f, 5.0f, -26.0f), 50.0f);
    HelloScene *hello_scene = new HelloScene(viewer);

    shs::RT_ColorDepthVelocity rt_scene(CANVAS_WIDTH, CANVAS_HEIGHT, viewer->camera->z_near, viewer->camera->z_far, shs::Color{20,20,25,255});
    shs::Canvas post_canvas(CANVAS_WIDTH, CANVAS_HEIGHT);

    SystemProcessor *sys = new SystemProcessor(hello_scene, job_system, &rt_scene);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();

    bool is_dragging = false;

    glm::mat4 prev_view_matrix = viewer->camera->view_matrix;

    shs::Job::WaitGroup blur_wait_group;

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

        // Save previous view before camera update
        prev_view_matrix = viewer->camera->view_matrix;
        sys->set_prev_view(prev_view_matrix);

        sys->process(delta_time);
        sys->render(delta_time);

        post_process_motion_blur(rt_scene, post_canvas, job_system, blur_wait_group);

        screen_canvas->buffer() = post_canvas.buffer();

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
