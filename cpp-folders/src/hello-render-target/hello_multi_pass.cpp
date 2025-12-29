/*
    3D Software Renderer - MULTI-PASS + Per-Object Motion Blur (John Chapman style) + Auto-Focus DOF
    - Pass 0: Render scene into RT_ColorDepthMotion (Color + Depth + MotionVec)
    - Pass 1: Per-Object Motion Blur: rt_scene.color + rt_scene.motion -> mb_out
    - Pass 2: DOF:
        - Copy mb_out -> sharp_copy
        - Gaussian blur mb_out -> blur_img (ping/pong)
        - Auto-focus from rt_scene.depth (median center)
        - Composite sharp vs blur using CoC -> dof_out
    - Pass 3: Fog (depth-based): dof_out + rt_scene.depth -> fog_out
    - Pass 4: Outline (depth-based): fog_out + rt_scene.depth -> final_out
    - Present: final_out -> SDL

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
#define SPACING_X         10.5f
#define SPACING_Z         12.5f
#define START_Z           14.0f
#define BASE_Y            0.0f

// Faster motion
#define WOBBLE_SPEED_MULT 0.5f
#define ROTATE_SPEED_MULT 1.5f

// ===============================
// JOHN CHAPMAN STYLE MOTION BLUR CONFIG
// ===============================
static const int   MB_SAMPLES   = 12;     // 8..16
static const float MB_STRENGTH  = 1.0f;   // 0.5..2.0
static const float MB_MAX_PIXELS = 40.0f; // clamp in pixels (canvas coords)

// ===============================
// OUTLINE PASS CONFIG
// ===============================
static const int   EDGE_RADIUS    = 1;       // 1 or 2
static const float EDGE_THRESHOLD = 0.75f;   // smaller = more edges
static const float EDGE_STRENGTH  = 0.15f;   // 0..1 (higher = darker lines)

// ===============================
// FOG PASS CONFIG
// ===============================
static const shs::Color FOG_COLOR = shs::Color{ 28, 30, 38, 255 };
static const float FOG_START_Z    = 20.0f;
static const float FOG_END_Z      = 80.0f;
static const float FOG_POWER      = 1.25f;

// ===============================
// DOF CONFIG
// ===============================
static const bool ENABLE_DOF       = true;
static const int  BLUR_ITERATIONS  = 4;
static const int  AUTOFOCUS_RADIUS = 6;
static float      DOF_RANGE        = 34.0f;
static float      DOF_MAXBLUR      = 0.75f;

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

// ==========================================
// UNIFORMS & SHADERS (Blinn-Phong)
// ==========================================

struct Uniforms {
    glm::mat4  mvp;
    glm::mat4  prev_mvp;
    glm::mat4  model;
    glm::mat4  view;
    glm::vec3  light_dir;
    glm::vec3  camera_pos;
    shs::Color color;
};

struct VaryingsMB
{
    glm::vec4 position;      // current clip
    glm::vec4 prev_position; // previous clip
    glm::vec3 world_pos;
    glm::vec3 normal;
    glm::vec2 uv;
    float     view_z;
};

static VaryingsMB blinn_phong_vertex_shader_mb(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    VaryingsMB out;
    out.position       = u.mvp * glm::vec4(aPos, 1.0f);
    out.prev_position  = u.prev_mvp * glm::vec4(aPos, 1.0f);

    out.world_pos      = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal         = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv             = glm::vec2(0.0f);

    glm::vec4 view_pos = u.view * u.model * glm::vec4(aPos, 1.0f);
    out.view_z         = view_pos.z; // my convention: forward is +z

    return out;
}

static shs::Color blinn_phong_fragment_shader(const VaryingsMB& in, const Uniforms& u)
{
    glm::vec3 norm     = glm::normalize(in.normal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - in.world_pos);

    float ambientStrength = 0.35f;
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
                    for (int k = 0; k < 3; k++) {
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
// + per-object motion blur state (prev_mvp updated after render)
// ==========================================

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(ModelGeometry* geom, glm::vec3 base_pos, shs::Color color, int idx)
    {
        this->geometry       = geom;
        this->base_position  = base_pos;
        this->position       = base_pos;
        this->scale          = glm::vec3(MONKEY_SCALE);
        this->color          = color;

        this->rotate_enabled   = (idx % 2 == 0);
        this->rotate_speed_deg = (20.0f + 12.0f * float(idx % 4)) * ROTATE_SPEED_MULT;

        this->bob_speed = (0.6f + 0.25f * float(idx)) * WOBBLE_SPEED_MULT;
        this->bob_amp   = 0.8f + 0.15f * float(idx % 3);
        this->phase     = 1.37f * float(idx);

        this->time_accum      = 0.0f;
        this->rotation_angle  = 0.0f;

        this->has_prev_mvp = false;
        this->prev_mvp     = glm::mat4(1.0f);
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

    bool      has_prev_mvp;
    glm::mat4 prev_mvp;
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
// MOTION BUFFER + RT (Color+Depth+Motion)
// motion stored in CANVAS coords (x right, y up) in pixels
// ==========================================

struct MotionBuffer
{
    MotionBuffer() : w(0), h(0) {}

    MotionBuffer(int W, int H)
    {
        init(W, H);
    }

    void init(int W, int H)
    {
        w = W;
        h = H;
        vel.assign((size_t)w * (size_t)h, glm::vec2(0.0f));
    }

    inline void clear()
    {
        std::fill(vel.begin(), vel.end(), glm::vec2(0.0f));
    }

    inline glm::vec2 get(int x, int y) const
    {
        x = clampi(x, 0, w - 1);
        y = clampi(y, 0, h - 1);
        return vel[(size_t)y * (size_t)w + (size_t)x];
    }

    inline void set(int x, int y, const glm::vec2& v)
    {
        if (x < 0 || x >= w || y < 0 || y >= h) return;
        vel[(size_t)y * (size_t)w + (size_t)x] = v;
    }

    int w, h;
    std::vector<glm::vec2> vel;
};

struct RT_ColorDepthMotion
{
    RT_ColorDepthMotion(int W, int H, float zn, float zf, shs::Color clear_col)
        : color(W, H, clear_col), depth(W, H, zn, zf), motion(W, H)
    {
        (void)zn; (void)zf;
        clear(clear_col);
    }

    inline void clear(shs::Color c)
    {
        color.buffer().clear(c);
        depth.clear();
        motion.clear();
    }

    inline int width() const  { return color.get_width(); }
    inline int height() const { return color.get_height(); }

    shs::Canvas   color;
    shs::ZBuffer  depth;
    MotionBuffer  motion;
};

// ==========================================
// TILED RASTERIZER (writes depth in CANVAS coords + writes motion vec per pixel)
// ==========================================

static inline glm::vec2 clip_to_screen_xy(const glm::vec4& clip, int w, int h)
{
    glm::vec3 s = shs::Canvas::clip_to_screen(clip, w, h); // screen coords (x right, y down)
    return glm::vec2(s.x, s.y);
}

static void draw_triangle_tile_color_depth_motion(
    RT_ColorDepthMotion &rt,
    const std::vector<glm::vec3> &vertices,
    const std::vector<glm::vec3> &normals,
    std::function<VaryingsMB(const glm::vec3&, const glm::vec3&)> vertex_shader,
    std::function<shs::Color(const VaryingsMB&)> fragment_shader,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    int w = rt.color.get_width();
    int h = rt.color.get_height();

    VaryingsMB vout[3];
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

                VaryingsMB interpolated;
                interpolated.normal     = glm::normalize(bc.x * vout[0].normal     + bc.y * vout[1].normal     + bc.z * vout[2].normal);
                interpolated.world_pos  = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                interpolated.view_z     = z;

                // interpolate clip positions (John Chapman style: prev vs curr)
                interpolated.position      = bc.x * vout[0].position      + bc.y * vout[1].position      + bc.z * vout[2].position;
                interpolated.prev_position = bc.x * vout[0].prev_position + bc.y * vout[1].prev_position + bc.z * vout[2].prev_position;

                // velocity computed in screen coords then converted to canvas coords
                glm::vec2 curr_s = clip_to_screen_xy(interpolated.position, w, h);
                glm::vec2 prev_s = clip_to_screen_xy(interpolated.prev_position, w, h);
                glm::vec2 v_screen = curr_s - prev_s;      // y down
                glm::vec2 v_canvas = glm::vec2(v_screen.x, -v_screen.y); // y up

                // clamp to avoid extreme streaks
                float len = glm::length(v_canvas);
                if (len > MB_MAX_PIXELS && len > 0.0001f) {
                    v_canvas = v_canvas * (MB_MAX_PIXELS / len);
                }

                rt.motion.set(px, cy, v_canvas);

                rt.color.draw_pixel_screen_space(px, py, fragment_shader(interpolated));
            }
        }
    }
}

// ==========================================
// PASS 1: PER-OBJECT MOTION BLUR (post)
// ==========================================

static void motion_blur_pass(
    const shs::Canvas& src,
    const MotionBuffer& motion,
    shs::Canvas& dst,
    int samples,
    float strength,
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

                        glm::vec2 v = motion.get(x, y) * strength;
                        float vlen = glm::length(v);

                        if (vlen < 0.001f || samples <= 1) {
                            dst.draw_pixel(x, y, src.get_color_at(x, y));
                            continue;
                        }

                        glm::vec2 dir = v / vlen;

                        float r = 0, g = 0, b = 0;
                        float wsum = 0.0f;

                        // sample along -v..+v (center weighted)
                        // (simple linear weights; keeps it stable on CPU)
                        for (int i = 0; i < samples; i++) {
                            float t = (samples == 1) ? 0.0f : (float(i) / float(samples - 1));
                            float a = (t - 0.5f) * 2.0f; // -1..+1
                            glm::vec2 p = glm::vec2(float(x), float(y)) + dir * (a * vlen);

                            int sx = clampi((int)std::round(p.x), 0, W - 1);
                            int sy = clampi((int)std::round(p.y), 0, H - 1);

                            float wgt = 1.0f - std::abs(a); // center heavier
                            shs::Color c = src.get_color_at(sx, sy);

                            r += wgt * float(c.r);
                            g += wgt * float(c.g);
                            b += wgt * float(c.b);
                            wsum += wgt;
                        }

                        if (wsum < 0.0001f) wsum = 1.0f;
                        dst.draw_pixel(x, y, shs::Color{
                            (uint8_t)clampi((int)(r / wsum), 0, 255),
                            (uint8_t)clampi((int)(g / wsum), 0, 255),
                            (uint8_t)clampi((int)(b / wsum), 0, 255),
                            255
                        });
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

// ==========================================
// PASS: OUTLINE (depth-based)
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
// PASS: FOG (depth-based)
// ==========================================

static void fog_pass(
    const shs::Canvas& src,
    const shs::ZBuffer& depth,
    shs::Canvas& dst,
    shs::Color fog_color,
    float fog_start,
    float fog_end,
    float fog_power,
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
                        t = std::pow(t, fog_power);

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
// GAUSSIAN BLUR (JOB SYSTEM) for DOF
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
// RENDERER SYSTEM (Threaded) -> RT_ColorDepthMotion
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, RT_ColorDepthMotion *rt)
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

                        glm::mat4 model = monkey->get_world_matrix();
                        glm::mat4 mvp   = proj * view * model;

                        glm::mat4 prev_mvp = mvp;
                        if (monkey->has_prev_mvp) prev_mvp = monkey->prev_mvp;

                        Uniforms uniforms;
                        uniforms.model      = model;
                        uniforms.view       = view;
                        uniforms.mvp        = mvp;
                        uniforms.prev_mvp   = prev_mvp;
                        uniforms.light_dir  = scene->light_direction;
                        uniforms.camera_pos = scene->viewer->position;
                        uniforms.color      = monkey->color;

                        const auto& verts = monkey->geometry->triangles;
                        const auto& norms = monkey->geometry->normals;

                        for (size_t i = 0; i < verts.size(); i += 3)
                        {
                            std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                            std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                            draw_triangle_tile_color_depth_motion(
                                *rt,
                                tri_verts,
                                tri_norms,
                                [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                                    return blinn_phong_vertex_shader_mb(p, n, uniforms);
                                },
                                [&uniforms](const VaryingsMB& v) {
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

        // after render: commit prev_mvp for next frame (John Chapman per-object history)
        glm::mat4 view2 = scene->viewer->camera->view_matrix;
        glm::mat4 proj2 = scene->viewer->camera->projection_matrix;

        for (shs::AbstractObject3D *object : scene->scene_objects)
        {
            MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
            if (!monkey) continue;

            glm::mat4 model = monkey->get_world_matrix();
            monkey->prev_mvp = proj2 * view2 * model;
            monkey->has_prev_mvp = true;
        }
    }

private:
    HelloScene                          *scene;
    shs::Job::ThreadedPriorityJobSystem *job_system;
    RT_ColorDepthMotion                 *rt;
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
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys, RT_ColorDepthMotion *rt)
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

    Viewer     *viewer = new Viewer(glm::vec3(0.0f, 6.0f, -28.0f), 50.0f);
    HelloScene *scene  = new HelloScene(viewer);

    // Pass 0 RT: Color + Depth + Motion
    RT_ColorDepthMotion rt_scene(CANVAS_WIDTH, CANVAS_HEIGHT, viewer->camera->z_near, viewer->camera->z_far, shs::Color{20,20,25,255});

    // Pass 1 output (motion blur)
    shs::Canvas mb_out(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    // DOF buffers
    shs::Canvas sharp_copy(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas blur_ping (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas blur_pong (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas dof_out   (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    // Fog/Outline ping-pong
    shs::Canvas ping(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas pong(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    SystemProcessor *sys = new SystemProcessor(scene, job_system, &rt_scene);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();

    bool is_dragging = false;

    shs::Job::WaitGroup wg_mb;
    shs::Job::WaitGroup wg_blur;
    shs::Job::WaitGroup wg_dof;
    shs::Job::WaitGroup wg_fog;
    shs::Job::WaitGroup wg_outline;

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

        // Pass 0: logic + render to rt_scene
        sys->process(delta_time);
        sys->render(delta_time);

        // Pass 1: John Chapman per-object motion blur
        motion_blur_pass(rt_scene.color, rt_scene.motion, mb_out, MB_SAMPLES, MB_STRENGTH, job_system, wg_mb);

        // Pass 2: DOF (on motion blurred image, depth from rt_scene)
        if (ENABLE_DOF)
        {
            sharp_copy.buffer() = mb_out.buffer();
            blur_pong.buffer()  = sharp_copy.buffer();

            for (int i = 0; i < BLUR_ITERATIONS; i++)
            {
                gaussian_blur_pass(blur_pong, blur_ping, true,  job_system, wg_blur);
                gaussian_blur_pass(blur_ping, blur_pong, false, job_system, wg_blur);
            }

            int cx = CANVAS_WIDTH  / 2;
            int cy = CANVAS_HEIGHT / 2;
            float focus_depth = autofocus_depth_median_center(rt_scene.depth, cx, cy, AUTOFOCUS_RADIUS);

            dof_composite_pass(
                sharp_copy,
                blur_pong,
                rt_scene.depth,
                dof_out,
                focus_depth,
                DOF_RANGE,
                DOF_MAXBLUR,
                job_system,
                wg_dof
            );
        }
        else
        {
            dof_out.buffer() = mb_out.buffer();
        }

        // Pass 3: Fog (dof_out -> pong)
        fog_pass(dof_out, rt_scene.depth, pong, FOG_COLOR, FOG_START_Z, FOG_END_Z, FOG_POWER, job_system, wg_fog);

        // Pass 4: Outline (pong -> ping)
        outline_pass(pong, rt_scene.depth, ping, job_system, wg_outline);

        // Present ping
        screen_canvas->buffer() = ping.buffer();
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
