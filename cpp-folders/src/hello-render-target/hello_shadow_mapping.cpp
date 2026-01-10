/*
    3D Software Renderer - WORKING SHADOW MAPPING (Z+ Forward Convention)
    - Pass 0: Shadow depth (light ortho) into ZBuffer [0..1]
    - Pass 1: Main forward render with PCF shadow lookup
    - Post: DOF + Outline + Motion Blur

    Coordinate Conventions (IMPORTANT):
    Model/World/View/Projection/NDC: +Z is forward, +Y up, +X right.
    Screen: origin top-left, +X right, +Y down (raster happens here).
    shs::Canvas/ZBuffer: origin bottom-left, +X right, +Y up.
    Screen -> Canvas: y_canvas = (H - 1) - y_screen.
*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <random>

// External Libraries
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

// ===============================
// CONFIGURATION
// ===============================
#define WINDOW_WIDTH      800
#define WINDOW_HEIGHT     600
#define CANVAS_WIDTH      380
#define CANVAS_HEIGHT     280
#define MOUSE_SENSITIVITY 0.2f

#define THREAD_COUNT      16
#define TILE_SIZE_X       80
#define TILE_SIZE_Y       80

// Shadow Config
#define SHADOW_MAP_SIZE   1024
#define SHADOW_ORTHO_SIZE 48.0f
#define SHADOW_NEAR       1.0f
#define SHADOW_FAR        260.0f
#define SHADOW_PCF_RADIUS 1
#define SHADOW_BIAS_BASE  0.0015f
#define SHADOW_BIAS_SLOPE 0.0030f
#define SHADOW_DARKNESS   0.30f

// Scene Config
#define GRID_X            3
#define GRID_Z            3
#define MONKEY_SCALE_BASE 3.2f
#define SPACING_X         10.5f
#define SPACING_Z         12.5f
#define START_Z           14.0f
#define BASE_Y            0.0f

// Floor
#define FLOOR_Y           -3.0f
#define FLOOR_SIZE        120.0f
#define FLOOR_DIVS        24

// Motion Blur Config
static const int   MB_SAMPLES      = 8;
static const float MB_STRENGTH     = 0.85f;
static const float MB_MAX_PIXELS   = 22.0f;
static const bool  MB_SOFT_KNEE    = true;
static const float MB_KNEE_PIXELS  = 18.0f;

// Outline Config
static const int   EDGE_RADIUS      = 1;
static const float EDGE_THRESHOLD   = 0.75f;
static const float EDGE_STRENGTH    = 0.15f;

// DOF Config
static const bool  ENABLE_DOF       = true;
static const int   BLUR_ITERATIONS  = 3;
static const int   AUTOFOCUS_RADIUS = 6;
static const float DOF_RANGE        = 34.0f;
static const float DOF_MAXBLUR      = 0.75f;

// ===============================
// UTILS
// ===============================
static inline int clampi(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static inline float smoothstep01(float t) {
    t = clampf(t, 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

static inline shs::Color lerp_color(const shs::Color& a, const shs::Color& b, float t) {
    t = clampf(t, 0.0f, 1.0f);
    float ia = 1.0f - t;
    return shs::Color{
        (uint8_t)(ia * a.r + t * b.r),
        (uint8_t)(ia * a.g + t * b.g),
        (uint8_t)(ia * a.b + t * b.b),
        255
    };
}

static inline shs::Color color_from_rgbaf(float r, float g, float b, float a) {
    return shs::Color{
        (uint8_t)clampi((int)r, 0, 255),
        (uint8_t)clampi((int)g, 0, 255),
        (uint8_t)clampi((int)b, 0, 255),
        (uint8_t)clampi((int)a, 0, 255)
    };
}

static inline glm::vec3 color_to_vec3(const shs::Color& c) {
    return glm::vec3(c.r, c.g, c.b) / 255.0f;
}

static inline shs::Color vec3_to_color(const glm::vec3& v) {
    glm::vec3 c = glm::clamp(v, 0.0f, 1.0f) * 255.0f;
    return shs::Color{ (uint8_t)c.r, (uint8_t)c.g, (uint8_t)c.b, 255 };
}

// Convert CLIP -> NDC Z01 using your convention (+Z forward)
static inline float ndc_z01_from_clip(const glm::vec4& clip) {
    float ndc_z = clip.z / clip.w;      // [-1..1]
    return ndc_z * 0.5f + 0.5f;         // [0..1]
}

// Screen -> Canvas Y conversion
static inline int screen_y_to_canvas_y(int y_screen, int H) {
    return (H - 1) - y_screen;
}

// ===============================
// Z+ FORWARD MATRICES (IMPORTANT)
// ===============================
static glm::mat4 lookAt_ZForward(const glm::vec3& eye, const glm::vec3& target, const glm::vec3& up) {
    // Camera looks toward +Z direction in its local space
    glm::vec3 f = glm::normalize(target - eye);          // forward (+Z)
    glm::vec3 r = glm::normalize(glm::cross(up, f));     // right
    glm::vec3 u = glm::cross(f, r);                      // up (re-orthonormalized)

    glm::mat4 M(1.0f);
    // Column-major (glm): columns are basis vectors
    M[0][0] = r.x; M[1][0] = r.y; M[2][0] = r.z;
    M[0][1] = u.x; M[1][1] = u.y; M[2][1] = u.z;
    M[0][2] = f.x; M[1][2] = f.y; M[2][2] = f.z;

    M[3][0] = -glm::dot(r, eye);
    M[3][1] = -glm::dot(u, eye);
    M[3][2] = -glm::dot(f, eye);
    return M;
}

static glm::mat4 ortho_ZForward(float l, float r, float b, float t, float n, float f) {
    // Maps z in [n..f] (along +Z) to NDC z in [-1..1]
    glm::mat4 M(1.0f);
    M[0][0] =  2.0f / (r - l);
    M[1][1] =  2.0f / (t - b);
    M[2][2] =  2.0f / (f - n);

    M[3][0] = -(r + l) / (r - l);
    M[3][1] = -(t + b) / (t - b);
    M[3][2] = -(f + n) / (f - n);
    return M;
}

// ===============================
// UNIFORMS & VARYINGS
// ===============================
struct Uniforms {
    glm::mat4  vp;
    glm::mat4  model;
    glm::mat4  prev_vp_model;

    glm::mat4  view;
    glm::vec3  light_dir;
    glm::vec3  camera_pos;
    shs::Color color;

    glm::mat4      light_vp;
    const shs::ZBuffer* shadow_map;
    int shadow_w;
    int shadow_h;
};

struct VaryingsMain {
    glm::vec4 position;
    glm::vec4 prev_position;
    glm::vec3 world_pos;
    glm::vec3 normal;
    float     depth_view;
    glm::vec4 light_clip;
};

struct VaryingsShadow {
    glm::vec4 position;
    float     depth01;
};

// ===============================
// GEOMETRY
// ===============================
class ModelGeometry {
public:
    ModelGeometry(const std::string& model_path) {
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(
            model_path.c_str(),
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_FlipUVs |
            aiProcess_JoinIdenticalVertices
        );
        if (!scene || !scene->mRootNode) { std::cerr << "Assimp Error" << std::endl; return; }

        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                if (mesh->mFaces[j].mNumIndices != 3) continue;
                for (int k = 0; k < 3; k++) {
                    aiVector3D v = mesh->mVertices[mesh->mFaces[j].mIndices[k]];
                    triangles.push_back(glm::vec3(v.x, v.y, v.z));
                    if (mesh->HasNormals()) {
                        aiVector3D n = mesh->mNormals[mesh->mFaces[j].mIndices[k]];
                        normals.push_back(glm::vec3(n.x, n.y, n.z));
                    } else normals.push_back(glm::vec3(0, 1, 0));
                }
            }
        }
    }
    std::vector<glm::vec3> triangles, normals;
};

struct GroundPlane {
    std::vector<glm::vec3> verts, norms;
    shs::Color color;

    GroundPlane() {
        color = shs::Color{ 160, 160, 165, 255 };
        float size = FLOOR_SIZE, step = size / float(FLOOR_DIVS), start = -size * 0.5f;
        glm::vec3 up(0, 1, 0);

        for (int z = 0; z < FLOOR_DIVS; z++) {
            for (int x = 0; x < FLOOR_DIVS; x++) {
                float x0 = start + x * step, z0 = start + z * step;
                float x1 = x0 + step, z1 = z0 + step;

                // Winding for your engine culling (keep as you had)
                verts.push_back(glm::vec3(x0, FLOOR_Y, z0)); norms.push_back(up);
                verts.push_back(glm::vec3(x1, FLOOR_Y, z1)); norms.push_back(up);
                verts.push_back(glm::vec3(x1, FLOOR_Y, z0)); norms.push_back(up);

                verts.push_back(glm::vec3(x0, FLOOR_Y, z0)); norms.push_back(up);
                verts.push_back(glm::vec3(x0, FLOOR_Y, z1)); norms.push_back(up);
                verts.push_back(glm::vec3(x1, FLOOR_Y, z1)); norms.push_back(up);
            }
        }
    }
};

class MonkeyObject : public shs::AbstractObject3D {
public:
    MonkeyObject(ModelGeometry* geom, const glm::vec3& base_pos, const shs::Color& col, int idx, float scale_mult)
        : geometry(geom), base_position(base_pos), position(base_pos), scale(glm::vec3(MONKEY_SCALE_BASE * scale_mult)), color(col)
    {
        bob_speed = (0.6f + 0.25f * float(idx % 4)) * 2.5f;
        bob_amp   = 0.8f + 0.15f * float(idx % 3);
        phase     = 1.37f * float(idx);
        rotate_speed = (20.0f + 10.0f * float(idx % 3));
        rotation = 0.0f;
        time_accum = 0.0f;
        has_prev_vp_model = false;
        prev_vp_model = glm::mat4(1.0f);
    }

    void update(float dt) override {
        time_accum += dt;
        position.y = base_position.y + std::sin(time_accum * bob_speed + phase) * bob_amp;
        rotation += rotate_speed * dt;
        if (rotation > 360.0f) rotation -= 360.0f;
    }

    glm::mat4 get_world_matrix() override {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(rotation), glm::vec3(0, 1, 0));
        return glm::scale(t * r, scale);
    }

    void render() override {}

    ModelGeometry* geometry;
    glm::vec3 base_position, position, scale;
    shs::Color color;
    float time_accum, bob_speed, bob_amp, phase, rotation, rotate_speed;
    bool has_prev_vp_model;
    glm::mat4 prev_vp_model;
};

// ===============================
// SCENE STATE
// ===============================
class HelloScene : public shs::AbstractSceneState {
public:
    HelloScene() {
        viewer = new shs::Camera3D();
        viewer->width  = float(CANVAS_WIDTH);
        viewer->height = float(CANVAS_HEIGHT);
        viewer->position = glm::vec3(0.0f, 6.0f, -28.0f);
        viewer->vertical_angle = -15.0f;

        monkey_geo = new ModelGeometry("./obj/monkey/monkey.rawobj");
        ground = new GroundPlane();

        light_target = glm::vec3(0, 0, 25);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> scale_dist(0.7f, 1.45f);

        int idx = 0;
        for (int gz = 0; gz < GRID_Z; gz++) {
            for (int gx = 0; gx < GRID_X; gx++) {
                float x = (float(gx) - float(GRID_X - 1) * 0.5f) * SPACING_X;
                float z = START_Z + float(gz) * SPACING_Z;

                shs::Color c;
                int m = idx % 3;
                if (m == 0) c = {200,  90,  80, 255};
                else if (m == 1) c = { 80, 200, 120, 255};
                else             c = { 60, 100, 200, 255};

                monkeys.push_back(new MonkeyObject(monkey_geo, glm::vec3(x, BASE_Y, z), c, idx, scale_dist(rng)));
                idx++;
            }
        }
    }

    ~HelloScene() {
        delete viewer; delete monkey_geo; delete ground;
        for (auto* m : monkeys) delete m;
    }

    void process() override {
        light_orbit_angle += 0.015f;
        float r = 60.0f;
        light_pos = glm::vec3(std::sin(light_orbit_angle) * r, 70.0f, std::cos(light_orbit_angle) * r + 20.0f);
        light_dir = glm::normalize(light_target - light_pos);
    }

    shs::Camera3D* viewer;
    ModelGeometry* monkey_geo;
    GroundPlane*   ground;
    std::vector<MonkeyObject*> monkeys;
    glm::vec3 light_pos, light_dir, light_target;
    float light_orbit_angle = 0.0f;
};

// ===============================
// RENDER BUFFERS
// ===============================
struct MotionBuffer {
    int w, h;
    std::vector<glm::vec2> vel;
    MotionBuffer(int W, int H) : w(W), h(H), vel(W * H, glm::vec2(0)) {}
    void clear() { std::fill(vel.begin(), vel.end(), glm::vec2(0)); }
    void set(int x, int y_canvas, glm::vec2 v) { if (x >= 0 && x < w && y_canvas >= 0 && y_canvas < h) vel[y_canvas * w + x] = v; }
    glm::vec2 get(int x, int y_screen) const {
        int cy = (h - 1) - clampi(y_screen, 0, h - 1);
        return vel[cy * w + clampi(x, 0, w - 1)];
    }
};

struct RT_Shadow {
    shs::ZBuffer depth;
    shs::Canvas  dummy;
    RT_Shadow(int size) : depth(size, size, 0.0f, 1.0f), dummy(size, size, shs::Color{0,0,0,0}) {}
    void clear() { depth.clear(); }
};

struct RT_Main {
    shs::Canvas  color;
    shs::ZBuffer depth;
    MotionBuffer motion;
    RT_Main(int w, int h) : color(w, h), depth(w, h, 0.0f, 1000.0f), motion(w, h) {}
    void clear(shs::Color c) { color.buffer().clear(c); depth.clear(); motion.clear(); }
};

// ===============================
// SHADERS
// ===============================
static VaryingsShadow shadow_vs(const glm::vec3& aPos, const Uniforms& u) {
    VaryingsShadow out;
    glm::vec4 clip = u.light_vp * u.model * glm::vec4(aPos, 1.0f);
    out.position = clip;
    out.depth01 = ndc_z01_from_clip(clip);
    return out;
}

static VaryingsMain main_vs(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u) {
    VaryingsMain out;
    glm::vec4 wp4 = u.model * glm::vec4(aPos, 1.0f);
    out.world_pos = glm::vec3(wp4);
    out.normal = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);

    out.position = u.vp * wp4;
    out.prev_position = u.prev_vp_model * glm::vec4(aPos, 1.0f);

    out.depth_view = std::abs((u.view * wp4).z);
    out.light_clip = u.light_vp * wp4;
    return out;
}

// Shadow clip -> shadowmap texel (screen space first, then stored in canvas coords)
static inline bool shadowmap_texel_from_lightclip(
    const glm::vec4& light_clip,
    int shadow_w, int shadow_h,
    int& sx, int& sy_canvas,
    float& z01
) {
    if (light_clip.w == 0.0f) return false;
    glm::vec3 ndc = glm::vec3(light_clip) / light_clip.w; // [-1..1]
    if (ndc.x < -1.0f || ndc.x > 1.0f || ndc.y < -1.0f || ndc.y > 1.0f || ndc.z < -1.0f || ndc.z > 1.0f) return false;

    float u01 = ndc.x * 0.5f + 0.5f;
    float v01 = ndc.y * 0.5f + 0.5f;
    z01 = ndc.z * 0.5f + 0.5f;

    int x_screen = clampi((int)std::floor(u01 * float(shadow_w - 1) + 0.5f), 0, shadow_w - 1);
    int y_screen = clampi((int)std::floor((1.0f - v01) * float(shadow_h - 1) + 0.5f), 0, shadow_h - 1);
    // NOTE: screen y is top-left origin; v01 is NDC up, so we invert for screen.
    // Then convert to canvas storage.
    sx = x_screen;
    sy_canvas = screen_y_to_canvas_y(y_screen, shadow_h);
    return true;
}

static float shadow_factor_pcf(const glm::vec4& light_clip, const glm::vec3& normal, const Uniforms& u) {
    if (!u.shadow_map) return 1.0f;

    int sx_center, sy_center_canvas;
    float z01;
    if (!shadowmap_texel_from_lightclip(light_clip, u.shadow_w, u.shadow_h, sx_center, sy_center_canvas, z01)) return 1.0f;

    glm::vec3 L = glm::normalize(-u.light_dir);
    float ndotl = glm::max(glm::dot(glm::normalize(normal), L), 0.0f);
    float bias = SHADOW_BIAS_BASE + SHADOW_BIAS_SLOPE * (1.0f - ndotl);

    float acc = 0.0f, wsum = 0.0f;

    for (int oy = -SHADOW_PCF_RADIUS; oy <= SHADOW_PCF_RADIUS; oy++) {
        for (int ox = -SHADOW_PCF_RADIUS; ox <= SHADOW_PCF_RADIUS; ox++) {
            int sx = clampi(sx_center + ox, 0, u.shadow_w - 1);
            int sy = clampi(sy_center_canvas + oy, 0, u.shadow_h - 1);

            float z_sm = u.shadow_map->get_depth_at(sx, sy); // canvas coords
            // If receiver is farther than stored depth + bias => shadowed
            float lit = (z01 <= (z_sm + bias)) ? 1.0f : SHADOW_DARKNESS;
            acc += lit;
            wsum += 1.0f;
        }
    }
    return (wsum > 0.0f) ? (acc / wsum) : 1.0f;
}

static shs::Color main_fs(const VaryingsMain& in, const Uniforms& u) {
    glm::vec3 N = glm::normalize(in.normal);
    glm::vec3 L = glm::normalize(-u.light_dir);
    glm::vec3 V = glm::normalize(u.camera_pos - in.world_pos);

    float diff = glm::max(glm::dot(N, L), 0.0f);
    glm::vec3 H = glm::normalize(L + V);
    float spec = std::pow(glm::max(glm::dot(N, H), 0.0f), 32.0f);

    float shadow = (diff > 0.0f) ? shadow_factor_pcf(in.light_clip, N, u) : 1.0f;

    glm::vec3 base = color_to_vec3(u.color);

    // Checkerboard floor
    if (in.world_pos.y < FLOOR_Y + 0.1f) {
        bool check = ((int(std::floor(in.world_pos.x * 0.5f)) + int(std::floor(in.world_pos.z * 0.5f))) & 1) == 0;
        base *= check ? 1.05f : 0.85f;
    }

    // Slightly brighter ambient (as requested)
    glm::vec3 ambient = 0.45f * glm::vec3(1.0f);

    glm::vec3 lighting = ambient + shadow * (diff * glm::vec3(1.0f) + 0.35f * spec * glm::vec3(1.0f));
    glm::vec3 result = lighting * base;

    return vec3_to_color(result);
}

// ===============================
// RASTERIZER
// ===============================
static inline bool tri_bbox_clamp(
    const glm::vec2& a, const glm::vec2& b, const glm::vec2& c,
    const glm::ivec2& tmin, const glm::ivec2& tmax,
    int W, int H, int& x0, int& x1, int& y0, int& y1
) {
    x0 = clampi((int)std::floor(std::min({a.x, b.x, c.x})), tmin.x, tmax.x);
    x1 = clampi((int)std::ceil (std::max({a.x, b.x, c.x})), tmin.x, tmax.x);
    y0 = clampi((int)std::floor(std::min({a.y, b.y, c.y})), tmin.y, tmax.y);
    y1 = clampi((int)std::ceil (std::max({a.y, b.y, c.y})), tmin.y, tmax.y);
    x0 = clampi(x0, 0, W - 1); x1 = clampi(x1, 0, W - 1);
    y0 = clampi(y0, 0, H - 1); y1 = clampi(y1, 0, H - 1);
    return !(x0 > x1 || y0 > y1);
}

static void draw_triangle_shadow(
    shs::ZBuffer& zbuf,
    const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2,
    const Uniforms& u,
    const glm::ivec2& tmin, const glm::ivec2& tmax
) {
    int W = zbuf.get_width(), H = zbuf.get_height();
    VaryingsShadow v0 = shadow_vs(p0, u), v1 = shadow_vs(p1, u), v2 = shadow_vs(p2, u);

    glm::vec3 sc0 = shs::Canvas::clip_to_screen(v0.position, W, H);
    glm::vec3 sc1 = shs::Canvas::clip_to_screen(v1.position, W, H);
    glm::vec3 sc2 = shs::Canvas::clip_to_screen(v2.position, W, H);

    std::vector<glm::vec2> tri2d = { {sc0.x, sc0.y}, {sc1.x, sc1.y}, {sc2.x, sc2.y} };
    int x0, x1, y0, y1;
    if (!tri_bbox_clamp(tri2d[0], tri2d[1], tri2d[2], tmin, tmax, W, H, x0, x1, y0, y1)) return;

    if (((tri2d[1].x - tri2d[0].x) * (tri2d[2].y - tri2d[0].y) - (tri2d[1].y - tri2d[0].y) * (tri2d[2].x - tri2d[0].x)) == 0.0f) return;

    for (int px = x0; px <= x1; px++) {
        for (int py = y0; py <= y1; py++) {
            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), tri2d);
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

            float z01 = bc.x * v0.depth01 + bc.y * v1.depth01 + bc.z * v2.depth01;

            // Raster is in screen-space (py top-down). ZBuffer storage uses canvas y.
            int cy = screen_y_to_canvas_y(py, H);
            zbuf.test_and_set_depth(px, cy, z01);
        }
    }
}

static void draw_triangle_main(
    RT_Main& rt,
    const glm::vec3& p0, const glm::vec3& n0,
    const glm::vec3& p1, const glm::vec3& n1,
    const glm::vec3& p2, const glm::vec3& n2,
    const Uniforms& u,
    const glm::ivec2& tmin, const glm::ivec2& tmax
) {
    int W = rt.color.get_width(), H = rt.color.get_height();
    VaryingsMain v0 = main_vs(p0, n0, u), v1 = main_vs(p1, n1, u), v2 = main_vs(p2, n2, u);

    glm::vec3 sc0 = shs::Canvas::clip_to_screen(v0.position, W, H);
    glm::vec3 sc1 = shs::Canvas::clip_to_screen(v1.position, W, H);
    glm::vec3 sc2 = shs::Canvas::clip_to_screen(v2.position, W, H);

    std::vector<glm::vec2> tri2d = { {sc0.x, sc0.y}, {sc1.x, sc1.y}, {sc2.x, sc2.y} };
    int x0, x1, y0, y1;
    if (!tri_bbox_clamp(tri2d[0], tri2d[1], tri2d[2], tmin, tmax, W, H, x0, x1, y0, y1)) return;

    if (((tri2d[1].x - tri2d[0].x) * (tri2d[2].y - tri2d[0].y) - (tri2d[1].y - tri2d[0].y) * (tri2d[2].x - tri2d[0].x)) == 0.0f) return;

    for (int px = x0; px <= x1; px++) {
        for (int py = y0; py <= y1; py++) {
            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), tri2d);
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

            int cy = screen_y_to_canvas_y(py, H);
            float dview = bc.x * v0.depth_view + bc.y * v1.depth_view + bc.z * v2.depth_view;

            if (!rt.depth.test_and_set_depth(px, cy, dview)) continue;

            VaryingsMain in;
            in.world_pos = bc.x * v0.world_pos + bc.y * v1.world_pos + bc.z * v2.world_pos;
            in.normal    = bc.x * v0.normal    + bc.y * v1.normal    + bc.z * v2.normal;
            in.light_clip= bc.x * v0.light_clip+ bc.y * v1.light_clip+ bc.z * v2.light_clip;

            glm::vec4 clip_now  = bc.x * v0.position      + bc.y * v1.position      + bc.z * v2.position;
            glm::vec4 clip_prev = bc.x * v0.prev_position + bc.y * v1.prev_position + bc.z * v2.prev_position;

            glm::vec3 csc = shs::Canvas::clip_to_screen(clip_now,  W, H);
            glm::vec3 psc = shs::Canvas::clip_to_screen(clip_prev, W, H);

            glm::vec2 vel_screen = glm::vec2(csc.x, csc.y) - glm::vec2(psc.x, psc.y);

            // Screen Y down -> Canvas Y up : invert Y
            glm::vec2 vel_canvas(vel_screen.x, -vel_screen.y);

            float vlen = glm::length(vel_canvas);
            if (vlen > MB_MAX_PIXELS) vel_canvas = (vel_canvas / vlen) * MB_MAX_PIXELS;
            rt.motion.set(px, cy, vel_canvas);

            rt.color.draw_pixel_screen_space(px, py, main_fs(in, u));
        }
    }
}

// ===============================
// RENDERER SYSTEM
// ===============================
class RendererSystem : public shs::AbstractSystem {
public:
    RendererSystem(HelloScene* scene, shs::Job::ThreadedPriorityJobSystem* job, RT_Main* rt_main, RT_Shadow* rt_shadow)
        : scene(scene), jobs(job), rt(rt_main), shadow(rt_shadow) {}

    void process(float dt) override {
        (void)dt;

        glm::mat4 view = scene->viewer->view_matrix;
        glm::mat4 proj = scene->viewer->projection_matrix;
        glm::mat4 vp = proj * view;

        // Light matrices using YOUR +Z forward convention
        glm::mat4 light_view = lookAt_ZForward(scene->light_pos, scene->light_target, glm::vec3(0,1,0));
        glm::mat4 light_proj = ortho_ZForward(-SHADOW_ORTHO_SIZE, SHADOW_ORTHO_SIZE, -SHADOW_ORTHO_SIZE, SHADOW_ORTHO_SIZE, SHADOW_NEAR, SHADOW_FAR);
        glm::mat4 light_vp = light_proj * light_view;

        shadow->clear();
        run_shadow_pass(light_vp);

        rt->clear(shs::Color{20, 20, 25, 255});
        run_main_pass(vp, view, light_vp);

        for (auto* m : scene->monkeys) {
            m->prev_vp_model = vp * m->get_world_matrix();
            m->has_prev_vp_model = true;
        }
    }

private:
    void run_shadow_pass(const glm::mat4& light_vp) {
        int w = shadow->depth.get_width(), h = shadow->depth.get_height();
        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X, rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
        wg.reset();

        for (int ty = 0; ty < rows; ty++) {
            for (int tx = 0; tx < cols; tx++) {
                glm::ivec2 tmin(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                glm::ivec2 tmax(std::min((tx + 1) * TILE_SIZE_X, w) - 1, std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                wg.add(1);
                jobs->submit({[=]() {
                    Uniforms u;
                    u.light_vp = light_vp;

                    u.model = glm::mat4(1.0f);
                    for (size_t i = 0; i < scene->ground->verts.size(); i += 3) {
                        draw_triangle_shadow(shadow->depth, scene->ground->verts[i], scene->ground->verts[i+1], scene->ground->verts[i+2], u, tmin, tmax);
                    }

                    for (auto* m : scene->monkeys) {
                        u.model = m->get_world_matrix();
                        for (size_t i = 0; i < m->geometry->triangles.size(); i += 3) {
                            draw_triangle_shadow(shadow->depth, m->geometry->triangles[i], m->geometry->triangles[i+1], m->geometry->triangles[i+2], u, tmin, tmax);
                        }
                    }

                    wg.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }
        wg.wait();
    }

    void run_main_pass(const glm::mat4& vp, const glm::mat4& view, const glm::mat4& light_vp) {
        int w = rt->color.get_width(), h = rt->color.get_height();
        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X, rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
        wg.reset();

        for (int ty = 0; ty < rows; ty++) {
            for (int tx = 0; tx < cols; tx++) {
                glm::ivec2 tmin(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                glm::ivec2 tmax(std::min((tx + 1) * TILE_SIZE_X, w) - 1, std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                wg.add(1);
                jobs->submit({[=]() {
                    Uniforms u;
                    u.vp = vp;
                    u.view = view;
                    u.light_vp = light_vp;
                    u.light_dir = scene->light_dir;
                    u.camera_pos = scene->viewer->position;

                    u.shadow_map = &shadow->depth;
                    u.shadow_w = shadow->depth.get_width();
                    u.shadow_h = shadow->depth.get_height();

                    u.model = glm::mat4(1.0f);
                    u.prev_vp_model = vp * u.model;
                    u.color = scene->ground->color;

                    for (size_t i = 0; i < scene->ground->verts.size(); i += 3) {
                        draw_triangle_main(*rt,
                            scene->ground->verts[i],   scene->ground->norms[i],
                            scene->ground->verts[i+1], scene->ground->norms[i+1],
                            scene->ground->verts[i+2], scene->ground->norms[i+2],
                            u, tmin, tmax
                        );
                    }

                    for (auto* m : scene->monkeys) {
                        u.model = m->get_world_matrix();
                        u.prev_vp_model = m->has_prev_vp_model ? m->prev_vp_model : (vp * u.model);
                        u.color = m->color;

                        for (size_t i = 0; i < m->geometry->triangles.size(); i += 3) {
                            draw_triangle_main(*rt,
                                m->geometry->triangles[i],   m->geometry->normals[i],
                                m->geometry->triangles[i+1], m->geometry->normals[i+1],
                                m->geometry->triangles[i+2], m->geometry->normals[i+2],
                                u, tmin, tmax
                            );
                        }
                    }

                    wg.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }
        wg.wait();
    }

    HelloScene* scene;
    shs::Job::ThreadedPriorityJobSystem* jobs;
    shs::Job::WaitGroup wg;
    RT_Main* rt;
    RT_Shadow* shadow;
};

// ===============================
// POST PROCESS
// ===============================
static void pass_blur(const shs::Canvas& src, shs::Canvas& dst, bool horiz, shs::Job::ThreadedPriorityJobSystem* job, shs::Job::WaitGroup& wg) {
    int w = src.get_width(), h = src.get_height();
    const float weights[] = {0.227027f, 0.1945946f, 0.1216216f, 0.054054f, 0.016216f};
    int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X, rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {
            wg.add(1);
            job->submit({[=, &src, &dst, &wg]() {
                int x0 = tx * TILE_SIZE_X, y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, w), y1 = std::min(y0 + TILE_SIZE_Y, h);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        float r = 0, g = 0, b = 0;
                        shs::Color cc = src.get_color_at(x, y);
                        r += cc.r * weights[0]; g += cc.g * weights[0]; b += cc.b * weights[0];

                        for (int i = 1; i < 5; i++) {
                            shs::Color c1 = src.get_color_at(clampi(x+(horiz?i:0),0,w-1), clampi(y+(horiz?0:i),0,h-1));
                            shs::Color c2 = src.get_color_at(clampi(x-(horiz?i:0),0,w-1), clampi(y-(horiz?0:i),0,h-1));
                            r += (c1.r + c2.r) * weights[i];
                            g += (c1.g + c2.g) * weights[i];
                            b += (c1.b + c2.b) * weights[i];
                        }
                        dst.draw_pixel_screen_space(x, y, color_from_rgbaf(r, g, b, 255));
                    }
                }
                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }
    wg.wait();
}

static void pass_dof_composite(const shs::Canvas& sharp, const shs::Canvas& blur, const shs::ZBuffer& depth, shs::Canvas& out, float focus_depth, shs::Job::ThreadedPriorityJobSystem* job, shs::Job::WaitGroup& wg) {
    int w = sharp.get_width(), h = sharp.get_height();
    int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X, rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {
            wg.add(1);
            job->submit({[=, &sharp, &blur, &depth, &out, &wg]() {
                int x0 = tx * TILE_SIZE_X, y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, w), y1 = std::min(y0 + TILE_SIZE_Y, h);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        float d = depth.get_depth_at(x, (h-1)-y);
                        if (d > 900.0f) d = focus_depth + DOF_RANGE;

                        float t = smoothstep01(std::abs(d - focus_depth) / DOF_RANGE) * DOF_MAXBLUR;
                        out.draw_pixel_screen_space(x, y, lerp_color(sharp.get_color_at(x, y), blur.get_color_at(x, y), t));
                    }
                }
                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }
    wg.wait();
}

static void pass_outline(const shs::Canvas& src, const shs::ZBuffer& depth, shs::Canvas& dst, shs::Job::ThreadedPriorityJobSystem* job, shs::Job::WaitGroup& wg) {
    int w = src.get_width(), h = src.get_height();
    int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X, rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {
            wg.add(1);
            job->submit({[=, &src, &depth, &dst, &wg]() {
                int x0 = tx * TILE_SIZE_X, y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, w), y1 = std::min(y0 + TILE_SIZE_Y, h);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        shs::Color c = src.get_color_at(x, y);
                        float dC = depth.get_depth_at(x, (h-1)-y);
                        if (dC > 900.0f) { dst.draw_pixel_screen_space(x, y, c); continue; }

                        float max_d = 0.0f;
                        for (int oy = -EDGE_RADIUS; oy <= EDGE_RADIUS; oy++) {
                            for (int ox = -EDGE_RADIUS; ox <= EDGE_RADIUS; ox++) {
                                float dN = depth.get_depth_at(clampi(x + ox, 0, w - 1), clampi((h-1) - (y + oy), 0, h - 1));
                                if (dN < 900.0f) max_d = std::max(max_d, std::abs(dC - dN));
                            }
                        }

                        float k = 1.0f - ((max_d > EDGE_THRESHOLD) ? 1.0f : 0.0f) * EDGE_STRENGTH;
                        dst.draw_pixel_screen_space(x, y, color_from_rgbaf(c.r * k, c.g * k, c.b * k, 255));
                    }
                }
                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }
    wg.wait();
}

static void pass_motion_blur(const shs::Canvas& src, const MotionBuffer& mbuf, shs::Canvas& dst, shs::Job::ThreadedPriorityJobSystem* job, shs::Job::WaitGroup& wg) {
    int w = src.get_width(), h = src.get_height();
    int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X, rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {
            wg.add(1);
            job->submit({[=, &src, &mbuf, &dst, &wg]() {
                int x0 = tx * TILE_SIZE_X, y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, w), y1 = std::min(y0 + TILE_SIZE_Y, h);

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        glm::vec2 v_total = mbuf.get(x, y);
                        float len = glm::length(v_total);

                        if (MB_SOFT_KNEE && len > MB_KNEE_PIXELS) {
                            float t = (len - MB_KNEE_PIXELS) / std::max(1e-4f, MB_MAX_PIXELS - MB_KNEE_PIXELS);
                            v_total *= (MB_KNEE_PIXELS + (MB_MAX_PIXELS - MB_KNEE_PIXELS) * (t/(1.0f+t))) / len;
                        }

                        if (glm::length(v_total) < 1.0f) { dst.draw_pixel_screen_space(x, y, src.get_color_at(x, y)); continue; }

                        v_total *= MB_STRENGTH;
                        float r=0,g=0,b=0, wsum=0;

                        for (int i = 0; i < MB_SAMPLES; i++) {
                            float t = (float)i / float(MB_SAMPLES - 1) - 0.5f;
                            glm::vec2 off = v_total * t;

                            shs::Color c = src.get_color_at(clampi(int(x+off.x), 0, w-1), clampi(int(y+off.y), 0, h-1));
                            float wt = 1.0f - std::abs(t*2.0f)*0.3f;

                            r += c.r * wt; g += c.g * wt; b += c.b * wt; wsum += wt;
                        }

                        dst.draw_pixel_screen_space(x, y, color_from_rgbaf(r/wsum, g/wsum, b/wsum, 255));
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }
    wg.wait();
}

static float autofocus_depth(const shs::ZBuffer& zbuf, int cx_screen, int cy_screen, int rad) {
    std::vector<float> samps;
    int w = zbuf.get_width(), h = zbuf.get_height();
    int cy_canvas = (h - 1) - cy_screen;

    for (int oy = -rad; oy <= rad; oy++) {
        for (int ox = -rad; ox <= rad; ox++) {
            float d = zbuf.get_depth_at(clampi(cx_screen+ox,0,w-1), clampi(cy_canvas+oy,0,h-1));
            if (d < 900.0f) samps.push_back(d);
        }
    }
    if (samps.empty()) return 15.0f;
    std::nth_element(samps.begin(), samps.begin() + samps.size() / 2, samps.end());
    return samps[samps.size() / 2];
}

// ===============================
// MAIN
// ===============================
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    SDL_Window* win = nullptr; SDL_Renderer* ren = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &win, &ren);

    auto* job_sys = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);
    shs::Canvas* screen = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface* surf = screen->create_sdl_surface();
    SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);

    HelloScene* scene = new HelloScene();
    RT_Main rt_main(CANVAS_WIDTH, CANVAS_HEIGHT);
    RT_Shadow rt_shadow(SHADOW_MAP_SIZE);

    shs::Canvas pp1(CANVAS_WIDTH, CANVAS_HEIGHT), pp2(CANVAS_WIDTH, CANVAS_HEIGHT);
    RendererSystem renderer(scene, job_sys, &rt_main, &rt_shadow);

    bool exit = false;
    SDL_Event evt;
    Uint32 last = SDL_GetTicks();
    shs::Job::WaitGroup wg_fx;
    bool mouse_down = false;

    while (!exit) {
        Uint32 now = SDL_GetTicks();
        float dt = (now - last) / 1000.0f;
        last = now;

        while (SDL_PollEvent(&evt)) {
            if (evt.type == SDL_QUIT || (evt.type == SDL_KEYDOWN && evt.key.keysym.sym == SDLK_ESCAPE)) exit = true;
            if (evt.type == SDL_MOUSEBUTTONDOWN) mouse_down = true;
            if (evt.type == SDL_MOUSEBUTTONUP) mouse_down = false;
            if (evt.type == SDL_MOUSEMOTION && mouse_down) {
                scene->viewer->horizontal_angle += evt.motion.xrel * MOUSE_SENSITIVITY;
                scene->viewer->vertical_angle   -= evt.motion.yrel * MOUSE_SENSITIVITY;
            }
        }

        const Uint8* ks = SDL_GetKeyboardState(nullptr);
        float spd = 30.0f * dt;
        if (ks[SDL_SCANCODE_W]) scene->viewer->position += scene->viewer->direction_vector * spd;
        if (ks[SDL_SCANCODE_S]) scene->viewer->position -= scene->viewer->direction_vector * spd;
        if (ks[SDL_SCANCODE_A]) scene->viewer->position -= scene->viewer->right_vector * spd;
        if (ks[SDL_SCANCODE_D]) scene->viewer->position += scene->viewer->right_vector * spd;

        scene->process();
        for (auto* m : scene->monkeys) m->update(dt);
        scene->viewer->update();

        renderer.process(dt);

        if (ENABLE_DOF) {
            pass_blur(rt_main.color, pp1, true, job_sys, wg_fx);
            pass_blur(pp1, pp2, false, job_sys, wg_fx);
            pass_blur(pp2, pp1, true, job_sys, wg_fx);
            pass_blur(pp1, pp2, false, job_sys, wg_fx);

            float focus = autofocus_depth(rt_main.depth, CANVAS_WIDTH / 2, CANVAS_HEIGHT / 2, AUTOFOCUS_RADIUS);
            pass_dof_composite(rt_main.color, pp2, rt_main.depth, pp1, focus, job_sys, wg_fx);
        } else {
            pp1.buffer() = rt_main.color.buffer();
        }

        pass_outline(pp1, rt_main.depth, pp2, job_sys, wg_fx);
        pass_motion_blur(pp2, rt_main.motion, pp1, job_sys, wg_fx);

        screen->buffer() = pp1.buffer();
        shs::Canvas::copy_to_SDLSurface(surf, screen);
        SDL_UpdateTexture(tex, nullptr, surf->pixels, surf->pitch);
        SDL_RenderCopy(ren, tex, nullptr, nullptr);
        SDL_RenderPresent(ren);

        SDL_SetWindowTitle(win, ("FPS: " + std::to_string((dt > 1e-6f) ? (1.0f / dt) : 0.0f)).c_str());
    }

    delete scene;
    delete screen;
    delete job_sys;

    SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
