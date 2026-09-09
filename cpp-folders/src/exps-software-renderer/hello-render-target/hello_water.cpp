/*
    WATER DEMO

    Ус: 
        Fresnel + planar reflection + specular + fog

    Reference:
        https://www.shadertoy.com/view/Xl2XRW

    Координат:
    - 3D          : LH, +Z forward, +Y up, +X right
    - Screen      : y down
    - shs::Canvas : y up (bottom-left)

*/

#define SDL_MAIN_HANDLED

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

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

// ------------------------------------------
// SHADOW MAP CONFIG
// ------------------------------------------
#define SHADOW_MAP_SIZE   2048

static const glm::vec3 LIGHT_DIR_WORLD = glm::normalize(glm::vec3(-1.0f, -1.2f, 0.9f));

// Shadow bias (acne vs peter-panning тохируулга)
static const float SHADOW_BIAS_BASE   = 0.0025f;
static const float SHADOW_BIAS_SLOPE  = 0.0100f;

// PCF (2x2) шүүлт
static const bool  SHADOW_USE_PCF     = true;

// ------------------------------------------
// MOTION BLUR CONFIG
// ------------------------------------------
static const int   MB_SAMPLES      = 12;
static const float MB_STRENGTH     = 0.85f;
static const float MB_MAX_PIXELS   = 22.0f;

static const float MB_W_OBJ        = 1.00f;
static const float MB_W_CAM        = 0.35f;

static const bool  MB_SOFT_KNEE    = true;
static const float MB_KNEE_PIXELS  = 18.0f;

// ------------------------------------------
// UV FLIP (хэрвээ texture урвуу байвал 1)
// ------------------------------------------
#define UV_FLIP_V 0

// ------------------------------------------
// WATER CONFIG
// ------------------------------------------
static const float FLOOR_Y = -3.0f;     // шал усны доор
static const float WATER_Y = -0.20f;    // усны түвшин (машиныг живүүлэхгүй)
static const shs::Color CLEAR_BG = shs::Color{ 24, 34, 58, 255 }; // background

// ==========================================
// HELPERS
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

static inline float clamp01(float v) { return (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v); }

// Using standardized shs::ShadowMap, shs::MotionBuffer, shs::RT_ColorDepthVelocity
//using ShadowMap    = shs::ShadowMap;
//using MotionBuffer = shs::MotionBuffer;
//using RT_ColorDepthMotion = shs::RT_ColorDepthVelocity;

// Using standardized shs::Viewer and shs::ModelGeometry
//using Viewer        = shs::Viewer;
//using ModelGeometry = shs::ModelGeometry;

// ==========================================
// TEXTURE SAMPLER (nearest)
// ==========================================

static inline shs::Color sample_nearest(const shs::Texture2D &tex, glm::vec2 uv)
{
    float u = uv.x;
    float v = uv.y;

#if UV_FLIP_V
    v = 1.0f - v;
#endif

    u = clamp01(u);
    v = clamp01(v);

    int x = (int)std::lround(u * (float)(tex.w - 1));
    int y = (int)std::lround(v * (float)(tex.h - 1));

    x = clampi(x, 0, tex.w - 1);
    y = clampi(y, 0, tex.h - 1);

    return tex.texels.at(x, y);
}

// ==========================================
// CANVAS SAMPLER (nearest) - Canvas coords (y up)
// ==========================================

static inline shs::Color sample_canvas_nearest(const shs::Canvas &c, int x, int y)
{
    x = clampi(x, 0, c.get_width() - 1);
    y = clampi(y, 0, c.get_height() - 1);
    return c.get_color_at(x, y);
}

// ==========================================
// ATMOSPHERE (Sky + Fog + Tonemap + Gamma)
// ==========================================

static inline glm::vec3 sky_color_simple(const glm::vec3& ray_dir, const glm::vec3& sun_dir)
{
    float t = clamp01(ray_dir.y * 0.5f + 0.5f);

    glm::vec3 sky_horizon = glm::vec3(0.62f, 0.74f, 0.92f);
    glm::vec3 sky_zenith  = glm::vec3(0.10f, 0.22f, 0.55f);
    glm::vec3 sky = glm::mix(sky_horizon, sky_zenith, t);

    float sd = glm::dot(glm::normalize(sun_dir), glm::normalize(ray_dir));
    float sun = std::pow(clampf(sd * 0.5f + 0.5f, 0.0f, 1.0f), 256.0f);
    sky += glm::vec3(1.0f, 0.88f, 0.65f) * sun * 8.0f;

    return sky;
}

static inline glm::vec3 apply_fog_exp2(const glm::vec3& col, const glm::vec3& fog_col, float dist, float density)
{
    float tr = std::exp2(-density * dist);
    return col * tr + fog_col * (1.0f - tr);
}

static inline glm::vec3 tonemap_reinhard(const glm::vec3& x)
{
    return x / (glm::vec3(1.0f) + x);
}

static inline glm::vec3 gamma_2p2(const glm::vec3& x)
{
    glm::vec3 c = glm::clamp(x, 0.0f, 1.0f);
    return glm::pow(c, glm::vec3(1.0f / 2.2f));
}

// ==========================================
// SHADOW MAP BUFFER (Depth only)
// ==========================================
/*
struct ShadowMap
{
    int w = 0;
    int h = 0;
    shs::Buffer<float> depth;  // light NDC z (0..1)

    ShadowMap() {}
    ShadowMap(int W, int H) { init(W, H); }

    void init(int W, int H)
    {
        w = W; h = H;
        depth = shs::Buffer<float>(w, h, std::numeric_limits<float>::max());
    }

    inline void clear()
    {
        depth.clear(std::numeric_limits<float>::max());
    }

    inline bool test_and_set(int x, int y, float z_ndc)
    {
        if (!depth.in_bounds(x,y)) return false;
        float& d = depth.at(x,y);
        if (z_ndc < d) { d = z_ndc; return true; }
        return false;
    }

    inline float sample(int x, int y) const
    {
        x = clampi(x, 0, w - 1);
        y = clampi(y, 0, h - 1);
        return depth.at(x,y);
    }
};
*/
// ==========================================
// MOTION BUFFER (Canvas coords, pixels, +Y up)
// ==========================================
/*
struct MotionBuffer
{
    MotionBuffer() : w(0), h(0) {}
    MotionBuffer(int W, int H) { init(W,H); }

    void init(int W, int H)
    {
        w = W; h = H;
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
*/

// ==========================================
// RT: Color + Depth(view_z) + Motion(full)
// ==========================================
/*
struct RT_ColorDepthMotion
{
    RT_ColorDepthMotion(int W, int H, float zn, float zf, shs::Color clear_col)
        : color(W, H, clear_col), depth(W, H, zn, zf), motion(W, H)
    {
        clear(clear_col);
    }

    inline void clear(shs::Color c)
    {
        color.buffer().clear(c);
        depth.clear();
        motion.clear();
    }

    shs::Canvas       color;
    shs::ZBuffer      depth;   // view_z
    shs::MotionBuffer motion;  // v_full
};
*/

// ==========================================
// CAMERA + VIEWER
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
// GEOMETRY (Assimp)
// ==========================================

class ModelGeometry
{
public:
    ModelGeometry(const std::string& model_path)
    {
        unsigned int flags =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices;

        const aiScene *scene = importer.ReadFile(model_path.c_str(), flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Model load error: " << importer.GetErrorString() << std::endl;
            return;
        }

        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];
            bool has_uv = mesh->HasTextureCoords(0);

            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                if (mesh->mFaces[j].mNumIndices != 3) continue;

                for (int k = 0; k < 3; k++) {
                    unsigned int idx = mesh->mFaces[j].mIndices[k];

                    aiVector3D v = mesh->mVertices[idx];
                    triangles.push_back(glm::vec3(v.x, v.y, v.z));

                    if (mesh->HasNormals()) {
                        aiVector3D n = mesh->mNormals[idx];
                        normals.push_back(glm::vec3(n.x, n.y, n.z));
                    } else {
                        normals.push_back(glm::vec3(0, 1, 0));
                    }

                    if (has_uv) {
                        aiVector3D t = mesh->mTextureCoords[0][idx];
                        uvs.push_back(glm::vec2(t.x, t.y));
                    } else {
                        uvs.push_back(glm::vec2(0.0f));
                    }
                }
            }
        }
    }

    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

private:
    Assimp::Importer importer;
};

// ==========================================
// SCENE OBJECTS
// ==========================================

class SubaruObject : public shs::AbstractObject3D
{
public:
    SubaruObject(glm::vec3 position, glm::vec3 scale, const shs::Texture2D *albedo)
    {
        this->position       = position;
        this->scale          = scale;
        this->geometry       = new ModelGeometry("./assets/obj/subaru/SUBARU_1.rawobj");
        this->rotation_angle = 0.0f;
        this->albedo         = albedo;
        this->has_prev_mvp   = false;
        this->prev_mvp       = glm::mat4(1.0f);
    }
    ~SubaruObject() { delete geometry; }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float dt) override
    {
        rotation_angle -= 12.0f * dt;
        if (rotation_angle < -360.0f) rotation_angle += 360.0f;
    }

    void render() override {}

    ModelGeometry        *geometry;
    const shs::Texture2D *albedo;

    glm::vec3 position;
    glm::vec3 scale;
    float     rotation_angle;

    bool      has_prev_mvp;
    glm::mat4 prev_mvp;
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 base_pos, glm::vec3 scale)
    {
        this->geometry = new ModelGeometry("./assets/obj/monkey/monkey.rawobj");

        this->base_position = base_pos;
        this->position      = base_pos;
        this->scale         = scale;

        this->time_accum     = 0.0f;
        this->rotation_angle = 0.0f;

        this->spin_deg_per_sec   = 320.0f;
        this->wobble_hz          = 2.6f;
        this->wobble_amp_y       = 0.55f;
        this->wobble_amp_xz      = 0.35f;
        //this->wobble_phase_speed = 6.2831853f;

        this->has_prev_mvp = false;
        this->prev_mvp     = glm::mat4(1.0f);
    }

    ~MonkeyObject() { delete geometry; }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float dt) override
    {
        time_accum += dt;

        float w = 6.2831853f * wobble_hz;

        position    = base_position;
        position.y += std::sin(time_accum * w) * wobble_amp_y;

        position.x += std::cos(time_accum * w * 1.15f) * wobble_amp_xz;
        position.z += std::sin(time_accum * w * 0.95f) * wobble_amp_xz;

        rotation_angle += spin_deg_per_sec * dt;
        if (rotation_angle > 360.0f) rotation_angle -= 360.0f;
    }

    void render() override {}

    ModelGeometry *geometry;

    glm::vec3 base_position;
    glm::vec3 position;
    glm::vec3 scale;

    float time_accum;
    float rotation_angle;

    float spin_deg_per_sec;
    float wobble_hz;
    float wobble_amp_y;
    float wobble_amp_xz;

    bool      has_prev_mvp;
    glm::mat4 prev_mvp;
};

// ==========================================
// FLOOR (grid)
// ==========================================

struct FloorPlane
{
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> norms;
    std::vector<glm::vec2> uvs;

    FloorPlane(float half_size, float z_forward, float floor_y)
    {
        verts.clear(); norms.clear(); uvs.clear();

        const int GRID_X = 48;
        const int GRID_Z = 48;

        float y  = floor_y;
        float S  = half_size;
        float Z0 = 0.0f;
        float Z1 = z_forward;

        glm::vec3 n(0.0f, 1.0f, 0.0f);

        for (int iz = 0; iz < GRID_Z; ++iz) {
            float tz0 = float(iz) / float(GRID_Z);
            float tz1 = float(iz + 1) / float(GRID_Z);

            float z0 = Z0 + (Z1 - Z0) * tz0;
            float z1 = Z0 + (Z1 - Z0) * tz1;

            for (int ix = 0; ix < GRID_X; ++ix) {
                float tx0 = float(ix) / float(GRID_X);
                float tx1 = float(ix + 1) / float(GRID_X);

                float x0 = -S + (2.0f * S) * tx0;
                float x1 = -S + (2.0f * S) * tx1;

                glm::vec3 p00(x0, y, z0);
                glm::vec3 p10(x1, y, z0);
                glm::vec3 p11(x1, y, z1);
                glm::vec3 p01(x0, y, z1);

                verts.push_back(p00); verts.push_back(p10); verts.push_back(p11);
                verts.push_back(p00); verts.push_back(p11); verts.push_back(p01);

                norms.push_back(n); norms.push_back(n); norms.push_back(n);
                norms.push_back(n); norms.push_back(n); norms.push_back(n);

                glm::vec2 uv00(tx0, tz0);
                glm::vec2 uv10(tx1, tz0);
                glm::vec2 uv11(tx1, tz1);
                glm::vec2 uv01(tx0, tz1);

                uvs.push_back(uv00); uvs.push_back(uv10); uvs.push_back(uv11);
                uvs.push_back(uv00); uvs.push_back(uv11); uvs.push_back(uv01);
            }
        }
    }
};

// ==========================================
// WATER PLANE (grid)
// ==========================================

struct WaterPlane
{
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> norms;
    std::vector<glm::vec2> uvs;

    WaterPlane(float half_size, float z_forward, float water_y)
    {
        verts.clear(); norms.clear(); uvs.clear();

        const int GRID_X = 56;
        const int GRID_Z = 56;

        float y  = water_y;
        float S  = half_size;
        float Z0 = 0.0f;
        float Z1 = z_forward;

        glm::vec3 n(0.0f, 1.0f, 0.0f);

        for (int iz = 0; iz < GRID_Z; ++iz) {
            float tz0 = float(iz) / float(GRID_Z);
            float tz1 = float(iz + 1) / float(GRID_Z);

            float z0 = Z0 + (Z1 - Z0) * tz0;
            float z1 = Z0 + (Z1 - Z0) * tz1;

            for (int ix = 0; ix < GRID_X; ++ix) {
                float tx0 = float(ix) / float(GRID_X);
                float tx1 = float(ix + 1) / float(GRID_X);

                float x0 = -S + (2.0f * S) * tx0;
                float x1 = -S + (2.0f * S) * tx1;

                glm::vec3 p00(x0, y, z0);
                glm::vec3 p10(x1, y, z0);
                glm::vec3 p11(x1, y, z1);
                glm::vec3 p01(x0, y, z1);

                verts.push_back(p00); verts.push_back(p10); verts.push_back(p11);
                verts.push_back(p00); verts.push_back(p11); verts.push_back(p01);

                norms.push_back(n); norms.push_back(n); norms.push_back(n);
                norms.push_back(n); norms.push_back(n); norms.push_back(n);

                glm::vec2 uv00(tx0, tz0);
                glm::vec2 uv10(tx1, tz0);
                glm::vec2 uv11(tx1, tz1);
                glm::vec2 uv01(tx0, tz1);

                uvs.push_back(uv00); uvs.push_back(uv10); uvs.push_back(uv11);
                uvs.push_back(uv00); uvs.push_back(uv11); uvs.push_back(uv01);
            }
        }
    }
};

// ==========================================
// UNIFORMS & VARYINGS
// ==========================================

struct Uniforms
{
    glm::mat4 mvp;
    glm::mat4 prev_mvp;
    glm::mat4 model;
    glm::mat4 view;

    glm::mat4 light_vp;

    glm::vec3 light_dir_world;
    glm::vec3 camera_pos;

    shs::Color base_color;
    const shs::Texture2D *albedo = nullptr;
    bool use_texture = false;

    const shs::ShadowMap *shadow = nullptr;

    const shs::Canvas* reflection_color = nullptr;
    glm::mat4          reflection_vp    = glm::mat4(1.0f);
    float              time_sec         = 0.0f;
};

struct VaryingsFull
{
    glm::vec4 position;
    glm::vec4 prev_position;
    glm::vec3 world_pos;
    glm::vec3 normal;
    glm::vec2 uv;
    float     view_z;
};

// ==========================================
// VERTEX SHADER (camera pass)
// ==========================================

static VaryingsFull vertex_shader_full(
    const glm::vec3& aPos,
    const glm::vec3& aNormal,
    const glm::vec2& aUV,
    const Uniforms&  u)
{
    VaryingsFull out;
    out.position      = u.mvp * glm::vec4(aPos, 1.0f);
    out.prev_position = u.prev_mvp * glm::vec4(aPos, 1.0f);

    out.world_pos     = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal        = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv            = aUV;

    glm::vec4 view_pos = u.view * u.model * glm::vec4(aPos, 1.0f);
    out.view_z = view_pos.z;

    return out;
}

// ==========================================
// SHADOW HELPERS (uv top-left, y down)
// ==========================================

static inline bool shadow_uvz_from_world(
    const glm::mat4& light_vp,
    const glm::vec3& world_pos,
    glm::vec2& out_uv,
    float& out_z_ndc)
{
    glm::vec4 clip = light_vp * glm::vec4(world_pos, 1.0f);
    if (std::abs(clip.w) < 1e-6f) return false;

    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    out_z_ndc = ndc.z;

    if (out_z_ndc < 0.0f || out_z_ndc > 1.0f) return false;

    out_uv.x = ndc.x * 0.5f + 0.5f;
    out_uv.y = 1.0f - (ndc.y * 0.5f + 0.5f);

    return true;
}

static inline float shadow_factor_pcf_2x2(
    const shs::ShadowMap& sm,
    glm::vec2 uv,
    float z_ndc,
    float bias)
{
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;

    if (!SHADOW_USE_PCF) {
        int x = (int)std::lround(uv.x * float(sm.w - 1));
        int y = (int)std::lround(uv.y * float(sm.h - 1));
        float d = sm.sample(x, y);
        if (d == std::numeric_limits<float>::max()) return 1.0f;
        return (z_ndc <= d + bias) ? 1.0f : 0.0f;
    }

    float fx = uv.x * float(sm.w - 1);
    float fy = uv.y * float(sm.h - 1);

    int x0 = clampi((int)std::floor(fx), 0, sm.w - 1);
    int y0 = clampi((int)std::floor(fy), 0, sm.h - 1);
    int x1 = clampi(x0 + 1, 0, sm.w - 1);
    int y1 = clampi(y0 + 1, 0, sm.h - 1);

    float d00 = sm.sample(x0,y0);
    float d10 = sm.sample(x1,y0);
    float d01 = sm.sample(x0,y1);
    float d11 = sm.sample(x1,y1);

    float s00 = (d00 == std::numeric_limits<float>::max()) ? 1.0f : ((z_ndc <= d00 + bias) ? 1.0f : 0.0f);
    float s10 = (d10 == std::numeric_limits<float>::max()) ? 1.0f : ((z_ndc <= d10 + bias) ? 1.0f : 0.0f);
    float s01 = (d01 == std::numeric_limits<float>::max()) ? 1.0f : ((z_ndc <= d01 + bias) ? 1.0f : 0.0f);
    float s11 = (d11 == std::numeric_limits<float>::max()) ? 1.0f : ((z_ndc <= d11 + bias) ? 1.0f : 0.0f);

    return 0.25f * (s00 + s10 + s01 + s11);
}


// ==========================================
// WATER FLOW NOISE (fast) : hash/noise/fbm + advected height -> normal + foam
// ==========================================


static inline float fractf(float x) { return x - std::floor(x); }

static inline glm::vec2 fract2(const glm::vec2& v) { return glm::vec2(fractf(v.x), fractf(v.y)); }

static inline float hash11(float p)
{
    p = fractf(p * 0.1031f);
    p *= p + 33.33f;
    p *= p + p;
    return fractf(p);
}

static inline float hash12(const glm::vec2& p)
{
    // жижиг, хурдан hash (texture-гүй)
    glm::vec2 q = fract2(p * glm::vec2(123.34f, 456.21f));
    q += glm::dot(q, q + 34.345f);
    return fractf(q.x * q.y);
}

static inline glm::vec2 hash22(float p)
{
    // Dave Hoskins-style hash2
    float x = fractf(p * 0.1031f);
    float y = fractf(p * 0.1030f);
    float z = fractf(p * 0.0973f);

    // p3 += dot(p3, p3.yzx + 19.19)
    float d = x * (y + 19.19f) + y * (z + 19.19f) + z * (x + 19.19f);
    x = fractf(x + d);
    y = fractf(y + d);
    z = fractf(z + d);

    // fract((p3.xx + p3.yz) * p3.zy)
    float r0 = fractf((x + y) * z);
    float r1 = fractf((x + z) * y);
    return glm::vec2(r0, r1);
}

static inline float noise2(const glm::vec2& p)
{
    glm::vec2 i(std::floor(p.x), std::floor(p.y));
    glm::vec2 f = fract2(p);

    float a = hash12(i + glm::vec2(0.0f, 0.0f));
    float b = hash12(i + glm::vec2(1.0f, 0.0f));
    float c = hash12(i + glm::vec2(0.0f, 1.0f));
    float d = hash12(i + glm::vec2(1.0f, 1.0f));

    // smoothstep
    glm::vec2 u = f * f * (glm::vec2(3.0f) - 2.0f * f);

    float x1 = a + (b - a) * u.x;
    float x2 = c + (d - c) * u.x;
    return x1 + (x2 - x1) * u.y;
}

static inline float fbm2(glm::vec2 p)
{
    float f = 0.0f;
    float a = 0.5f;
    for (int i = 0; i < 4; i++) {
        f += a * noise2(p);
        p *= 2.02f;
        a *= 0.5f;
    }
    return f;
}

static inline glm::vec2 flow_dir_base()
{
    // гол чиглэл (+Z) + бага зэрэг X
    return glm::normalize(glm::vec2(0.18f, 1.0f));
}

static inline glm::vec3 sample_water_normal_core(const glm::vec2& uv, const glm::vec2& flow_offset, float foam, float& outH)
{
    glm::vec2 p = uv * 7.5f + flow_offset;

    float h0 = fbm2(p * 1.00f);
    float h1 = fbm2(p * 2.35f + glm::vec2(17.0f, 3.0f));
    float h  = (h0 * 0.65f + h1 * 0.35f) * 2.0f - 1.0f;

    float detail_flat = clamp01(1.0f - foam * 2.5f);

    // gradient finite diff
    float e = 0.12f;
    float hx1 = (fbm2((p + glm::vec2(e, 0.0f)) * 1.00f) * 0.65f + fbm2((p + glm::vec2(e, 0.0f)) * 2.35f + glm::vec2(17.0f, 3.0f)) * 0.35f) * 2.0f - 1.0f;
    float hx0 = (fbm2((p - glm::vec2(e, 0.0f)) * 1.00f) * 0.65f + fbm2((p - glm::vec2(e, 0.0f)) * 2.35f + glm::vec2(17.0f, 3.0f)) * 0.35f) * 2.0f - 1.0f;
    float hz1 = (fbm2((p + glm::vec2(0.0f, e)) * 1.00f) * 0.65f + fbm2((p + glm::vec2(0.0f, e)) * 2.35f + glm::vec2(17.0f, 3.0f)) * 0.35f) * 2.0f - 1.0f;
    float hz0 = (fbm2((p - glm::vec2(0.0f, e)) * 1.00f) * 0.65f + fbm2((p - glm::vec2(0.0f, e)) * 2.35f + glm::vec2(17.0f, 3.0f)) * 0.35f) * 2.0f - 1.0f;

    float dhdx = (hx1 - hx0) / (2.0f * e);
    float dhdz = (hz1 - hz0) / (2.0f * e);

    float normal_amp = 2.7f * detail_flat;
    glm::vec3 N = glm::normalize(glm::vec3(-dhdx * normal_amp, 1.0f, -dhdz * normal_amp));

    outH = h;
    return N;
}


static inline void water_flow_normal_and_foam(
    const glm::vec3& world_pos,
    float t,
    glm::vec3& outN,
    float& outFoam)
{
    // SHADERTOY-STYLE FLOW PARAMETERS
    
    const float FLOW_SPEED   = 0.18f;   // ocean-like
    const float FLOW_STRETCH = 2.8f;    // anisotropy 
    const float NORMAL_AMP   = 2.4f;    // wave strength

    glm::vec2 xz(world_pos.x, world_pos.z);

    // урсгалын чиглэл (гол / далайн урсгал )
    glm::vec2 flow_dir  = glm::normalize(glm::vec2(0.15f, 1.0f));
    glm::vec2 flow_perp = glm::vec2(-flow_dir.y, flow_dir.x);

    // anisotropic coordinates
    glm::vec2 uv;
    uv.x = glm::dot(xz, flow_perp);         
    uv.y = glm::dot(xz, flow_dir) * FLOW_STRETCH;

    // урсах хугацаа  (slow)
    float time = t * FLOW_SPEED;

    // blending
    float t0 = fractf(time);
    float t1 = fractf(time + 0.5f);
    float w  = std::abs(t0 - 0.5f) * 2.0f;

    float i0 = std::floor(time);
    float i1 = std::floor(time + 0.5f);

    glm::vec2 j0 = hash22(i0) * 2.0f;
    glm::vec2 j1 = hash22(i1) * 2.0f;

    glm::vec2 uv0 = uv * 0.055f + j0 + flow_dir * (t0 - 0.5f) * 6.0f;
    glm::vec2 uv1 = uv * 0.055f + j1 + flow_dir * (t1 - 0.5f) * 6.0f;

    auto sampleN = [&](const glm::vec2& p) -> glm::vec3
    {
        float h0 = fbm2(p * 1.2f);
        float h1 = fbm2(p * 2.7f + 11.0f);
        //float h  = (h0 * 0.65f + h1 * 0.35f);

        float e = 0.18f;
        float hx = fbm2((p + glm::vec2(e, 0)) * 1.2f) - fbm2((p - glm::vec2(e, 0)) * 1.2f);
        float hz = fbm2((p + glm::vec2(0, e)) * 1.2f) - fbm2((p - glm::vec2(0, e)) * 1.2f);

        return glm::normalize(glm::vec3(
            -hx * NORMAL_AMP,
             1.0f,
            -hz * NORMAL_AMP
        ));
    };

    glm::vec3 nA = sampleN(uv0);
    glm::vec3 nB = sampleN(uv1);

    glm::vec3 N = glm::normalize(glm::mix(nA, nB, w));

    N = glm::normalize(N + glm::vec3(flow_dir.x, 0.0f, flow_dir.y) * 0.15f);

    outN = N;

    float curvature = glm::length(glm::vec2(N.x, N.z));
    outFoam = clamp01((curvature - 0.55f) * 1.6f);
}



// ==========================================
// FRAGMENT SHADER - atmosphere
// ==========================================

static shs::Color fragment_shader_full(const VaryingsFull& in, const Uniforms& u)
{
    glm::vec3 N   = glm::normalize(in.normal);
    glm::vec3 L   = glm::normalize(-u.light_dir_world);
    glm::vec3 V   = glm::normalize(u.camera_pos - in.world_pos);   // point  -> camera
    glm::vec3 RAY = -V;                                            // camera -> point

    glm::vec3 baseColor;
    if (u.use_texture && u.albedo && u.albedo->valid()) {
        shs::Color tc = shs::sample_nearest(*u.albedo, in.uv);
        baseColor     = glm::vec3(tc.r, tc.g, tc.b) / 255.0f;
    } else {
        baseColor = glm::vec3(u.base_color.r, u.base_color.g, u.base_color.b) / 255.0f;
    }

    float shadow = 1.0f;
    if (u.shadow) {
        glm::vec2 suv;
        float sz;
        if (shadow_uvz_from_world(u.light_vp, in.world_pos, suv, sz)) {
            float slope = 1.0f - glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
            float bias  = SHADOW_BIAS_BASE + SHADOW_BIAS_SLOPE * slope;
            shadow = shadow_factor_pcf_2x2(*u.shadow, suv, sz, bias);
        }
    }

    float NoL = glm::max(glm::dot(N, L), 0.0f);

    // Атмосфер/гэрэл: тэнгэрийн өнгөнөөс бага хэмжээгээр ambient утга авна
    glm::vec3 sky = sky_color_simple(glm::normalize(RAY), glm::normalize(-L));

    glm::vec3 ambient = sky * 0.08f;                 
    glm::vec3 diffuse = glm::vec3(1.0f) * NoL * 0.90f;

    glm::vec3 H = glm::normalize(L + V);
    float spec = std::pow(glm::max(glm::dot(N, H), 0.0f), 64.0f);
    glm::vec3 specular = glm::vec3(1.0f) * spec * 0.35f;

    glm::vec3 hdr = baseColor * (ambient + shadow * (diffuse + specular));

    // Манан: өтгөн байдлыг бууруулна (усан доорхи мэдрэмж алга болно)
    float dist = glm::length(u.camera_pos - in.world_pos);
    dist = clampf(dist, 0.0f, 250.0f);
    hdr = apply_fog_exp2(hdr, sky, dist, 0.0065f);   

    // Exposure
    hdr *= 1.00f;

    glm::vec3 ldr = gamma_2p2(tonemap_reinhard(hdr));

    return shs::Color{
        (uint8_t)clampi((int)std::lround(ldr.r * 255.0f), 0, 255),
        (uint8_t)clampi((int)std::lround(ldr.g * 255.0f), 0, 255),
        (uint8_t)clampi((int)std::lround(ldr.b * 255.0f), 0, 255),
        255
    };
}

// ==========================================
// FRAGMENT SHADER (WATER) - fresnel + reflection + fog
// ==========================================

static shs::Color fragment_shader_water(const VaryingsFull& in, const Uniforms& u)
{
    glm::vec3 N;
    float foam = 0.0f;
    water_flow_normal_and_foam(in.world_pos, u.time_sec, N, foam);

    glm::vec3 L = glm::normalize(-u.light_dir_world);
    glm::vec3 V = glm::normalize(u.camera_pos - in.world_pos); // point -> camera
    glm::vec3 RAY = -V;

    float NoV = glm::clamp(glm::dot(N, V), 0.0f, 1.0f);
    float NoL = glm::clamp(glm::dot(N, L), 0.0f, 1.0f);

    glm::vec3 sky = sky_color_simple(glm::normalize(RAY), glm::normalize(-L));

    // Fresnel (Schlick)
    float F0 = 0.02f;
    float fresnel = F0 + (1.0f - F0) * std::pow(1.0f - NoV, 5.0f);

    // Planar reflection
    glm::vec3 refl_col = sky;
    if (u.reflection_color)
    {
        glm::vec4 rclip = u.reflection_vp * glm::vec4(in.world_pos, 1.0f);
        if (std::abs(rclip.w) > 1e-6f)
        {
            glm::vec3 rndc = glm::vec3(rclip) / rclip.w;
            if (rndc.x >= -1.0f && rndc.x <= 1.0f && rndc.y >= -1.0f && rndc.y <= 1.0f)
            {
                float sx = (rndc.x * 0.5f + 0.5f) * float(u.reflection_color->get_width()  - 1);
                float sy = (1.0f - (rndc.y * 0.5f + 0.5f)) * float(u.reflection_color->get_height() - 1);

                float disto = 0.75f;
                sx += N.x * disto;
                sy += N.z * disto;

                int rx = (int)std::lround(sx);
                int ry_screen = (int)std::lround(sy);
                int ry_canvas = (u.reflection_color->get_height() - 1) - ry_screen;

                shs::Color rc = sample_canvas_nearest(*u.reflection_color, rx, ry_canvas);
                refl_col      = glm::vec3(rc.r, rc.g, rc.b) / 255.0f;
            }
        }
    }

    // Shadow
    float shadow = 1.0f;
    if (u.shadow) {
        glm::vec2 suv;
        float sz;
        if (shadow_uvz_from_world(u.light_vp, in.world_pos, suv, sz)) {
            float slope = 1.0f - glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
            float bias  = SHADOW_BIAS_BASE + SHADOW_BIAS_SLOPE * slope;
            shadow      = shadow_factor_pcf_2x2(*u.shadow, suv, sz, bias);
        }
    }

    // Base water
    glm::vec3 water_base = glm::vec3(0.03f, 0.12f, 0.16f);

    // Specular
    glm::vec3 H        = glm::normalize(L + V);
    float gloss        = 520.0f;
    float spec         = std::pow(glm::max(glm::dot(N, H), 0.0f), gloss);
    float spec_kill    = (1.0f - foam * 0.90f);
    glm::vec3 specular = glm::vec3(1.0f) * spec * 0.95f * spec_kill;

    // Lighting
    glm::vec3 ambient = sky * 0.12f;
    glm::vec3 diffuse = glm::vec3(1.0f) * NoL * 0.05f;

    // Fresnel reflect
    glm::vec3 surface = glm::mix(water_base, refl_col, fresnel);

    // Foam color (цагаан судал/хөөс)
    glm::vec3 foam_col = glm::mix(glm::vec3(0.75f, 0.85f, 0.95f), sky, 0.25f);
    float foam_gain = foam * (0.35f + 0.65f * (1.0f - NoV)); // glancing дээр илүү
    surface = glm::mix(surface, foam_col, foam_gain);
    

    glm::vec3 hdr = surface * (ambient + shadow * diffuse) + shadow * specular;

    // Манан
    float dist_to_cam = glm::length(u.camera_pos - in.world_pos);
    dist_to_cam = clampf(dist_to_cam, 0.0f, 250.0f);
    hdr = apply_fog_exp2(hdr, sky, dist_to_cam, 0.0065f);

    hdr *= 0.90f;
    glm::vec3 ldr = gamma_2p2(tonemap_reinhard(hdr));

    return shs::Color{
        (uint8_t)clampi((int)std::lround(ldr.r * 255.0f), 0, 255),
        (uint8_t)clampi((int)std::lround(ldr.g * 255.0f), 0, 255),
        (uint8_t)clampi((int)std::lround(ldr.b * 255.0f), 0, 255),
        255
    };
}


// ==========================================
// SHADOW PASS VARYINGS
// ==========================================

struct VaryingsShadow
{
    glm::vec4 position;
};

static inline VaryingsShadow shadow_vertex_shader(const glm::vec3& aPos, const Uniforms& u)
{
    VaryingsShadow out;
    out.position = u.light_vp * u.model * glm::vec4(aPos, 1.0f);
    return out;
}

// ==========================================
// SHADOW MAP RASTER (tiled)
// ==========================================

static inline glm::vec3 clip_to_shadow_screen(const glm::vec4& clip, int W, int H)
{
    glm::vec3 ndc = glm::vec3(clip) / clip.w;
    glm::vec3 s;
    s.x = (ndc.x * 0.5f + 0.5f) * float(W - 1);
    s.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * float(H - 1);
    s.z = ndc.z;
    return s;
}

static void draw_triangle_tile_shadow(
    shs::ShadowMap& sm,
    const std::vector<glm::vec3>& tri_verts,
    std::function<VaryingsShadow(const glm::vec3&)> vs,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    VaryingsShadow vout[3];
    glm::vec3 sc[3];

    for (int i = 0; i < 3; i++) {
        vout[i] = vs(tri_verts[i]);
        if (std::abs(vout[i].position.w) < 1e-6f) return;
        sc[i] = clip_to_shadow_screen(vout[i].position, sm.w, sm.h);
    }

    glm::vec2 bboxmin(tile_max.x, tile_max.y);
    glm::vec2 bboxmax(tile_min.x, tile_min.y);
    std::vector<glm::vec2> v2d = { glm::vec2(sc[0]), glm::vec2(sc[1]), glm::vec2(sc[2]) };

    for (int i = 0; i < 3; i++) {
        bboxmin = glm::max(glm::vec2(tile_min), glm::min(bboxmin, v2d[i]));
        bboxmax = glm::min(glm::vec2(tile_max), glm::max(bboxmax, v2d[i]));
    }

    if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

    float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) -
                 (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);

    if (std::abs(area) < 1e-8f) return;

    for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
        for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

            float z = bc.x * sc[0].z + bc.y * sc[1].z + bc.z * sc[2].z;
            if (z < 0.0f || z > 1.0f) continue;

            sm.test_and_set(px, py, z);
        }
    }
}

// ==========================================
// CAMERA PASS RASTER (color+depth+motion+shadow)
// ==========================================

static inline glm::vec2 clip_to_screen_xy(const glm::vec4& clip, int w, int h)
{
    glm::vec3 s = shs::Canvas::clip_to_screen(clip, w, h);
    return glm::vec2(s.x, s.y);
}

static void draw_triangle_tile_color_depth_motion_shadow(
    shs::RT_ColorDepthMotion& rt,
    const std::vector<glm::vec3>& tri_verts,
    const std::vector<glm::vec3>& tri_norms,
    const std::vector<glm::vec2>& tri_uvs,
    std::function<VaryingsFull(const glm::vec3&, const glm::vec3&, const glm::vec2&)> vs,
    std::function<shs::Color(const VaryingsFull&)> fs,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    int W = rt.color.get_width();
    int H = rt.color.get_height();

    auto lerp_vary = [](const VaryingsFull& a, const VaryingsFull& b, float t) -> VaryingsFull {
        VaryingsFull o;
        o.position      = a.position      + (b.position      - a.position)      * t;
        o.prev_position = a.prev_position + (b.prev_position - a.prev_position) * t;
        o.world_pos     = a.world_pos     + (b.world_pos     - a.world_pos)     * t;
        o.normal        = a.normal        + (b.normal        - a.normal)        * t;
        o.uv            = a.uv            + (b.uv            - a.uv)            * t;
        o.view_z        = a.view_z        + (b.view_z        - a.view_z)        * t;
        return o;
    };

    auto clip_poly_near_z = [&](const std::vector<VaryingsFull>& in_poly) -> std::vector<VaryingsFull> {
        std::vector<VaryingsFull> out;
        out.reserve(6);

        auto inside = [](const VaryingsFull& v) -> bool {
            return (v.position.w > 1e-6f) && (v.position.z >= 0.0f);
        };

        auto intersect = [&](const VaryingsFull& a, const VaryingsFull& b) -> VaryingsFull {
            float az = a.position.z;
            float bz = b.position.z;
            float denom = (bz - az);
            float t = (std::abs(denom) < 1e-8f) ? 0.0f : ((0.0f - az) / denom);
            t = clampf(t, 0.0f, 1.0f);
            return lerp_vary(a, b, t);
        };

        for (int i = 0; i < (int)in_poly.size(); ++i) {
            const VaryingsFull& A = in_poly[i];
            const VaryingsFull& B = in_poly[(i + 1) % (int)in_poly.size()];

            bool A_in = inside(A);
            bool B_in = inside(B);

            if (A_in && B_in) {
                out.push_back(B);
            } else if (A_in && !B_in) {
                out.push_back(intersect(A, B));
            } else if (!A_in && B_in) {
                out.push_back(intersect(A, B));
                out.push_back(B);
            }
        }

        return out;
    };

    VaryingsFull v0 = vs(tri_verts[0], tri_norms[0], tri_uvs[0]);
    VaryingsFull v1 = vs(tri_verts[1], tri_norms[1], tri_uvs[1]);
    VaryingsFull v2 = vs(tri_verts[2], tri_norms[2], tri_uvs[2]);

    std::vector<VaryingsFull> poly = { v0, v1, v2 };

    poly = clip_poly_near_z(poly);
    if (poly.size() < 3) return;

    for (int ti = 1; ti + 1 < (int)poly.size(); ++ti)
    {
        VaryingsFull tv[3] = { poly[0], poly[ti], poly[ti + 1] };

        glm::vec3 sc3[3];
        for (int i = 0; i < 3; ++i) {
            if (tv[i].position.w <= 1e-6f) goto next_tri;
            sc3[i] = shs::Canvas::clip_to_screen(tv[i].position, W, H);
        }

        {
            glm::vec2 bboxmin(tile_max.x, tile_max.y);
            glm::vec2 bboxmax(tile_min.x, tile_min.y);

            std::vector<glm::vec2> v2d = {
                glm::vec2(sc3[0]),
                glm::vec2(sc3[1]),
                glm::vec2(sc3[2])
            };

            for (int i = 0; i < 3; i++) {
                bboxmin = glm::max(glm::vec2(tile_min), glm::min(bboxmin, v2d[i]));
                bboxmax = glm::min(glm::vec2(tile_max), glm::max(bboxmax, v2d[i]));
            }
            if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) goto next_tri;

            float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) -
                         (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);

            if (std::abs(area) < 1e-8f) goto next_tri;

            for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
                for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

                    glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                    if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                    float vz = bc.x * tv[0].view_z + bc.y * tv[1].view_z + bc.z * tv[2].view_z;

                    if (rt.depth.test_and_set_depth_screen_space(px, py, vz)) {

                        float w0 = tv[0].position.w;
                        float w1 = tv[1].position.w;
                        float w2 = tv[2].position.w;

                        float invw0 = (std::abs(w0) < 1e-6f) ? 0.0f : 1.0f / w0;
                        float invw1 = (std::abs(w1) < 1e-6f) ? 0.0f : 1.0f / w1;
                        float invw2 = (std::abs(w2) < 1e-6f) ? 0.0f : 1.0f / w2;

                        float invw_sum = bc.x * invw0 + bc.y * invw1 + bc.z * invw2;
                        if (invw_sum <= 1e-8f) continue;

                        VaryingsFull in;
                        in.position      = bc.x * tv[0].position      + bc.y * tv[1].position      + bc.z * tv[2].position;
                        in.prev_position = bc.x * tv[0].prev_position + bc.y * tv[1].prev_position + bc.z * tv[2].prev_position;

                        in.normal = glm::normalize(bc.x * tv[0].normal + bc.y * tv[1].normal + bc.z * tv[2].normal);

                        glm::vec3 wp_over_w =
                            bc.x * (tv[0].world_pos * invw0) +
                            bc.y * (tv[1].world_pos * invw1) +
                            bc.z * (tv[2].world_pos * invw2);
                        in.world_pos = wp_over_w / invw_sum;

                        glm::vec2 uv_over_w =
                            bc.x * (tv[0].uv * invw0) +
                            bc.y * (tv[1].uv * invw1) +
                            bc.z * (tv[2].uv * invw2);
                        in.uv = uv_over_w / invw_sum;

                        in.view_z = vz;

                        glm::vec2 curr_s = clip_to_screen_xy(in.position, W, H);
                        glm::vec2 prev_s = clip_to_screen_xy(in.prev_position, W, H);
                        glm::vec2 v_screen = curr_s - prev_s;

                        // Canvas (+Y up) velocity
                        glm::vec2 v_canvas = glm::vec2(v_screen.x, -v_screen.y);

                        float len = glm::length(v_canvas);
                        if (len > MB_MAX_PIXELS && len > 1e-6f) {
                            v_canvas *= (MB_MAX_PIXELS / len);
                        }
                        rt.velocity.set_screen_space(px, py, v_canvas);

                        rt.color.draw_pixel_screen_space(px, py, fs(in));
                    }
                }
            }
        }

    next_tri:
        continue;
    }
}

// ==========================================
// CAMERA-ONLY VELOCITY RECONSTRUCTION (depth + matrices)
// ==========================================

static inline float viewz_to_ndcz(float view_z, const glm::mat4& proj)
{
    glm::vec4 clip = proj * glm::vec4(0.0f, 0.0f, view_z, 1.0f);
    if (std::abs(clip.w) < 1e-6f) return 0.0f;
    return clip.z / clip.w;
}

static inline glm::vec2 canvas_to_ndc_xy(int x, int y, int W, int H)
{
    int py_screen = (H - 1) - y;

    float fx = (float(x) + 0.5f) / float(W);
    float fy = (float(py_screen) + 0.5f) / float(H);

    float ndc_x = fx * 2.0f - 1.0f;
    float ndc_y = 1.0f - fy * 2.0f;
    return glm::vec2(ndc_x, ndc_y);
}

static inline glm::vec2 ndc_to_screen_xy(const glm::vec3& ndc, int W, int H)
{
    float sx = (ndc.x * 0.5f + 0.5f) * float(W - 1);
    float sy = (1.0f - (ndc.y * 0.5f + 0.5f)) * float(H - 1);
    return glm::vec2(sx, sy);
}

static inline glm::vec2 compute_camera_velocity_canvas(
    int x, int y,
    float view_z,
    int W, int H,
    const glm::mat4& curr_viewproj,
    const glm::mat4& prev_viewproj,
    const glm::mat4& curr_proj)
{
    if (view_z == std::numeric_limits<float>::max()) return glm::vec2(0.0f);

    glm::vec2 ndc_xy = canvas_to_ndc_xy(x, y, W, H);
    float ndc_z = viewz_to_ndcz(view_z, curr_proj);

    glm::vec4 clip_curr(ndc_xy.x, ndc_xy.y, ndc_z, 1.0f);

    glm::mat4 inv_curr_vp = glm::inverse(curr_viewproj);
    glm::vec4 world_h     = inv_curr_vp * clip_curr;
    if (std::abs(world_h.w) < 1e-6f) return glm::vec2(0.0f);
    glm::vec3 world = glm::vec3(world_h) / world_h.w;

    glm::vec4 prev_clip = prev_viewproj * glm::vec4(world, 1.0f);
    if (std::abs(prev_clip.w) < 1e-6f) return glm::vec2(0.0f);
    glm::vec3 prev_ndc = glm::vec3(prev_clip) / prev_clip.w;

    int py_screen = (H - 1) - y;
    glm::vec2 curr_screen{ float(x), float(py_screen) };
    glm::vec2 prev_screen = ndc_to_screen_xy(prev_ndc, W, H);

    glm::vec2 v_screen = curr_screen - prev_screen;
    glm::vec2 v_canvas = glm::vec2(v_screen.x, -v_screen.y);
    return v_canvas;
}

static inline glm::vec2 apply_soft_knee(glm::vec2 v, float knee, float max_len)
{
    float len = glm::length(v);
    if (len <= 1e-6f) return v;
    if (len <= knee) return v;

    float t  = (len - knee) / std::max(1e-6f, (max_len - knee));
    float t2 = t / (1.0f + t);
    float new_len = knee + (max_len - knee) * t2;

    return v * (new_len / len);
}

// ==========================================
// COMBINED MOTION BLUR PASS (whole-screen)
// ==========================================

static void combined_motion_blur_pass(
    const shs::Canvas& src,
    const shs::ZBuffer& depth,
    const shs::Buffer<glm::vec2>& v_full_buf, //const shs::MotionBuffer& v_full_buf,
    shs::Canvas& dst,
    const glm::mat4& curr_view,
    const glm::mat4& curr_proj,
    const glm::mat4& prev_view,
    const glm::mat4& prev_proj,
    int samples,
    float strength,
    float w_obj,
    float w_cam,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    int W = src.get_width();
    int H = src.get_height();

    glm::mat4 curr_vp = curr_proj * curr_view;
    glm::mat4 prev_vp = prev_proj * prev_view;

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

                auto sample = [&](int sx, int sy) -> shs::Color {
                    sx = clampi(sx, 0, W - 1);
                    sy = clampi(sy, 0, H - 1);
                    return src.get_color_at(sx, sy);
                };

                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {

                        float vz = depth.get_depth_at(x, y);

                        glm::vec2 v_cam = compute_camera_velocity_canvas(
                            x, y, vz, W, H, curr_vp, prev_vp, curr_proj
                        );

                        //glm::vec2 v_full = v_full_buf.get(x, y);
                        glm::vec2 v_full = v_full_buf.at(x, y); 
                        glm::vec2 v_obj_only = v_full - v_cam;

                        glm::vec2 v_total = w_obj * v_obj_only + w_cam * v_cam;
                        v_total *= strength;

                        if (MB_SOFT_KNEE) {
                            v_total = apply_soft_knee(v_total, MB_KNEE_PIXELS, MB_MAX_PIXELS);
                        }

                        float len = glm::length(v_total);
                        if (len > MB_MAX_PIXELS && len > 1e-6f) {
                            v_total *= (MB_MAX_PIXELS / len);
                            len = MB_MAX_PIXELS;
                        }

                        if (len < 0.001f || samples <= 1) {
                            dst.draw_pixel(x, y, src.get_color_at(x, y));
                            continue;
                        }

                        glm::vec2 dir = v_total / len;

                        float r = 0, g = 0, b = 0;
                        float wsum = 0.0f;

                        for (int i = 0; i < samples; i++) {
                            float t = (samples == 1) ? 0.0f : (float(i) / float(samples - 1));
                            float a = (t - 0.5f) * 2.0f; // -1..+1
                            glm::vec2 p = glm::vec2(float(x), float(y)) + dir * (a * len);

                            int sx = clampi((int)std::round(p.x), 0, W - 1);
                            int sy = clampi((int)std::round(p.y), 0, H - 1);

                            float wgt = 1.0f - std::abs(a);
                            shs::Color c = sample(sx, sy);

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
// REFLECTION CAMERA (planar, y=WATER_Y)
// ==========================================

static inline glm::mat4 compute_reflection_view_lh(const Viewer* viewer)
{
    glm::vec3 pos = viewer->position;
    glm::vec3 dir = viewer->camera->direction_vector;
    glm::vec3 up  = viewer->camera->up_vector;

    // Усны хавтгайн дагуух тусгал (y-тэнхлэгээр толин тусгал)
    pos.y = 2.0f * WATER_Y - pos.y;
    dir.y = -dir.y;
    up.y  = -up.y;

    glm::vec3 target = pos + dir;
    return glm::lookAtLH(pos, target, up);
}

// ==========================================
// SCENE STATE
// ==========================================

class DemoScene : public shs::AbstractSceneState
{
public:
    DemoScene(shs::Canvas* canvas, Viewer* viewer, const shs::Texture2D* car_tex)
    {
        this->canvas = canvas;
        this->viewer = viewer;

        floor  = new FloorPlane(55.0f, 140.0f, FLOOR_Y);
        water  = new WaterPlane(60.0f, 160.0f, WATER_Y);

        car    = new SubaruObject(glm::vec3(-6.0f, 0.0f, 26.0f), glm::vec3(0.08f), car_tex);
        monkey = new MonkeyObject(glm::vec3(-6.0f, 12.2f, 26.0f), glm::vec3(1.65f));

        scene_objects.push_back(car);
        scene_objects.push_back(monkey);
    }

    ~DemoScene()
    {
        for (auto* o : scene_objects) delete o;
        delete floor;
        delete water;
    }

    void process() override {}

    shs::Canvas* canvas;
    Viewer* viewer;

    FloorPlane* floor;
    WaterPlane* water;

    SubaruObject* car;
    MonkeyObject* monkey;

    std::vector<shs::AbstractObject3D*> scene_objects;
};

// ==========================================
// RENDERER SYSTEM (Shadow + Reflection + Camera + MotionBlur)
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(DemoScene* scene, shs::Job::ThreadedPriorityJobSystem* job_sys)
        : scene(scene), job_system(job_sys)
    {
        rt = new shs::RT_ColorDepthMotion(
            CANVAS_WIDTH, CANVAS_HEIGHT,
            scene->viewer->camera->z_near,
            scene->viewer->camera->z_far,
            CLEAR_BG
        );

        mb_out = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CLEAR_BG);

        reflection_rt = new shs::RT_ColorDepthMotion(
            CANVAS_WIDTH, CANVAS_HEIGHT,
            scene->viewer->camera->z_near,
            scene->viewer->camera->z_far,
            CLEAR_BG
        );

        shadow = new shs::ShadowMap(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);

        has_prev_cam = false;
        prev_view = glm::mat4(1.0f);
        prev_proj = glm::mat4(1.0f);

        time_accum = 0.0f;
    }

    ~RendererSystem()
    {
        delete rt;
        delete mb_out;
        delete reflection_rt;
        delete shadow;
    }

    void process(float dt) override
    {
        time_accum += dt;

        glm::mat4 view = scene->viewer->camera->view_matrix;
        glm::mat4 proj = scene->viewer->camera->projection_matrix;

        // Light VP (Directional, Ortho)
        glm::vec3 center(0.0f, 6.0f, 45.0f);
        glm::vec3 light_dir = LIGHT_DIR_WORLD;
        glm::vec3 light_pos = center - light_dir * 80.0f;

        glm::mat4 light_view = glm::lookAtLH(light_pos, center, glm::vec3(0,1,0));

        float L  = -85.0f, R  = 85.0f;
        float B  = -55.0f, T  = 95.0f;
        float zn =   0.1f, zf = 240.0f;

        glm::mat4 light_proj = shs::Math::ortho_lh_zo(L, R, B, T, zn, zf);
        glm::mat4 light_vp   = light_proj * light_view;

        // -----------------------
        // PASS0: ShadowMap depth
        // -----------------------
        shadow->clear();

        {
            int W = shadow->w;
            int H = shadow->h;

            int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
            int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

            wg_shadow.reset();

            for (int ty = 0; ty < rows; ty++) {
                for (int tx = 0; tx < cols; tx++) {

                    wg_shadow.add(1);
                    job_system->submit({[=, this]() {

                        glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                        glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                         std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                        // Floor
                        {
                            Uniforms u;
                            u.model = glm::mat4(1.0f);
                            u.light_vp = light_vp;

                            for (size_t i = 0; i < scene->floor->verts.size(); i += 3) {
                                std::vector<glm::vec3> tri = {
                                    scene->floor->verts[i],
                                    scene->floor->verts[i+1],
                                    scene->floor->verts[i+2]
                                };

                                draw_triangle_tile_shadow(
                                    *shadow,
                                    tri,
                                    [&u](const glm::vec3& p) { return shadow_vertex_shader(p, u); },
                                    t_min, t_max
                                );
                            }
                        }

                        // Objects
                        for (shs::AbstractObject3D* obj : scene->scene_objects)
                        {
                            if (auto* car = dynamic_cast<SubaruObject*>(obj)) {
                                Uniforms u;
                                u.model = car->get_world_matrix();
                                u.light_vp = light_vp;

                                const auto& v = car->geometry->triangles;
                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri = { v[i], v[i+1], v[i+2] };
                                    draw_triangle_tile_shadow(
                                        *shadow,
                                        tri,
                                        [&u](const glm::vec3& p) { return shadow_vertex_shader(p, u); },
                                        t_min, t_max
                                    );
                                }
                            }

                            if (auto* mk = dynamic_cast<MonkeyObject*>(obj)) {
                                Uniforms u;
                                u.model = mk->get_world_matrix();
                                u.light_vp = light_vp;

                                const auto& v = mk->geometry->triangles;
                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri = { v[i], v[i+1], v[i+2] };
                                    draw_triangle_tile_shadow(
                                        *shadow,
                                        tri,
                                        [&u](const glm::vec3& p) { return shadow_vertex_shader(p, u); },
                                        t_min, t_max
                                    );
                                }
                            }
                        }

                        wg_shadow.done();
                    }, shs::Job::PRIORITY_HIGH});
                }
            }

            wg_shadow.wait();
        }

        // -----------------------
        // PASS1A: Reflection pass (planar) -> reflection_rt
        // -----------------------
        reflection_rt->clear(CLEAR_BG);

        glm::mat4 refl_view = compute_reflection_view_lh(scene->viewer);
        glm::mat4 refl_proj = proj;
        glm::mat4 refl_vp   = refl_proj * refl_view;

        {
            int W = reflection_rt->color.get_width();
            int H = reflection_rt->color.get_height();

            int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
            int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

            wg_refl.reset();

            for (int ty = 0; ty < rows; ty++) {
                for (int tx = 0; tx < cols; tx++) {

                    wg_refl.add(1);
                    job_system->submit({[=, this]() {

                        glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                        glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                         std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                        // Усан доорхи шал 
                        {
                            Uniforms u;
                            u.model           = glm::mat4(1.0f);
                            u.view            = refl_view;
                            u.mvp             = refl_proj * refl_view * u.model;
                            u.prev_mvp        = u.mvp;
                            u.light_vp        = light_vp;
                            u.light_dir_world = LIGHT_DIR_WORLD;
                            u.shadow          = shadow;

                            glm::vec3 refl_cam_pos = scene->viewer->position;
                            refl_cam_pos.y = 2.0f * WATER_Y - refl_cam_pos.y;
                            u.camera_pos = refl_cam_pos;

                            u.base_color      = shs::Color{ 85, 95, 115, 255 };
                            u.albedo          = nullptr;
                            u.use_texture     = false;

                            u.reflection_color = nullptr;
                            u.reflection_vp    = glm::mat4(1.0f);
                            u.time_sec         = time_accum;

                            for (size_t i = 0; i < scene->floor->verts.size(); i += 3) {
                                std::vector<glm::vec3> tri_v = {
                                    scene->floor->verts[i],
                                    scene->floor->verts[i+1],
                                    scene->floor->verts[i+2]
                                };
                                std::vector<glm::vec3> tri_n = {
                                    scene->floor->norms[i],
                                    scene->floor->norms[i+1],
                                    scene->floor->norms[i+2]
                                };
                                std::vector<glm::vec2> tri_uv = {
                                    scene->floor->uvs[i],
                                    scene->floor->uvs[i+1],
                                    scene->floor->uvs[i+2]
                                };

                                draw_triangle_tile_color_depth_motion_shadow(
                                    *reflection_rt,
                                    tri_v, tri_n, tri_uv,
                                    [&u](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
                                        return vertex_shader_full(p, n, uv, u);
                                    },
                                    [&u](const VaryingsFull& vin) {
                                        return fragment_shader_full(vin, u);
                                    },
                                    t_min, t_max
                                );
                            }
                        }

                        // car + monkey
                        for (shs::AbstractObject3D* obj : scene->scene_objects)
                        {
                            if (auto* car = dynamic_cast<SubaruObject*>(obj))
                            {
                                glm::mat4 model = car->get_world_matrix();
                                glm::mat4 mvp   = refl_proj * refl_view * model;

                                Uniforms u;
                                u.model           = model;
                                u.view            = refl_view;
                                u.mvp             = mvp;
                                u.prev_mvp        = mvp;
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.shadow          = shadow;

                                glm::vec3 refl_cam_pos = scene->viewer->position;
                                refl_cam_pos.y = 2.0f * WATER_Y - refl_cam_pos.y;
                                u.camera_pos = refl_cam_pos;

                                u.base_color      = shs::Color{200,200,200,255};
                                u.albedo          = car->albedo;
                                u.use_texture     = (car->albedo && car->albedo->valid());

                                u.reflection_color = nullptr;
                                u.reflection_vp    = glm::mat4(1.0f);
                                u.time_sec         = time_accum;

                                const auto& v = car->geometry->triangles;
                                const auto& n = car->geometry->normals;
                                const auto& t = car->geometry->uvs;

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { t[i], t[i+1], t[i+2] };

                                    draw_triangle_tile_color_depth_motion_shadow(
                                        *reflection_rt,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin) {
                                            return fragment_shader_full(vin, u);
                                        },
                                        t_min, t_max
                                    );
                                }
                            }

                            if (auto* mk = dynamic_cast<MonkeyObject*>(obj))
                            {
                                glm::mat4 model = mk->get_world_matrix();
                                glm::mat4 mvp   = refl_proj * refl_view * model;

                                Uniforms u;
                                u.model           = model;
                                u.view            = refl_view;
                                u.mvp             = mvp;
                                u.prev_mvp        = mvp;
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.shadow          = shadow;

                                glm::vec3 refl_cam_pos = scene->viewer->position;
                                refl_cam_pos.y = 2.0f * WATER_Y - refl_cam_pos.y;
                                u.camera_pos = refl_cam_pos;

                                u.base_color      = shs::Color{ 180, 150, 95, 255 };
                                u.albedo          = nullptr;
                                u.use_texture     = false;

                                u.reflection_color = nullptr;
                                u.reflection_vp    = glm::mat4(1.0f);
                                u.time_sec         = time_accum;

                                const auto& v = mk->geometry->triangles;
                                const auto& n = mk->geometry->normals;
                                static const glm::vec2 uv0(0.0f);

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { uv0, uv0, uv0 };

                                    draw_triangle_tile_color_depth_motion_shadow(
                                        *reflection_rt,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin) {
                                            return fragment_shader_full(vin, u);
                                        },
                                        t_min, t_max
                                    );
                                }
                            }
                        }

                        wg_refl.done();
                    }, shs::Job::PRIORITY_HIGH});
                }
            }

            wg_refl.wait();
        }

        // -----------------------
        // PASS1B: Camera render -> rt (Floor + Water + Objects)
        // -----------------------
        rt->clear(CLEAR_BG);

        {
            int W = rt->color.get_width();
            int H = rt->color.get_height();

            int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
            int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

            wg_cam.reset();

            for (int ty = 0; ty < rows; ty++) {
                for (int tx = 0; tx < cols; tx++) {

                    wg_cam.add(1);
                    job_system->submit({[=, this]() {

                        glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                        glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                         std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                        // Усан доохир шал 
                        {
                            Uniforms u;
                            u.model           = glm::mat4(1.0f);
                            u.view            = view;
                            u.mvp             = proj * view * u.model;
                            u.prev_mvp        = u.mvp;
                            u.light_vp        = light_vp;
                            u.light_dir_world = LIGHT_DIR_WORLD;
                            u.camera_pos      = scene->viewer->position;
                            u.shadow          = shadow;

                            u.base_color      = shs::Color{ 92, 105, 125, 255 };
                            u.albedo          = nullptr;
                            u.use_texture     = false;

                            u.reflection_color = &reflection_rt->color;
                            u.reflection_vp    = refl_vp;
                            u.time_sec         = time_accum;

                            for (size_t i = 0; i < scene->floor->verts.size(); i += 3) {
                                std::vector<glm::vec3> tri_v = {
                                    scene->floor->verts[i],
                                    scene->floor->verts[i+1],
                                    scene->floor->verts[i+2]
                                };
                                std::vector<glm::vec3> tri_n = {
                                    scene->floor->norms[i],
                                    scene->floor->norms[i+1],
                                    scene->floor->norms[i+2]
                                };
                                std::vector<glm::vec2> tri_uv = {
                                    scene->floor->uvs[i],
                                    scene->floor->uvs[i+1],
                                    scene->floor->uvs[i+2]
                                };

                                draw_triangle_tile_color_depth_motion_shadow(
                                    *rt,
                                    tri_v, tri_n, tri_uv,
                                    [&u](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
                                        return vertex_shader_full(p, n, uv, u);
                                    },
                                    [&u](const VaryingsFull& vin) {
                                        return fragment_shader_full(vin, u);
                                    },
                                    t_min, t_max
                                );
                            }
                        }

                        // Усны хавтан
                        {
                            Uniforms u;
                            u.model           = glm::mat4(1.0f);
                            u.view            = view;
                            u.mvp             = proj * view * u.model;
                            u.prev_mvp        = u.mvp;
                            u.light_vp        = light_vp;
                            u.light_dir_world = LIGHT_DIR_WORLD;
                            u.camera_pos      = scene->viewer->position;
                            u.shadow          = shadow;

                            u.base_color      = shs::Color{ 40, 90, 105, 255 };
                            u.albedo          = nullptr;
                            u.use_texture     = false;

                            u.reflection_color = &reflection_rt->color;
                            u.reflection_vp    = refl_vp;
                            u.time_sec         = time_accum;

                            for (size_t i = 0; i < scene->water->verts.size(); i += 3) {
                                std::vector<glm::vec3> tri_v = {
                                    scene->water->verts[i],
                                    scene->water->verts[i+1],
                                    scene->water->verts[i+2]
                                };
                                std::vector<glm::vec3> tri_n = {
                                    scene->water->norms[i],
                                    scene->water->norms[i+1],
                                    scene->water->norms[i+2]
                                };
                                std::vector<glm::vec2> tri_uv = {
                                    scene->water->uvs[i],
                                    scene->water->uvs[i+1],
                                    scene->water->uvs[i+2]
                                };

                                draw_triangle_tile_color_depth_motion_shadow(
                                    *rt,
                                    tri_v, tri_n, tri_uv,
                                    [&u](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
                                        return vertex_shader_full(p, n, uv, u);
                                    },
                                    [&u](const VaryingsFull& vin) {
                                        return fragment_shader_water(vin, u);
                                    },
                                    t_min, t_max
                                );
                            }
                        }

                        // car + monkey
                        for (shs::AbstractObject3D* obj : scene->scene_objects)
                        {
                            if (auto* car = dynamic_cast<SubaruObject*>(obj))
                            {
                                glm::mat4 model = car->get_world_matrix();
                                glm::mat4 mvp   = proj * view * model;

                                glm::mat4 prev_mvp = mvp;
                                if (car->has_prev_mvp) prev_mvp = car->prev_mvp;

                                Uniforms u;
                                u.model           = model;
                                u.view            = view;
                                u.mvp             = mvp;
                                u.prev_mvp        = prev_mvp;
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.camera_pos      = scene->viewer->position;
                                u.shadow          = shadow;

                                u.base_color      = shs::Color{200,200,200,255};
                                u.albedo          = car->albedo;
                                u.use_texture     = (car->albedo && car->albedo->valid());

                                u.reflection_color = &reflection_rt->color;
                                u.reflection_vp    = refl_vp;
                                u.time_sec         = time_accum;

                                const auto& v = car->geometry->triangles;
                                const auto& n = car->geometry->normals;
                                const auto& t = car->geometry->uvs;

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { t[i], t[i+1], t[i+2] };

                                    draw_triangle_tile_color_depth_motion_shadow(
                                        *rt,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin) {
                                            return fragment_shader_full(vin, u);
                                        },
                                        t_min, t_max
                                    );
                                }
                            }

                            if (auto* mk = dynamic_cast<MonkeyObject*>(obj))
                            {
                                glm::mat4 model = mk->get_world_matrix();
                                glm::mat4 mvp   = proj * view * model;

                                glm::mat4 prev_mvp = mvp;
                                if (mk->has_prev_mvp) prev_mvp = mk->prev_mvp;

                                Uniforms u;
                                u.model           = model;
                                u.view            = view;
                                u.mvp             = mvp;
                                u.prev_mvp        = prev_mvp;
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.camera_pos      = scene->viewer->position;
                                u.shadow          = shadow;

                                u.base_color      = shs::Color{ 180, 150, 95, 255 };
                                u.albedo          = nullptr;
                                u.use_texture     = false;

                                u.reflection_color = &reflection_rt->color;
                                u.reflection_vp    = refl_vp;
                                u.time_sec         = time_accum;

                                const auto& v = mk->geometry->triangles;
                                const auto& n = mk->geometry->normals;
                                static const glm::vec2 uv0(0.0f);

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { uv0, uv0, uv0 };

                                    draw_triangle_tile_color_depth_motion_shadow(
                                        *rt,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin) {
                                            return fragment_shader_full(vin, u);
                                        },
                                        t_min, t_max
                                    );
                                }
                            }
                        }

                        wg_cam.done();
                    }, shs::Job::PRIORITY_HIGH});
                }
            }

            wg_cam.wait();
        }

        // per-object prev_mvp commit
        {
            glm::mat4 view2 = scene->viewer->camera->view_matrix;
            glm::mat4 proj2 = scene->viewer->camera->projection_matrix;

            for (shs::AbstractObject3D* obj : scene->scene_objects)
            {
                if (auto* car = dynamic_cast<SubaruObject*>(obj)) {
                    glm::mat4 model = car->get_world_matrix();
                    car->prev_mvp = proj2 * view2 * model;
                    car->has_prev_mvp = true;
                }
                if (auto* mk = dynamic_cast<MonkeyObject*>(obj)) {
                    glm::mat4 model = mk->get_world_matrix();
                    mk->prev_mvp = proj2 * view2 * model;
                    mk->has_prev_mvp = true;
                }
            }
        }

        // -----------------------
        // PASS2: Combined Motion Blur
        // -----------------------
        glm::mat4 curr_view = scene->viewer->camera->view_matrix;
        glm::mat4 curr_proj = scene->viewer->camera->projection_matrix;

        if (!has_prev_cam) {
            prev_view = curr_view;
            prev_proj = curr_proj;
            has_prev_cam = true;
        }

        combined_motion_blur_pass(
            rt->color,
            rt->depth,
            rt->velocity, //rt->motion,
            *mb_out,
            curr_view,
            curr_proj,
            prev_view,
            prev_proj,
            MB_SAMPLES,
            MB_STRENGTH,
            MB_W_OBJ,
            MB_W_CAM,
            job_system,
            wg_mb
        );

        prev_view = curr_view;
        prev_proj = curr_proj;
    }

    shs::Canvas& output() { return *mb_out; }

private:
    DemoScene* scene;
    shs::Job::ThreadedPriorityJobSystem* job_system;

    shs::RT_ColorDepthMotion* rt;
    shs::RT_ColorDepthMotion* reflection_rt;
    shs::Canvas* mb_out;

    shs::ShadowMap* shadow;

    shs::Job::WaitGroup wg_shadow;
    shs::Job::WaitGroup wg_refl;
    shs::Job::WaitGroup wg_cam;
    shs::Job::WaitGroup wg_mb;

    bool has_prev_cam;
    glm::mat4 prev_view;
    glm::mat4 prev_proj;

    float time_accum;
};

// ==========================================
// LOGIC SYSTEM
// ==========================================

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(DemoScene* scene) : scene(scene) {}
    void process(float dt) override
    {
        scene->viewer->update();
        for (auto* o : scene->scene_objects) o->update(dt);
    }
private:
    DemoScene* scene;
};

// ==========================================
// SYSTEM PROCESSOR
// ==========================================

class SystemProcessor
{
public:
    SystemProcessor(DemoScene* scene, shs::Job::ThreadedPriorityJobSystem* job_sys)
    {
        command_processor = new shs::CommandProcessor();
        logic_system      = new LogicSystem(scene);
        renderer_system   = new RendererSystem(scene, job_sys);
    }

    ~SystemProcessor()
    {
        delete command_processor;
        delete logic_system;
        delete renderer_system;
    }

    void process(float dt)
    {
        command_processor->process();
        logic_system->process(dt);
    }

    void render(float dt)
    {
        renderer_system->process(dt);
    }

    shs::Canvas& output() { return renderer_system->output(); }

    shs::CommandProcessor* command_processor;
    LogicSystem*           logic_system;
    RendererSystem*        renderer_system;
};



// ==========================================
// MAIN
// ==========================================

int main(int argc, char* argv[])
{
    
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    IMG_Init(IMG_INIT_PNG | IMG_INIT_JPG);

    auto* job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas* screen_canvas  = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CLEAR_BG);
    SDL_Surface* screen_surface = screen_canvas->create_sdl_surface();
    SDL_Texture* screen_texture = SDL_CreateTextureFromSurface(renderer, screen_surface);

    shs::Texture2D car_tex = shs::load_texture_sdl_image("./assets/obj/subaru/SUBARU1_M.bmp", true);

    Viewer*    viewer    = new Viewer(glm::vec3(0.0f, 10.0f, -42.0f), 55.0f);
    DemoScene* scene     = new DemoScene(screen_canvas, viewer, &car_tex);

    SystemProcessor* sys = new SystemProcessor(scene, job_system);

    bool exit = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();
    bool is_dragging = false;

    while (!exit)
    {
        Uint32 current_tick = SDL_GetTicks();
        float dt = (current_tick - last_tick) / 1000.0f;
        last_tick = current_tick;

        while (SDL_PollEvent(&e))
        {
            if (e.type == SDL_QUIT) exit = true;

            if (e.type == SDL_MOUSEBUTTONDOWN) {
                if (e.button.button == SDL_BUTTON_LEFT) is_dragging = true;
            }
            if (e.type == SDL_MOUSEBUTTONUP) {
                if (e.button.button == SDL_BUTTON_LEFT) is_dragging = false;
            }

            if (e.type == SDL_MOUSEMOTION)
            {
                if (is_dragging) {
                    viewer->horizontal_angle += e.motion.xrel * MOUSE_SENSITIVITY;
                    viewer->vertical_angle   -= e.motion.yrel * MOUSE_SENSITIVITY;

                    if (viewer->vertical_angle >  89.0f) viewer->vertical_angle =  89.0f;
                    if (viewer->vertical_angle < -89.0f) viewer->vertical_angle = -89.0f;
                }
            }

            if (e.type == SDL_KEYDOWN)
            {
                if (e.key.keysym.sym == SDLK_ESCAPE) exit = true;

                if (e.key.keysym.sym == SDLK_w) sys->command_processor->add_command(
                    new shs::MoveForwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, dt)
                );
                if (e.key.keysym.sym == SDLK_s) sys->command_processor->add_command(
                    new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, dt)
                );
                if (e.key.keysym.sym == SDLK_a) sys->command_processor->add_command(
                    new shs::MoveLeftCommand(viewer->position, viewer->get_right_vector(), viewer->speed, dt)
                );
                if (e.key.keysym.sym == SDLK_d) sys->command_processor->add_command(
                    new shs::MoveRightCommand(viewer->position, viewer->get_right_vector(), viewer->speed, dt)
                );
            }
        }

        sys->process(dt);
        sys->render(dt);

        screen_canvas->buffer() = sys->output().buffer();
        shs::Canvas::copy_to_SDLSurface(screen_surface, screen_canvas);

        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(renderer);
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

    IMG_Quit();
    SDL_Quit();
    return 0;
}

