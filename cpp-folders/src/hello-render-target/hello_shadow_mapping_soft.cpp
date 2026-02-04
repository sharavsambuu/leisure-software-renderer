/*

    Soft Shadows PCSS (Percentage Closer Soft Shadows)


Зорилго:
Өмнөч Shadow Mapping (PASS0 depth + PASS1 shading) хэрэгжүүлэлтээ ашиглаад
penumbra сүүдэр бүхий soft shadows хэрэгжүүлэх


PCSS хэрэгжүүлэлт :
1) PASS0: Directional light-аас ortho projection ашиглан shadow map (depth) бичнэ.
2) PASS1: Pixel бүр дээр light-space (uv,z) координат гаргаж авна:
    - Blocker Search    : Receiver (pixel)-ийн дор байгаа blocker-уудын дундаж depth-ийг (avgBlockerZ) олно.
    - Penumbra Estimate : (zReceiver - avgBlockerZ) / avgBlockerZ гэсэн харьцаагаар penumbra хэмжээ (radius) тооцно.
    - Variable PCF      : Penumbra radius-тай proportional radius-аар олон sample авч зөөлөн filter хийнэ.

Soft shadow болгож байгаа шалтгаанууд: 
- Receiver ба blocker-ийн зай ихсэх тусам penumbra өргөснө (сүүдрийн ирмэг улам их blur-даж эхлэнэ).
- Blocker ойрхон бол penumbra жижиг буюу хатуу маягтай сүүдэр болно.


Бүрэлдэхүүн хэсгүүд
- ShadowMap: 
    depth buffer (float, light NDC z: 0..1)
- shadow_uvz_from_world(): 
    world_pos -> (uv, z_ndc)
- pcss_shadow_factor(): 
    PCSS алгоритм (blocker search + penumbra + pcf)

Тайлбарууд
- LIGHT_UV_RADIUS_BASE: 
  гэрлийн диск том байх тусам сүүдэр зөөлөн болно
- PCSS_BLOCKER_SAMPLES: 
  blocker search дээжүүдийн тоо
- PCSS_PCF_SAMPLES: 
  PCF дээжүүдийн тоо
- PCSS_MIN/MAX_FILTER_RADIUS_TEXELS: 
  хэт их blur хийгдэхээс сэргийлнэ
- SHADOW_BIAS_BASE + SHADOW_BIAS_SLOPE: 
  acne vs peter-panning тэнцвэрийг барина

*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

// ------------------------------------------
// WINDOW / CANVAS
// ------------------------------------------
#define WINDOW_WIDTH      800
#define WINDOW_HEIGHT     600
#define CANVAS_WIDTH      800
#define CANVAS_HEIGHT     600

#define MOUSE_SENSITIVITY 0.2f

// ------------------------------------------
// THREAD / TILING
// ------------------------------------------
#define THREAD_COUNT      20
#define TILE_SIZE_X       160
#define TILE_SIZE_Y       160

// ------------------------------------------
// SHADOW MAP CONFIG
// ------------------------------------------
#define SHADOW_MAP_SIZE   2048

static const glm::vec3 LIGHT_DIR_WORLD = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f));

// Bias (acne vs peter-panning)
static const float SHADOW_BIAS_BASE   = 0.0025f;
static const float SHADOW_BIAS_SLOPE  = 0.0100f;

// ------------------------------------------
// PCSS (SOFT SHADOWS) CONFIG
// ------------------------------------------
// Гэрлийн хэмжээ томрох тусам penumbra томрох буюу soft болно
// Үүнийг UV-space дээрх үндсэн radius гэж ойлгож болно (0..1)
static const float LIGHT_UV_RADIUS_BASE = 0.0035f;

// Blocker search radius (receiver-ээс blocker хайх хүрээ) - texel-ээр clamp хийнэ
static const float PCSS_BLOCKER_SEARCH_RADIUS_TEXELS = 18.0f;

// Filter radius clamp (PCF-ийн final blur) - хэт их blur болохоос хамгаална
static const float PCSS_MIN_FILTER_RADIUS_TEXELS = 1.0f;
static const float PCSS_MAX_FILTER_RADIUS_TEXELS = 42.0f;

// Sample counts (ихсэх тусам CPU тооцооллын өртөг нэмэгдэнэ)
static const int   PCSS_BLOCKER_SAMPLES = 12;
static const int   PCSS_PCF_SAMPLES     = 24;

// Blocker олдохгүй үед: full lit (1.0) гэж үзнэ
// Penumbra тооцооллын тогтвортой байдал
static const float PCSS_EPSILON = 1e-5f;

// ------------------------------------------
// HELPERS
// ------------------------------------------
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
static inline float clamp01(float v)
{
    return (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);
}
static inline glm::vec3 color_to_rgb01(const shs::Color& c)
{
    return glm::vec3(float(c.r), float(c.g), float(c.b)) / 255.0f;
}
static inline shs::Color rgb01_to_color(const glm::vec3& c01)
{
    glm::vec3 c = glm::clamp(c01, 0.0f, 1.0f) * 255.0f;
    return shs::Color{ (uint8_t)c.x, (uint8_t)c.y, (uint8_t)c.z, 255 };
}

// ------------------------------------------
// LH Ortho matrix (NDC z: 0..1)
// ------------------------------------------
static inline glm::mat4 ortho_lh_zo(float left, float right, float bottom, float top, float znear, float zfar)
{
    glm::mat4 m(1.0f);
    m[0][0] =  2.0f / (right - left);
    m[1][1] =  2.0f / (top - bottom);
    m[2][2] =  1.0f / (zfar - znear);

    m[3][0] = -(right + left) / (right - left);
    m[3][1] = -(top + bottom) / (top - bottom);
    m[3][2] = -znear / (zfar - znear);
    return m;
}

// ------------------------------------------
// TEXTURE SAMPLER (nearest)
// ------------------------------------------
#define UV_FLIP_V 0
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
// SHADOW MAP BUFFER (Depth only)
// ==========================================
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

// ==========================================
// SHADOW HELPERS
// гаралтын uv нь shadow-map ийн тохиролцооны дагуу байна (0,0 top-left, y down)
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
    out_z_ndc     = ndc.z;

    // light frustum-аас гадуур бол shadow хэрэглэхгүй (lit гэж үзнэ)
    if (out_z_ndc < 0.0f || out_z_ndc > 1.0f) return false;

    out_uv.x = ndc.x * 0.5f + 0.5f;
    out_uv.y = 1.0f - (ndc.y * 0.5f + 0.5f);

    return true;
}

static inline float shadow_sample_depth_uv(const ShadowMap& sm, glm::vec2 uv)
{
    // uv -> nearest depth
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f)
        return std::numeric_limits<float>::max();

    int x = (int)std::lround(uv.x * float(sm.w - 1));
    int y = (int)std::lround(uv.y * float(sm.h - 1));
    return sm.sample(x, y);
}

// ==========================================
// Poisson disk offsets (2D) — fixed pattern
// - CPU дээр sample count бага үед давталтыг нь багасгах зорилготой
// ==========================================
static const glm::vec2 POISSON_32[32] =
{
    glm::vec2(-0.613392f,  0.617481f),
    glm::vec2( 0.170019f, -0.040254f),
    glm::vec2(-0.299417f,  0.791925f),
    glm::vec2( 0.645680f,  0.493210f),
    glm::vec2(-0.651784f,  0.717887f),
    glm::vec2( 0.421003f,  0.027070f),
    glm::vec2(-0.817194f, -0.271096f),
    glm::vec2(-0.705374f, -0.668203f),
    glm::vec2( 0.977050f, -0.108615f),
    glm::vec2( 0.063326f,  0.142369f),
    glm::vec2( 0.203528f,  0.214331f),
    glm::vec2(-0.667531f,  0.326090f),
    glm::vec2(-0.098422f, -0.295755f),
    glm::vec2(-0.885922f,  0.215369f),
    glm::vec2( 0.566637f,  0.605213f),
    glm::vec2( 0.039766f, -0.396100f),
    glm::vec2( 0.751946f,  0.453352f),
    glm::vec2( 0.078707f, -0.715323f),
    glm::vec2(-0.075838f, -0.529344f),
    glm::vec2( 0.724479f, -0.580798f),
    glm::vec2( 0.222999f, -0.215125f),
    glm::vec2(-0.467574f, -0.405438f),
    glm::vec2(-0.248268f, -0.814753f),
    glm::vec2( 0.354411f, -0.887570f),
    glm::vec2( 0.175817f,  0.382366f),
    glm::vec2( 0.487472f, -0.063082f),
    glm::vec2(-0.084078f,  0.898312f),
    glm::vec2( 0.488876f, -0.783441f),
    glm::vec2( 0.470016f,  0.217933f),
    glm::vec2(-0.696890f, -0.549791f),
    glm::vec2(-0.149693f,  0.605762f),
    glm::vec2( 0.034211f,  0.979980f)
};

// ------------------------------------------
// Hash / rotate (per-pixel) — banding багасгах
// ------------------------------------------
static inline uint32_t hash_u32(uint32_t x)
{
    // mix
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}
static inline float hash01(uint32_t x)
{
    return float(hash_u32(x) & 0x00FFFFFFu) / float(0x01000000u);
}
static inline glm::vec2 rotate2(const glm::vec2& p, float a)
{
    float c = std::cos(a);
    float s = std::sin(a);
    return glm::vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

// ==========================================
// PCSS — SOFT SHADOW FACTOR
// return: 
//      1.0 = lit
//      0.0 = full shadow
// ==========================================
static inline float pcss_shadow_factor(
    const ShadowMap& sm,
    const glm::vec2& uv,
    float z_receiver,
    float bias,
    int px, int py) // per-pixel рандом утгаар дүүргэхэд зориулсан дэлгэцийн пикселийн байршил
{
    // UV гадуур бол lit
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;

    // Shadow map дээр depth бичигдээгүй хэсэг -> lit
    float centerDepth = shadow_sample_depth_uv(sm, uv);
    if (centerDepth == std::numeric_limits<float>::max()) return 1.0f;

    // --------------------------------------
    // Blocker Search
    // --------------------------------------
    // texel space radius -> uv space
    float texelSizeU = 1.0f / float(sm.w);
    float texelSizeV = 1.0f / float(sm.h);

    float searchRadiusTex = PCSS_BLOCKER_SEARCH_RADIUS_TEXELS;
    float searchRadiusU   = searchRadiusTex * texelSizeU;
    float searchRadiusV   = searchRadiusTex * texelSizeV;

    // per-pixel rotation
    uint32_t seed = (uint32_t)(px * 1973u ^ py * 9277u ^ 0x9e3779b9u);
    float ang = hash01(seed) * 6.2831853f;

    float blockerSum = 0.0f;
    int   blockerCnt = 0;

    // blocker: depth < receiver - bias
    float zTest = z_receiver - bias;

    for (int i = 0; i < PCSS_BLOCKER_SAMPLES; ++i)
    {
        glm::vec2 o   = POISSON_32[i & 31];
        o             = rotate2(o, ang);

        glm::vec2 suv = uv + glm::vec2(o.x * searchRadiusU, o.y * searchRadiusV);

        float d = shadow_sample_depth_uv(sm, suv);
        if (d == std::numeric_limits<float>::max()) continue;

        if (d < zTest)
        {
            blockerSum += d;
            blockerCnt++;
        }
    }

    // blocker байхгүй бол shadow байхгүй гэж үзнэ
    if (blockerCnt <= 0) return 1.0f;

    float avgBlocker = blockerSum / float(blockerCnt);

    // --------------------------------------
    // Penumbra ойролцоолох
    // --------------------------------------
    // Directional light-ийн PCSS-ийн түгээмэл approximation:
    // penumbra ~ (zR - zB) / zB * lightSize
    float zB = std::max(PCSS_EPSILON, avgBlocker);
    float zR = std::max(PCSS_EPSILON, z_receiver);

    float penumbraRatio = (zR - zB) / zB;
    penumbraRatio = std::max(0.0f, penumbraRatio);

    // UV radius base нь гэрлийн хэмжээний proxy
    float filterRadiusUvU  = LIGHT_UV_RADIUS_BASE * penumbraRatio;
    float filterRadiusUvV  = LIGHT_UV_RADIUS_BASE * penumbraRatio;

    // texel clamp (хэт их blur хийгдэхээс хамгаална)
    float filterRadiusTexU = filterRadiusUvU / texelSizeU;
    float filterRadiusTexV = filterRadiusUvV / texelSizeV;

    float filterRadiusTex  = 0.5f * (filterRadiusTexU + filterRadiusTexV);
    filterRadiusTex        = clampf(filterRadiusTex, PCSS_MIN_FILTER_RADIUS_TEXELS, PCSS_MAX_FILTER_RADIUS_TEXELS);

    filterRadiusUvU = filterRadiusTex * texelSizeU;
    filterRadiusUvV = filterRadiusTex * texelSizeV;

    // --------------------------------------
    // Variable PCF (final soft filtering)
    // --------------------------------------
    float litSum = 0.0f;
    int   litCnt = 0;

    // дахин өөр seed ашиглаад rotation өөрчилж болно
    float ang2 = hash01(seed ^ 0xB5297A4Du) * 6.2831853f;

    for (int i = 0; i < PCSS_PCF_SAMPLES; ++i)
    {
        glm::vec2 o = POISSON_32[i & 31];
        o = rotate2(o, ang2);

        glm::vec2 suv = uv + glm::vec2(o.x * filterRadiusUvU, o.y * filterRadiusUvV);

        float d = shadow_sample_depth_uv(sm, suv);
        if (d == std::numeric_limits<float>::max()) {
            // map-д бичигдээгүй хэсгийг lit гэж үзээд тогтвортой болгож болно
            litSum += 1.0f;
            litCnt++;
            continue;
        }

        litSum += (z_receiver <= d + bias) ? 1.0f : 0.0f;
        litCnt++;
    }

    if (litCnt <= 0) return 1.0f;
    return litSum / float(litCnt);
}

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
// GEOMETRY (Assimp) - triangles + normals + uvs
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
// FLOOR (tessellated grid) - XZ plane at y=0
// ==========================================
struct FloorPlane
{
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> norms;
    std::vector<glm::vec2> uvs;

    FloorPlane(float half_size, float z_forward)
    {
        verts.clear(); norms.clear(); uvs.clear();

        const int GRID_X = 48;
        const int GRID_Z = 48;

        float y  = 0.0f;
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
// SCENE OBJECTS
// ==========================================
class SubaruObject : public shs::AbstractObject3D
{
public:
    SubaruObject(glm::vec3 position, glm::vec3 scale, const shs::Texture2D *albedo)
    {
        this->position       = position;
        this->scale          = scale;
        this->geometry       = new ModelGeometry("./obj/subaru/SUBARU_1.obj");
        this->rotation_angle = 0.0f;
        this->albedo         = albedo;
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
        rotation_angle += 12.0f * dt;
        if (rotation_angle >= 360.0f) rotation_angle -= 360.0f;
    }

    void render() override {}

    ModelGeometry        *geometry;
    const shs::Texture2D *albedo;

    glm::vec3 position;
    glm::vec3 scale;
    float     rotation_angle;
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 base_pos, glm::vec3 scale)
    {
        this->geometry           = new ModelGeometry("./obj/monkey/monkey.rawobj");

        this->base_position      = base_pos;
        this->position           = base_pos;
        this->scale              = scale;

        this->time_accum         = 0.0f;
        this->rotation_angle     = 0.0f;

        this->spin_deg_per_sec   = 320.0f;
        this->wobble_hz          = 2.6f;
        this->wobble_amp_y       = 0.55f;
        this->wobble_amp_xz      = 0.35f;
        this->wobble_phase_speed = 6.2831853f;
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
        time_accum     += dt;

        float w         = wobble_phase_speed * wobble_hz;

        position        = base_position;
        position.y     += std::sin(time_accum * w) * wobble_amp_y;

        position.x     += std::cos(time_accum * w * 1.15f) * wobble_amp_xz;
        position.z     += std::sin(time_accum * w * 0.95f) * wobble_amp_xz;

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
    float wobble_phase_speed;
};

// ==========================================
// UNIFORMS & VARYINGS
// ==========================================
struct Uniforms
{
    glm::mat4 mvp;
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 mv;
    glm::mat3 normal_mat;

    glm::mat4 light_vp;

    glm::vec3 light_dir_world;
    glm::vec3 camera_pos;

    shs::Color base_color;
    const shs::Texture2D *albedo = nullptr;
    bool use_texture = false;

    const ShadowMap *shadow = nullptr;
};

struct VaryingsFull
{
    glm::vec4 position;
    glm::vec3 world_pos;
    glm::vec3 normal;
    glm::vec2 uv;
    float     view_z;
};

// ==========================================
// VERTEX SHADER
// ==========================================
static VaryingsFull vertex_shader_full(
    const glm::vec3& aPos,
    const glm::vec3& aNormal,
    const glm::vec2& aUV,
    const Uniforms&  u)
{
    VaryingsFull out;

    out.position      = u.mvp * glm::vec4(aPos, 1.0f);

    glm::vec4 world_h = u.model * glm::vec4(aPos, 1.0f);
    out.world_pos     = glm::vec3(world_h);

    out.normal        = glm::normalize(u.normal_mat * aNormal);
    out.uv            = aUV;

    glm::vec4 view_pos = u.mv * glm::vec4(aPos, 1.0f);
    out.view_z = view_pos.z;

    return out;
}

// ==========================================
// SHADOW PASS VARYINGS (depth only)
// ==========================================
struct VaryingsShadow
{
    glm::vec4 position; // light clip
};

static inline VaryingsShadow shadow_vertex_shader(const glm::vec3& aPos, const Uniforms& u)
{
    VaryingsShadow out;
    out.position = u.light_vp * u.model * glm::vec4(aPos, 1.0f);
    return out;
}

// ==========================================
// SHADOW MAP RASTER
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
    ShadowMap& sm,
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
// CAMERA PASS RASTER: Color + Depth + Soft Shadows
// (near-plane clipping in clip-space: z >= 0)
// ==========================================
static void draw_triangle_tile_color_depth_softshadow(
    shs::Canvas&  color,
    shs::ZBuffer& depth,
    const std::vector<glm::vec3>& tri_verts,
    const std::vector<glm::vec3>& tri_norms,
    const std::vector<glm::vec2>& tri_uvs,
    std::function<VaryingsFull(const glm::vec3&, const glm::vec3&, const glm::vec2&)> vs,
    std::function<shs::Color(const VaryingsFull&, int px, int py)> fs,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    int W = color.get_width();
    int H = color.get_height();

    auto lerp_vary = [](const VaryingsFull& a, const VaryingsFull& b, float t) -> VaryingsFull {
        VaryingsFull o;
        o.position  = a.position  + (b.position  - a.position)  * t;
        o.world_pos = a.world_pos + (b.world_pos - a.world_pos) * t;
        o.normal    = a.normal    + (b.normal    - a.normal)    * t;
        o.uv        = a.uv        + (b.uv        - a.uv)        * t;
        o.view_z    = a.view_z    + (b.view_z    - a.view_z)    * t;
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

        bool tri_ok = true;

        glm::vec3 sc3[3];
        for (int i = 0; i < 3; ++i) {
            if (tv[i].position.w <= 1e-6f) { tri_ok = false; break; }
            sc3[i] = shs::Canvas::clip_to_screen(tv[i].position, W, H);
        }
        if (!tri_ok) continue;

        glm::vec2 bboxmin(tile_max.x, tile_max.y);
        glm::vec2 bboxmax(tile_min.x, tile_min.y);

        std::vector<glm::vec2> v2d = { glm::vec2(sc3[0]), glm::vec2(sc3[1]), glm::vec2(sc3[2]) };
        for (int i = 0; i < 3; i++) {
            bboxmin = glm::max(glm::vec2(tile_min), glm::min(bboxmin, v2d[i]));
            bboxmax = glm::min(glm::vec2(tile_max), glm::max(bboxmax, v2d[i]));
        }
        if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) continue;

        float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) -
                     (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
        if (std::abs(area) < 1e-8f) continue;

        for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
            for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

                glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                float vz = bc.x * tv[0].view_z + bc.y * tv[1].view_z + bc.z * tv[2].view_z;

                int cy = (H - 1) - py;

                if (depth.test_and_set_depth(px, cy, vz)) {

                    float w0 = tv[0].position.w;
                    float w1 = tv[1].position.w;
                    float w2 = tv[2].position.w;

                    float invw0 = (std::abs(w0) < 1e-6f) ? 0.0f : 1.0f / w0;
                    float invw1 = (std::abs(w1) < 1e-6f) ? 0.0f : 1.0f / w1;
                    float invw2 = (std::abs(w2) < 1e-6f) ? 0.0f : 1.0f / w2;

                    float invw_sum = bc.x * invw0 + bc.y * invw1 + bc.z * invw2;
                    if (invw_sum <= 1e-8f) continue;

                    VaryingsFull in;
                    in.position = bc.x * tv[0].position + bc.y * tv[1].position + bc.z * tv[2].position;
                    in.normal   = glm::normalize(bc.x * tv[0].normal + bc.y * tv[1].normal + bc.z * tv[2].normal);

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

                    color.draw_pixel_screen_space(px, py, fs(in, px, py));
                }
            }
        }
    }

    
}

// ==========================================
// FRAGMENT SHADER (Direct Blinn-Phong + PCSS Soft Shadow)
// ==========================================
static shs::Color fragment_shader_softshadow(const VaryingsFull& in, const Uniforms& u, int px, int py)
{
    glm::vec3 N = glm::normalize(in.normal);
    glm::vec3 L = glm::normalize(-u.light_dir_world);
    glm::vec3 V = glm::normalize(u.camera_pos - in.world_pos);

    // BaseColor
    glm::vec3 baseColor;
    if (u.use_texture && u.albedo && u.albedo->valid()) {
        shs::Color tc = sample_nearest(*u.albedo, in.uv);
        baseColor = color_to_rgb01(tc);
    } else {
        baseColor = color_to_rgb01(u.base_color);
    }

    // Blinn-Phong
    float ambientStrength = 0.18f;

    float diff = glm::max(glm::dot(N, L), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f);

    glm::vec3 H = glm::normalize(L + V);
    float specularStrength = 0.45f;
    float shininess = 64.0f;
    float spec = glm::pow(glm::max(glm::dot(N, H), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f);

    // --------------------------------------
    // PCSS shadow factor (1=lit, 0=shadow)
    // --------------------------------------
    float shadow = 1.0f;
    if (u.shadow) {
        glm::vec2 suv;
        float sz;
        if (shadow_uvz_from_world(u.light_vp, in.world_pos, suv, sz)) {

            // slope-scaled bias: N·L бага үед bias нэмэгдэнэ
            float slope = 1.0f - glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
            float bias  = SHADOW_BIAS_BASE + SHADOW_BIAS_SLOPE * slope;

            shadow = pcss_shadow_factor(*u.shadow, suv, sz, bias, px, py);
        }
    }

    glm::vec3 amb    = ambientStrength * baseColor;
    glm::vec3 direct = shadow * (diffuse * baseColor + specular);

    glm::vec3 result = glm::clamp(amb + direct, 0.0f, 1.0f);
    return rgb01_to_color(result);
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

        floor  = new FloorPlane(55.0f, 140.0f);
        car    = new SubaruObject(glm::vec3(-6.0f, 0.0f, 26.0f), glm::vec3(0.08f), car_tex);
        monkey = new MonkeyObject(glm::vec3(-6.0f, 12.2f, 26.0f), glm::vec3(1.65f));

        scene_objects.push_back(car);
        scene_objects.push_back(monkey);
    }

    ~DemoScene()
    {
        for (auto* o : scene_objects) delete o;
        delete floor;
    }

    void process() override {}

    shs::Canvas*  canvas;
    Viewer*       viewer;

    FloorPlane*   floor;
    SubaruObject* car;
    MonkeyObject* monkey;

    std::vector<shs::AbstractObject3D*> scene_objects;
};

// ==========================================
// RENDERER SYSTEM (Shadow + Camera)
// ==========================================
class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(DemoScene* scene, shs::Job::ThreadedPriorityJobSystem* job_sys)
        : scene(scene), job_system(job_sys)
    {
        color  = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
        depth  = new shs::ZBuffer(CANVAS_WIDTH, CANVAS_HEIGHT, scene->viewer->camera->z_near, scene->viewer->camera->z_far);
        shadow = new ShadowMap(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);
    }

    ~RendererSystem()
    {
        delete color;
        delete depth;
        delete shadow;
    }

    void process(float dt) override
    {
        (void)dt;

        glm::mat4 view = scene->viewer->camera->view_matrix;
        glm::mat4 proj = scene->viewer->camera->projection_matrix;

        // Light VP (Directional, Ortho)
        glm::vec3 center(0.0f, 6.0f, 45.0f);
        glm::vec3 light_dir = LIGHT_DIR_WORLD;
        glm::vec3 light_pos = center - light_dir * 80.0f;

        glm::mat4 light_view = glm::lookAtLH(light_pos, center, glm::vec3(0,1,0));

        float L = -85.0f, R = 85.0f;
        float B = -55.0f, T = 95.0f;
        float zn = 0.1f, zf = 240.0f;

        glm::mat4 light_proj = ortho_lh_zo(L, R, B, T, zn, zf);
        glm::mat4 light_vp = light_proj * light_view;

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
                    job_system->submit({[=, this, light_vp]() {

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
        // PASS1: Camera render
        // -----------------------
        color->buffer().clear(shs::Color{20,20,25,255});
        depth->clear();

        {
            int W = color->get_width();
            int H = color->get_height();

            int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
            int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

            wg_cam.reset();

            for (int ty = 0; ty < rows; ty++) {
                for (int tx = 0; tx < cols; tx++) {

                    wg_cam.add(1);
                    job_system->submit({[=, this, view, proj, light_vp]() {

                        glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                        glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                         std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                        // Floor draw
                        {
                            Uniforms u;
                            u.model           = glm::mat4(1.0f);
                            u.view            = view;

                            u.mv              = u.view * u.model;
                            u.mvp             = proj   * u.mv;
                            u.normal_mat      = glm::mat3(1.0f);

                            u.light_vp        = light_vp;
                            u.light_dir_world = LIGHT_DIR_WORLD;
                            u.camera_pos      = scene->viewer->position;
                            u.base_color      = shs::Color{ 120, 122, 128, 255 };
                            u.albedo          = nullptr;
                            u.use_texture     = false;
                            u.shadow          = shadow;

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

                                draw_triangle_tile_color_depth_softshadow(
                                    *color,
                                    *depth,
                                    tri_v, tri_n, tri_uv,
                                    [&u](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
                                        return vertex_shader_full(p, n, uv, u);
                                    },
                                    [&u](const VaryingsFull& vin, int px, int py) {
                                        return fragment_shader_softshadow(vin, u, px, py);
                                    },
                                    t_min, t_max
                                );
                            }
                        }

                        // Objects draw
                        for (shs::AbstractObject3D* obj : scene->scene_objects)
                        {
                            // Subaru
                            if (auto* car = dynamic_cast<SubaruObject*>(obj))
                            {
                                glm::mat4 model = car->get_world_matrix();

                                Uniforms u;
                                u.model           = model;
                                u.view            = view;
                                u.mv              = u.view * u.model;
                                u.mvp             = proj * u.mv;
                                u.normal_mat      = glm::transpose(glm::inverse(glm::mat3(u.model)));
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.camera_pos      = scene->viewer->position;
                                u.base_color      = shs::Color{200,200,200,255};
                                u.albedo          = car->albedo;
                                u.use_texture     = (car->albedo && car->albedo->valid());
                                u.shadow          = shadow;

                                const auto& v = car->geometry->triangles;
                                const auto& n = car->geometry->normals;
                                const auto& t = car->geometry->uvs;

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { t[i], t[i+1], t[i+2] };

                                    draw_triangle_tile_color_depth_softshadow(
                                        *color,
                                        *depth,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin, int px, int py) {
                                            return fragment_shader_softshadow(vin, u, px, py);
                                        },
                                        t_min, t_max
                                    );
                                }
                            }

                            // Monkey
                            if (auto* mk = dynamic_cast<MonkeyObject*>(obj))
                            {
                                glm::mat4 model = mk->get_world_matrix();

                                Uniforms u;
                                u.model           = model;
                                u.view            = view;
                                u.mv              = u.view * u.model;
                                u.mvp             = proj   * u.mv;
                                u.normal_mat      = glm::transpose(glm::inverse(glm::mat3(u.model)));
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.camera_pos      = scene->viewer->position;
                                u.base_color      = shs::Color{ 180, 150, 95, 255 };
                                u.albedo          = nullptr;
                                u.use_texture     = false;
                                u.shadow          = shadow;

                                const auto& v = mk->geometry->triangles;
                                const auto& n = mk->geometry->normals;
                                static const glm::vec2 uv0(0.0f);

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { uv0, uv0, uv0 };

                                    draw_triangle_tile_color_depth_softshadow(
                                        *color,
                                        *depth,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin, int px, int py) {
                                            return fragment_shader_softshadow(vin, u, px, py);
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
    }

    shs::Canvas& output() { return *color; }

private:
    DemoScene* scene;
    shs::Job::ThreadedPriorityJobSystem* job_system;

    shs::Canvas*  color;
    shs::ZBuffer* depth;
    ShadowMap*    shadow;

    shs::Job::WaitGroup wg_shadow;
    shs::Job::WaitGroup wg_cam;
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
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    IMG_Init(IMG_INIT_PNG | IMG_INIT_JPG);

    auto* job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    // Present canvas
    shs::Canvas* screen_canvas  = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    SDL_Surface* screen_surface = screen_canvas->create_sdl_surface();
    SDL_Texture* screen_texture = SDL_CreateTextureFromSurface(renderer, screen_surface);

    // Texture load (Subaru albedo)
    shs::Texture2D car_tex = shs::load_texture_sdl_image("./obj/subaru/SUBARU1_M.bmp", true);

    // Scene
    Viewer*    viewer    = new Viewer(glm::vec3(0.0f, 10.0f, -42.0f), 55.0f);
    DemoScene* scene     = new DemoScene(screen_canvas, viewer, &car_tex);

    SystemProcessor* sys = new SystemProcessor(scene, job_system);

    bool exit = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();
    bool is_dragging = false;
    int   frames     = 0;
    float fps_timer  = 0.0f;

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

        // Logic + render
        sys->process(dt);
        sys->render(dt);

        // Present
        screen_canvas->buffer() = sys->output().buffer();
        shs::Canvas::copy_to_SDLSurface(screen_surface, screen_canvas);

        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        frames++;
        fps_timer += dt;
        if (fps_timer >= 1.0f) {
            std::string title =
                "PCSS Soft Shadows | FPS: " + std::to_string(frames) +
                " | Threads: " + std::to_string(THREAD_COUNT) +
                " | ShadowMap: " + std::to_string(SHADOW_MAP_SIZE) +
                " | Canvas: " + std::to_string(CANVAS_WIDTH) + "x" + std::to_string(CANVAS_HEIGHT);
            SDL_SetWindowTitle(window, title.c_str());
            frames = 0;
            fps_timer = 0.0f;
        }
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
