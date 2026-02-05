/*

IBL + Shadow Mapping + Motion Blur + Blinn-Phong + XSIMD (PASS0/PASS2)

Координат систем: 
    LH (+Z forward, +Y up, +X right)
Screen: 
    (0,0) top-left, +Y down
Canvas: 
    (0,0) bottom-left, +Y up


Зорилго:
- CPU Software Renderer дээр:
  1) Skybox background
  2) Shadow mapping (Directional light, Ortho)
  3) Direct lighting (Blinn-Phong) + Shadow
  4) IBL lighting (Diffuse Irradiance + Prefiltered Specular)
  5) Post-process Motion Blur (object + camera velocity)
- Performance оновчлох:
  - PASS0 ShadowMap depth raster дээр XSIMD (edge функц + SIMD depth test/store)
  - PASS2 Motion blur дээр raw u32 pack/unpack (memory overhead багасгах)


Rendering pipeline:
PASS 0 : ShadowMap Depth (Directional light, Ortho, z in [0..1])
        - Triangle raster -> contiguous depth buffer
        - XSIMD: multiple pixels per iteration (compare/select/store)
PASS 1 : Camera Render (Color + Depth(view_z) + Motion(full))
         1) Skybox fill (background)
         2) Geometry raster:
            - Z test (canvas coords)
            - Motion vector write (canvas coords)
            - Shading:
              a) Direct Blinn-Phong * Shadow
              b) IBL:
                 - Diffuse = baseColor * irradiance(N)
                 - Specular = prefilteredEnv(R, roughness->LOD) * Fresnel
              c) Shadow зөвхөн direct дээр үйлчилнэ (IBL дээр shadow тавихгүй)
PASS 2 : Combined Motion Blur (full-screen post)
         - v_full  = object + camera (from raster)
         - v_cam   = depth + matrices дээрээс reconstruction
         - v_obj   = v_full - v_cam
         - v_total = w_obj*v_obj + w_cam*v_cam
         - raw u32 sample accumulation (triangle weights)


IBL ARCHITECTURE (програм эхлэхэд нэг удаа precompute):
[1] LDR Skybox Cubemap -> float RGB01 CubemapF
[2] Diffuse Irradiance Cubemap:
    - Cosine-weighted hemisphere integration per texel (low-res)
[3] Specular Prefiltered Cubemap (Mip chain):
    - Roughness бүрээр Phong-lobe sampling
    - Mip level ~= roughness
    - Trilinear mip sampling runtime дээр
[4] Fresnel (Schlick):
    - ks = F, kd = 1-F
    - Diffuse/Specluar energy split


- Canvas/ZBuffer/MotionBuffer бүгд CANVAS coords (y up) дээр хадгалагдана.
- Screen-space raster (y down) -> Canvas руу бичихдээ y flip хийнэ.
- Shadow map UV нь (0,0) top-left, y down гэсэн convention ашиглана.
- Precompute дээр env face size их байвал хэт удаан болохоос хамгаалж:
  IBL_SPEC_BASE_CAP ашиглаж base resolution-г cap хийнэ.


*/

#define SDL_MAIN_HANDLED

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

#include <xsimd/xsimd.hpp>

#include "shs_renderer.hpp"

//#define WINDOW_WIDTH      800
//#define WINDOW_HEIGHT     600
#define WINDOW_WIDTH      1200
#define WINDOW_HEIGHT     900
#define CANVAS_WIDTH      1200
#define CANVAS_HEIGHT     900

#define MOUSE_SENSITIVITY 0.2f

#define THREAD_COUNT      20
#define TILE_SIZE_X       160
#define TILE_SIZE_Y       160

// ------------------------------------------
// SHADOW MAP CONFIG
// ------------------------------------------
#define SHADOW_MAP_SIZE   2048

static const glm::vec3 LIGHT_DIR_WORLD = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f));

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
// IBL PRECOMPUTE CONFIG (startup нэг удаа)
// ------------------------------------------
static const int   IBL_IRR_SIZE      = 16;   // diffuse irradiance cubemap size
static const int   IBL_IRR_SAMPLES   = 64;   // irradiance sample/texel

static const int   IBL_SPEC_MIPCOUNT = 6;    // spec mip тоо (0..m-1)
static const int   IBL_SPEC_SAMPLES  = 16;   // mip бүрийн texel тутмын sample
static const int   IBL_SPEC_BASE_CAP = 256;  // env face size-г дээд тал нь 256 (freeze хамгаалалт)

// ==========================================
// HELPERS
// ==========================================

// Helpers are now used from shs::Math and shs namespace in shs_renderer.hpp.

// ------------------------------------------
// Color pack/unpack (little-endian x86 дээр хурдан)
// ------------------------------------------
static inline uint32_t pack_rgba_u32(const shs::Color& c)
{
    return (uint32_t)c.r | ((uint32_t)c.g << 8) | ((uint32_t)c.b << 16) | ((uint32_t)c.a << 24);
}

static inline shs::Color unpack_rgba_u32(uint32_t u)
{
    shs::Color c;
    c.r = (uint8_t)(u & 0xFF);
    c.g = (uint8_t)((u >> 8) & 0xFF);
    c.b = (uint8_t)((u >> 16) & 0xFF);
    c.a = (uint8_t)((u >> 24) & 0xFF);
    return c;
}

// ==========================================
// LH Ortho matrix (NDC z: 0..1)
// ==========================================

// shs::Math::ortho_lh_zo is now shs::Math::ortho_lh_zo

// ==========================================
// TEXTURE SAMPLER (nearest)
// ==========================================

// shs::sample_nearest logic is now standardized.

// ==========================================
// SKYBOX USE SHARED shs::CubeMap
// ==========================================

// (sample_cubemap_nearest_rgb01 removed)

// ==========================================
// IBL FLOAT CUBEMAP + BILINEAR SAMPLING + PRECOMPUTE
// ==========================================

struct CubeMapF
{
    int size = 0; // square face
    std::vector<glm::vec3> face[6]; // rgb in [0,1]

    inline bool valid() const {
        if (size <= 0) return false;
        for (int i=0;i<6;i++) if ((int)face[i].size() != size*size) return false;
        return true;
    }
    inline const glm::vec3& at(int f, int x, int y) const {
        return face[f][(size_t)y * (size_t)size + (size_t)x];
    }
};

static inline CubeMapF cubemap_to_float_rgb01(const shs::CubeMap& cm)
{
    CubeMapF out;
    if (!cm.valid()) return out;

    out.size = cm.face[0].w;

    for (int f=0; f<6; ++f) {
        out.face[f].resize((size_t)out.size * (size_t)out.size);
        for (int y=0; y<out.size; ++y) {
            for (int x=0; x<out.size; ++x) {
                shs::Color c = cm.face[f].texels.at(x, y);
                out.face[f][(size_t)y*out.size + (size_t)x] = shs::color_to_rgb01(c);
            }
        }
    }
    return out;
}

static inline glm::vec3 sample_face_bilinear(const CubeMapF& cm, int face, float u, float v)
{
    u = shs::Math::saturate(u);
    v = shs::Math::saturate(v);

    float fx = u * float(cm.size - 1);
    float fy = v * float(cm.size - 1);

    int x0 = shs::Math::clamp((int)std::floor(fx), 0, cm.size - 1);
    int y0 = shs::Math::clamp((int)std::floor(fy), 0, cm.size - 1);
    int x1 = shs::Math::clamp(x0 + 1, 0, cm.size - 1);
    int y1 = shs::Math::clamp(y0 + 1, 0, cm.size - 1);

    float tx = fx - float(x0);
    float ty = fy - float(y0);

    const glm::vec3& c00 = cm.at(face, x0, y0);
    const glm::vec3& c10 = cm.at(face, x1, y0);
    const glm::vec3& c01 = cm.at(face, x0, y1);
    const glm::vec3& c11 = cm.at(face, x1, y1);

    glm::vec3 cx0 = glm::mix(c00, c10, tx);
    glm::vec3 cx1 = glm::mix(c01, c11, tx);
    return glm::mix(cx0, cx1, ty);
}

// dir_world -> cubemap (face + uv) (bilinear)
static inline glm::vec3 sample_cubemap_bilinear_rgb01(const CubeMapF& cm, const glm::vec3& dir_world)
{
    if (!cm.valid()) return glm::vec3(0.0f);

    glm::vec3 d = dir_world;
    float len = glm::length(d);
    if (len < 1e-8f) return glm::vec3(0.0f);
    d /= len;

    float ax = std::abs(d.x);
    float ay = std::abs(d.y);
    float az = std::abs(d.z);

    int face = 0;
    float u = 0.5f, v = 0.5f;

    if (ax >= ay && ax >= az) {
        if (d.x > 0.0f) { face = 0; u = (-d.z / ax); v = ( d.y / ax); }
        else            { face = 1; u = ( d.z / ax); v = ( d.y / ax); }
    } else if (ay >= ax && ay >= az) {
        if (d.y > 0.0f) { face = 2; u = ( d.x / ay); v = (-d.z / ay); }
        else            { face = 3; u = ( d.x / ay); v = ( d.z / ay); }
    } else {
        if (d.z > 0.0f) { face = 4; u = ( d.x / az); v = ( d.y / az); }
        else            { face = 5; u = (-d.x / az); v = ( d.y / az); }
    }

    u = 0.5f * (u + 1.0f);
    v = 0.5f * (v + 1.0f);

    return sample_face_bilinear(cm, face, u, v);
}

static inline glm::vec3 cubemap_dir_from_face_uv(int face, float u, float v)
{
    // u,v [0,1] -> a,b [-1,1]
    float a = 2.0f * u - 1.0f;
    float b = 2.0f * v - 1.0f;

    glm::vec3 d(0.0f);

    // sample mapping-тэй яг тааруулах ёстой
    switch(face) {
        case 0:  d = glm::vec3( 1.0f,  b, -a); break; // +X
        case 1:  d = glm::vec3(-1.0f,  b,  a); break; // -X
        case 2:  d = glm::vec3( a,  1.0f, -b); break; // +Y
        case 3:  d = glm::vec3( a, -1.0f,  b); break; // -Y
        case 4:  d = glm::vec3( a,  b,  1.0f); break; // +Z
        case 5:  d = glm::vec3(-a,  b, -1.0f); break; // -Z
        default: d = glm::vec3(0,0,1); break;
    }
    return glm::normalize(d);
}

// Cosine-weighted hemisphere sample (tangent space, +Z axis)
static inline glm::vec3 cosine_sample_hemisphere(float u1, float u2)
{
    float r   = std::sqrt(u1);
    float phi = 6.2831853f * u2;
    float x   = r * std::cos(phi);
    float y   = r * std::sin(phi);
    float z   = std::sqrt(std::max(0.0f, 1.0f - u1));
    return glm::vec3(x, y, z);
}

static inline void tangent_basis(const glm::vec3& N, glm::vec3& T, glm::vec3& B)
{
    // Тогтвортой tangent basis
    glm::vec3 up = (std::abs(N.y) < 0.999f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    T            = glm::normalize(glm::cross(up, N));
    B            = glm::cross(N, T);
}

static inline CubeMapF build_irradiance_cubemap(const CubeMapF& env, int outSize, int sampleCount)
{
    CubeMapF irr;
    irr.size = outSize;
    for (int f=0; f<6; ++f) irr.face[f].assign((size_t)outSize*(size_t)outSize, glm::vec3(0.0f));

    for (int f=0; f<6; ++f) {
        for (int y=0; y<outSize; ++y) {
            for (int x=0; x<outSize; ++x) {

                float u = (float(x) + 0.5f) / float(outSize);
                float v = (float(y) + 0.5f) / float(outSize);

                glm::vec3 N = cubemap_dir_from_face_uv(f, u, v);
                glm::vec3 T,B;
                tangent_basis(N, T, B);

                glm::vec3 sum(0.0f);

                // deterministic random: хурдан + тогтвортой
                uint32_t seed = (uint32_t)(f*73856093u ^ x*19349663u ^ y*83492791u);
                auto rnd01 = [&seed]() {
                    seed = 1664525u * seed + 1013904223u;
                    return float(seed & 0x00FFFFFFu) / float(0x01000000u);
                };

                for (int i=0; i<sampleCount; ++i) {
                    float r1 = rnd01();
                    float r2 = rnd01();

                    glm::vec3 h = cosine_sample_hemisphere(r1, r2); // tangent-space
                    glm::vec3 L = glm::normalize(T*h.x + B*h.y + N*h.z);

                    sum += sample_cubemap_bilinear_rgb01(env, L);
                }

                irr.face[f][(size_t)y*outSize + (size_t)x] = sum / float(sampleCount);
            }
        }
    }

    return irr;
}

// Roughness -> Phong exponent (simple mapping)
static inline float roughness_to_phong_exp(float rough)
{
    rough     = shs::Math::saturate(rough);
    float r2  = std::max(1e-4f, rough*rough);
    float exp = (2.0f / r2) - 2.0f;
    return std::max(1.0f, exp);
}

// Phong lobe sample around +Z (tangent space)
static inline glm::vec3 phong_lobe_sample(float u1, float u2, float exp)
{
    float phi  = 6.2831853f * u1;
    float cosT = std::pow(1.0f - u2, 1.0f / (exp + 1.0f));
    float sinT = std::sqrt(std::max(0.0f, 1.0f - cosT*cosT));
    return glm::vec3(std::cos(phi)*sinT, std::sin(phi)*sinT, cosT);
}

struct PrefilteredSpec
{
    std::vector<CubeMapF> mip; // mip[0]=sharp, mip[last]=blur
    inline bool valid() const { return !mip.empty() && mip[0].valid(); }
    inline int  maxMip() const { return (int)mip.size(); }
};

static inline PrefilteredSpec build_prefiltered_spec(const CubeMapF& env, int baseSize, int mipCount, int samplesPerTexel)
{
    PrefilteredSpec out;
    out.mip.resize(mipCount);

    for (int m=0; m<mipCount; ++m) {
        int sz = std::max(1, baseSize >> m);

        std::cout << "STATUS :   IBL spec mip " << m << "/" << (mipCount - 1)
                  << " | size=" << sz << " | samples=" << samplesPerTexel << "\n";

        out.mip[m].size = sz;
        for (int f=0; f<6; ++f) out.mip[m].face[f].assign((size_t)sz*(size_t)sz, glm::vec3(0.0f));

        float rough = float(m) / float(std::max(1, mipCount - 1));
        float exp   = roughness_to_phong_exp(rough);

        for (int f=0; f<6; ++f) {
            for (int y=0; y<sz; ++y) {
                for (int x=0; x<sz; ++x) {

                    float u = (float(x) + 0.5f) / float(sz);
                    float v = (float(y) + 0.5f) / float(sz);

                    glm::vec3 R = cubemap_dir_from_face_uv(f, u, v); // reflection dir
                    glm::vec3 T,B;
                    tangent_basis(R, T, B);

                    glm::vec3 sum(0.0f);

                    uint32_t seed = (uint32_t)(m*2654435761u ^ f*97531u ^ x*31337u ^ y*1337u);
                    auto rnd01 = [&seed]() {
                        seed = 1664525u * seed + 1013904223u;
                        return float(seed & 0x00FFFFFFu) / float(0x01000000u);
                    };

                    for (int i=0; i<samplesPerTexel; ++i) {
                        float r1 = rnd01();
                        float r2 = rnd01();

                        glm::vec3 s = phong_lobe_sample(r1, r2, exp);
                        glm::vec3 L = glm::normalize(T*s.x + B*s.y + R*s.z);

                        sum += sample_cubemap_bilinear_rgb01(env, L);
                    }

                    out.mip[m].face[f][(size_t)y*sz + (size_t)x] = sum / float(samplesPerTexel);
                }
            }
        }
    }

    return out;
}

// Specular cubemap: trilinear mip sampling (m0,m1 mix)
static inline glm::vec3 sample_cubemap_spec_trilinear(const PrefilteredSpec& ps, const glm::vec3& dir, float lod)
{
    if (!ps.valid()) return glm::vec3(0.0f);

    float mmax = float(ps.maxMip() - 1);
    lod        = shs::Math::clampf(lod, 0.0f, mmax);

    int   m0 = (int)std::floor(lod);
    int   m1 = std::min(m0 + 1, ps.maxMip() - 1);
    float t  = lod - float(m0);

    glm::vec3 c0 = sample_cubemap_bilinear_rgb01(ps.mip[m0], dir);
    glm::vec3 c1 = sample_cubemap_bilinear_rgb01(ps.mip[m1], dir);
    return glm::mix(c0, c1, t);
}

struct IBLResources
{
    CubeMapF env;        // float env
    CubeMapF irradiance; // diffuse convolved
    PrefilteredSpec spec;// prefiltered spec mips

    inline bool valid() const { return env.valid() && irradiance.valid() && spec.valid(); }
};

// ==========================================
// SHADOW MAP BUFFER (Depth only)
// ==========================================

// Using standardized shs::ShadowMap, shs::MotionBuffer, shs::RT_ColorDepthVelocity
//using ShadowMap           = shs::ShadowMap;
//using MotionBuffer        = shs::MotionBuffer;
//using RT_ColorDepthMotion = shs::RT_ColorDepthVelocity;

// ==========================================
// CAMERA + VIEWER
// ==========================================

// Using standardized shs::Viewer and shs::ModelGeometry
//using Viewer        = shs::Viewer;
//using ModelGeometry = shs::ModelGeometry;

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
        this->geometry       = new shs::ModelGeometry("./obj/subaru/SUBARU_1.obj");
        this->rotation_angle = 0.0f;
        this->albedo         = albedo;
        this->has_prev_mvp   = false;
        this->prev_mvp       = glm::mat4(1.0f);
    }
    ~SubaruObject() { delete geometry; }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), position);
        // удаан clockwise (LH-д +Y тэнхлэгээр эргүүлнэ)
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float dt) override
    {
        rotation_angle += 12.0f * dt; // clockwise
        if (rotation_angle >= 360.0f) rotation_angle -= 360.0f;
    }

    void render() override {}

    shs::ModelGeometry        *geometry;
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
        this->geometry           = new shs::ModelGeometry("./obj/monkey/monkey.rawobj");

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

        this->has_prev_mvp       = false;
        this->prev_mvp           = glm::mat4(1.0f);
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

        float w     = wobble_phase_speed * wobble_hz;

        position    = base_position;
        position.y += std::sin(time_accum * w) * wobble_amp_y;

        position.x += std::cos(time_accum * w * 1.15f) * wobble_amp_xz;
        position.z += std::sin(time_accum * w * 0.95f) * wobble_amp_xz;

        rotation_angle += spin_deg_per_sec * dt;
        if (rotation_angle > 360.0f) rotation_angle -= 360.0f;
    }

    void render() override {}

    shs::ModelGeometry *geometry;

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

    bool      has_prev_mvp;
    glm::mat4 prev_mvp;
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
// UNIFORMS & VARYINGS
// ==========================================

struct Uniforms
{
    glm::mat4 mvp;
    glm::mat4 prev_mvp;
    glm::mat4 model;
    glm::mat4 view;

    // per-object урьдчилж бодох
    glm::mat4 mv;
    glm::mat3 normal_mat;

    glm::mat4 light_vp;

    glm::vec3 light_dir_world;
    glm::vec3 camera_pos;

    shs::Color base_color;
    const shs::Texture2D *albedo = nullptr;
    bool use_texture = false;

    const shs::ShadowMap *shadow = nullptr;

    // Skybox (background)
    const shs::AbstractSky *sky = nullptr;

    // IBL (irradiance + prefiltered spec)
    const IBLResources *ibl = nullptr;

    // IBL knobs
    float ibl_ambient  = 0.30f;
    float ibl_refl     = 0.35f;
    float ibl_f0       = 0.04f;
    float ibl_refl_mix = 1.0f;

    // Blinn-Phong shininess -> roughness mapping
    float shininess = 64.0f;
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

    out.position       = u.mvp * glm::vec4(aPos, 1.0f);
    out.prev_position  = u.prev_mvp * glm::vec4(aPos, 1.0f);

    glm::vec4 world_h  = u.model * glm::vec4(aPos, 1.0f);
    out.world_pos      = glm::vec3(world_h);

    out.normal         = glm::normalize(u.normal_mat * aNormal);
    out.uv             = aUV;

    glm::vec4 view_pos = u.mv * glm::vec4(aPos, 1.0f);
    out.view_z         = view_pos.z;

    return out;
}

// ==========================================
// SHADOW HELPERS
// ==========================================

static inline bool shadow_uvz_from_world(
    const glm::mat4& light_vp,
    const glm::vec3& world_pos,
    glm::vec2& out_uv,
    float& out_z_ndc)
{
    glm::vec4 clip = light_vp * glm::vec4(world_pos, 1.0f);
    if (std::abs(clip.w) < 1e-6f) return false;

    glm::vec3 ndc = glm::vec3(clip) / clip.w; // x,y in [-1,1], z in [0,1]
    out_z_ndc = ndc.z;

    if (out_z_ndc < 0.0f || out_z_ndc > 1.0f) return false;

    // ndc -> uv (shadow map: (0,0) top-left, y down)
    out_uv.x = ndc.x * 0.5f + 0.5f;
    out_uv.y = 1.0f - (ndc.y * 0.5f + 0.5f);

    return true;
}

static inline float shadow_compare(
    const shs::ShadowMap& sm,
    glm::vec2 uv,
    float z_ndc,
    float bias)
{
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;

    int x = (int)std::lround(uv.x * float(sm.w - 1));
    int y = (int)std::lround(uv.y * float(sm.h - 1));

    float d = sm.sample(x, y);
    if (d == std::numeric_limits<float>::max()) return 1.0f;

    return (z_ndc <= d + bias) ? 1.0f : 0.0f;
}

static inline float shadow_factor_pcf_2x2(
    const shs::ShadowMap& sm,
    glm::vec2 uv,
    float z_ndc,
    float bias)
{
    if (!SHADOW_USE_PCF) return shadow_compare(sm, uv, z_ndc, bias);

    float fx = uv.x * float(sm.w - 1);
    float fy = uv.y * float(sm.h - 1);

    int x0 = shs::Math::clamp((int)std::floor(fx), 0, sm.w - 1);
    int y0 = shs::Math::clamp((int)std::floor(fy), 0, sm.h - 1);
    int x1 = shs::Math::clamp(x0 + 1, 0, sm.w - 1);
    int y1 = shs::Math::clamp(y0 + 1, 0, sm.h - 1);

    float s00 = (z_ndc <= sm.sample(x0,y0) + bias) ? 1.0f : 0.0f;
    float s10 = (z_ndc <= sm.sample(x1,y0) + bias) ? 1.0f : 0.0f;
    float s01 = (z_ndc <= sm.sample(x0,y1) + bias) ? 1.0f : 0.0f;
    float s11 = (z_ndc <= sm.sample(x1,y1) + bias) ? 1.0f : 0.0f;

    return 0.25f * (s00 + s10 + s01 + s11);
}

// ==========================================
// FRAGMENT SHADER
// - Direct: Blinn-Phong + Shadow
// - IBL   : Diffuse irradiance + Prefiltered specular (mip chain) + Schlick split
// ==========================================

static shs::Color fragment_shader_full(const VaryingsFull& in, const Uniforms& u)
{
    glm::vec3 N = glm::normalize(in.normal);
    glm::vec3 L = glm::normalize(-u.light_dir_world);
    glm::vec3 V = glm::normalize(u.camera_pos - in.world_pos);

    // -----------------------------
    // Direct light (Blinn-Phong)
    // -----------------------------
    float ambientStrength = 0.18f;

    float diff        = glm::max(glm::dot(N, L), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f);

    glm::vec3 H = glm::normalize(L + V);
    float specularStrength = 0.45f;

    float shininess = u.shininess;
    float spec      = glm::pow(glm::max(glm::dot(N, H), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f);

    // BaseColor
    glm::vec3 baseColor;
    if (u.use_texture && u.albedo && u.albedo->valid()) {
        shs::Color tc = shs::sample_nearest(*u.albedo, in.uv);
        baseColor     = shs::color_to_rgb01(tc);
    } else {
        baseColor     = shs::color_to_rgb01(u.base_color);
    }

    // Shadow factor (1=lit, 0=shadow) — зөвхөн direct дээр үйлчилнэ
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

    // -----------------------------
    // IBL (Diffuse irradiance + Prefiltered specular)
    // -----------------------------
    glm::vec3 ibl_diffuse(0.0f);
    glm::vec3 ibl_spec(0.0f);

    if (u.ibl && u.ibl->valid()) {

        // 1) Diffuse irradiance: N чиглэлээр sample
        glm::vec3 irr = sample_cubemap_bilinear_rgb01(u.ibl->irradiance, N);

        // 2) Specular prefiltered: R чиглэл + roughness -> LOD
        float rough = std::sqrt(2.0f / (shininess + 2.0f)); // 0..1
        float lod   = rough * float(u.ibl->spec.maxMip() - 1);

        glm::vec3 R = glm::reflect(-V, N);
        glm::vec3 prefiltered = sample_cubemap_spec_trilinear(u.ibl->spec, R, lod);

        // Fresnel (Schlick)
        float NoV = glm::max(0.0f, glm::dot(N, V));
        float F   = shs::Math::schlick_fresnel(u.ibl_f0, NoV);

        // Энерги split (энгийн боловч тогтвортой)
        float ks = F;
        float kd = 1.0f - ks;

        ibl_diffuse = kd * irr * shs::Math::saturate(u.ibl_ambient);
        ibl_spec    = ks * prefiltered * shs::Math::saturate(u.ibl_refl) * shs::Math::saturate(u.ibl_refl_mix);
    }

    // -----------------------------
    // Final combine
    // -----------------------------
    glm::vec3 direct = shadow * (diffuse * baseColor + specular);
    glm::vec3 amb    = ambientStrength * baseColor; // жижиг legacy ambient
    glm::vec3 result = amb + direct + (ibl_diffuse * baseColor) + ibl_spec;

    result = glm::clamp(result, 0.0f, 1.0f);
    return shs::rgb01_to_color(result);
}

// ==========================================
// SKYBOX BACKGROUND PASS (rt.color fill)
// ==========================================

static void skybox_background_pass(
    shs::Canvas& dst,
    const shs::AbstractSky& sky,
    const shs::Camera3D& cam,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    // if (!sky.valid()) return; 

    int W = dst.get_width();
    int H = dst.get_height();

    float aspect = float(W) / float(H);
    float tan_half_fov = std::tan(glm::radians(cam.field_of_view) * 0.5f);

    glm::vec3 forward = glm::normalize(cam.direction_vector);
    glm::vec3 right   = glm::normalize(cam.right_vector);
    glm::vec3 up      = glm::normalize(cam.up_vector);

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wg.reset();

    // raw buffer дээр бичнэ (Canvas y-up)
    shs::Color* dst_raw = dst.buffer().raw();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wg.add(1);
            //job_system->submit({[=, &wg, &sky, forward, right, up, aspect, tan_half_fov]() {
            job_system->submit({[=, &wg, &sky]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                for (int y = y0; y < y1; y++) {
                    int row_off = y * W;
                    for (int x = x0; x < x1; x++) {

                        float fx = (float(x) + 0.5f) / float(W);
                        float fy = (float(y) + 0.5f) / float(H);

                        float ndc_x = fx * 2.0f - 1.0f;
                        float ndc_y = fy * 2.0f - 1.0f;

                        glm::vec3 dir =
                            forward +
                            right * (ndc_x * aspect * tan_half_fov) +
                            up    * (ndc_y * tan_half_fov);

                        dir = glm::normalize(dir);

                        // sample returns Linear
                        glm::vec3 c_lin = sky.sample(dir);
                        c_lin = glm::clamp(c_lin, 0.0f, 1.0f);
                        glm::vec3 c_srgb = shs::linear_to_srgb(c_lin);
                        dst_raw[row_off + x] = shs::rgb01_to_color(c_srgb);
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
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
// SHADOW MAP RASTER (XSIMD - edge function + contiguous depth)
// ==========================================

static inline glm::vec3 clip_to_shadow_screen(const glm::vec4& clip, int W, int H)
{
    glm::vec3 ndc = glm::vec3(clip) / clip.w; // x,y in [-1,1], z in [0,1]
    glm::vec3 s;
    s.x = (ndc.x * 0.5f + 0.5f) * float(W - 1);
    s.y = (1.0f - (ndc.y * 0.5f + 0.5f)) * float(H - 1); // (0,0) top-left
    s.z = ndc.z;
    return s;
}

struct ShadowTriProcessed
{
    // Edge функцүүд: E(x,y)=A*x + B*y + C
    float A0, B0, C0;
    float A1, B1, C1;
    float A2, B2, C2;

    float inv_area;

    // z interpolation (screen space)
    float z0, z1, z2;

    int min_x, min_y, max_x, max_y;
};

static inline bool build_shadow_tri(
    ShadowTriProcessed& out,
    const glm::vec3& s0,
    const glm::vec3& s1,
    const glm::vec3& s2,
    int W, int H,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    // Bounding box (tile-д clip)
    int min_x = std::max(tile_min.x, (int)std::floor(std::min({s0.x, s1.x, s2.x})));
    int min_y = std::max(tile_min.y, (int)std::floor(std::min({s0.y, s1.y, s2.y})));
    int max_x = std::min(tile_max.x, (int)std::ceil (std::max({s0.x, s1.x, s2.x})));
    int max_y = std::min(tile_max.y, (int)std::ceil (std::max({s0.y, s1.y, s2.y})));

    min_x = shs::Math::clamp(min_x, 0, W - 1);
    max_x = shs::Math::clamp(max_x, 0, W - 1);
    min_y = shs::Math::clamp(min_y, 0, H - 1);
    max_y = shs::Math::clamp(max_y, 0, H - 1);

    if (min_x > max_x || min_y > max_y) return false;

    // Screen-space signed area
    float area = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
    if (std::abs(area) < 1e-8f) return false;

    // area>0 гэсэн winding гэж үзээд inside тестийг (E>=0) ашиглана.
    if (area <= 0.0f) return false;

    out.inv_area = 1.0f / area;

    out.z0 = s0.z;
    out.z1 = s1.z;
    out.z2 = s2.z;

    out.min_x = min_x;
    out.min_y = min_y;
    out.max_x = max_x;
    out.max_y = max_y;

    // Edge функцүүдийг triangle (s0,s1,s2)-ийн дагуу байгуулна
    out.A0 = s0.y - s1.y; out.B0 = s1.x - s0.x; out.C0 = s0.x * s1.y - s0.y * s1.x;
    out.A1 = s1.y - s2.y; out.B1 = s2.x - s1.x; out.C1 = s1.x * s2.y - s1.y * s2.x;
    out.A2 = s2.y - s0.y; out.B2 = s0.x - s2.x; out.C2 = s2.x * s0.y - s2.y * s0.x;

    return true;
}

static void draw_triangle_tile_shadow_xsimd(
    shs::ShadowMap& sm,
    const glm::vec3 tri_verts[3],
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

    ShadowTriProcessed tri;
    if (!build_shadow_tri(tri, sc[0], sc[1], sc[2], sm.w, sm.h, tile_min, tile_max)) return;

    namespace xs = xsimd;
    using bf = xs::batch<float>;

    const int LANES = bf::size;

    // iota: [0.5, 1.5, 2.5, ...] (pixel center)
    static bf iota = [](){
        alignas(64) float tmp[64];
        for (int i = 0; i < 64; ++i) tmp[i] = float(i) + 0.5f;
        return bf::load_aligned(tmp);
    }();

    bf A0(tri.A0), B0(tri.B0), C0(tri.C0);
    bf A1(tri.A1), B1(tri.B1), C1(tri.C1);
    bf A2(tri.A2), B2(tri.B2), C2(tri.C2);

    bf invArea(tri.inv_area);

    bf z0(tri.z0), z1(tri.z1), z2(tri.z2);

    //float* zbuf = sm.depth.raw();
    float* zbuf = sm.depth().raw();
    const int W = sm.w;

    for (int y = tri.min_y; y <= tri.max_y; ++y)
    {
        bf yv(float(y) + 0.5f);

        // Row constants: B*y + C
        bf rowE0 = B0 * yv + C0;
        bf rowE1 = B1 * yv + C1;
        bf rowE2 = B2 * yv + C2;

        int row_off = y * W;

        int x = tri.min_x;

        // SIMD хэсэг
        for (; x + LANES - 1 <= tri.max_x; x += LANES)
        {
            bf xv(float(x) + 0.0f);
            bf xpix = xv + iota; // center

            bf E0 = A0 * xpix + rowE0;
            bf E1 = A1 * xpix + rowE1;
            bf E2 = A2 * xpix + rowE2;

            auto inside = (E0 >= 0.0f) & (E1 >= 0.0f) & (E2 >= 0.0f);
            if (!xs::any(inside)) continue;

            // Barycentric weights (edge mapping)
            bf w0 = E1 * invArea;
            bf w1 = E2 * invArea;
            bf w2 = E0 * invArea;

            bf z_new = w0 * z0 + w1 * z1 + w2 * z2;

            float* zptr = zbuf + row_off + x;

            bf z_old = bf::load_unaligned(zptr);

            // valid z range (0..1) check (cheap)
            auto inz = (z_new >= 0.0f) & (z_new <= 1.0f);

            auto pass = inside & inz & (z_new < z_old);
            if (!xs::any(pass)) continue;

            bf z_out = xs::select(pass, z_new, z_old);
            z_out.store_unaligned(zptr);
        }

        // Tail scalar
        for (; x <= tri.max_x; ++x)
        {
            float fx = float(x) + 0.5f;
            float fy = float(y) + 0.5f;

            float e0 = tri.A0 * fx + tri.B0 * fy + tri.C0;
            float e1 = tri.A1 * fx + tri.B1 * fy + tri.C1;
            float e2 = tri.A2 * fx + tri.B2 * fy + tri.C2;

            if (e0 >= 0.0f && e1 >= 0.0f && e2 >= 0.0f)
            {
                float w0s = e1 * tri.inv_area;
                float w1s = e2 * tri.inv_area;
                float w2s = e0 * tri.inv_area;

                float z = w0s * tri.z0 + w1s * tri.z1 + w2s * tri.z2;
                if (z < 0.0f || z > 1.0f) continue;

                int idx = row_off + x;
                if (z < zbuf[idx]) zbuf[idx] = z;
            }
        }
    }
}

// ==========================================
// CAMERA PASS RASTER HELPERS
// ==========================================

static inline glm::vec2 clip_to_screen_xy(const glm::vec4& clip, int w, int h)
{
    glm::vec3 s = shs::Canvas::clip_to_screen(clip, w, h); // screen (x right, y down)
    return glm::vec2(s.x, s.y);
}

// ======================================================
// CAMERA PASS RASTER: Color + Depth(view_z) + Motion(full) + Shadow
// ======================================================

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
            float az    = a.position.z;
            float bz    = b.position.z;
            float denom = (bz - az);
            float t     = (std::abs(denom) < 1e-8f) ? 0.0f : ((0.0f - az) / denom);
            t           = shs::Math::clampf(t, 0.0f, 1.0f);
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
// CAMERA-ONLY VELOCITY RECONSTRUCTION
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

static inline glm::vec2 compute_camera_velocity_canvas_fast(
    int x, int y,
    float view_z,
    int W, int H,
    const glm::mat4& curr_viewproj,
    const glm::mat4& prev_viewproj,
    const glm::mat4& inv_curr_viewproj,
    const glm::mat4& curr_proj)
{
    if (view_z == std::numeric_limits<float>::max()) return glm::vec2(0.0f);

    glm::vec2 ndc_xy = canvas_to_ndc_xy(x, y, W, H);
    float ndc_z      = viewz_to_ndcz(view_z, curr_proj);

    glm::vec4 clip_curr(ndc_xy.x, ndc_xy.y, ndc_z, 1.0f);

    glm::vec4 world_h = inv_curr_viewproj * clip_curr;
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

    float t       = (len - knee) / std::max(1e-6f, (max_len - knee));
    float t2      = t / (1.0f + t);
    float new_len = knee + (max_len - knee) * t2;

    return v * (new_len / len);
}

// ==========================================
// COMBINED MOTION BLUR PASS (PASS2) - raw u32 path + precompute weights
// ==========================================

struct MotionBlurKernel
{
    int samples = 0;
    float a[64];
    float w[64];

    MotionBlurKernel(int s=0) { init(s); }

    void init(int s)
    {
        samples = s;
        if (samples < 1) samples = 1;
        if (samples > 64) samples = 64;

        for (int i = 0; i < samples; ++i) {
            float t  = (samples == 1) ? 0.0f : (float(i) / float(samples - 1));
            float aa = (t - 0.5f) * 2.0f;       // -1..+1
            float ww = 1.0f - std::abs(aa);     // triangle weight
            a[i]     = aa;
            w[i]     = ww;
        }
    }
};

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

    glm::mat4 curr_vp     = curr_proj * curr_view;
    glm::mat4 prev_vp     = prev_proj * prev_view;
    glm::mat4 inv_curr_vp = glm::inverse(curr_vp);

    MotionBlurKernel kernel(samples);

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    // raw буферүүд
    const shs::Color* src_raw  = src.buffer().raw();
    shs::Color*       dst_raw  = dst.buffer().raw();
    const float*      z_raw    = depth.buffer().raw();
    //const glm::vec2*  v_raw    = v_full_buf.vel.data();
    //const glm::vec2*  v_raw    = v_full_buf.vel().data();
    const glm::vec2*  v_raw    = v_full_buf.raw();

    // u32 view (strict aliasing эрсдэлтэй, гэхдээ практик дээр OK)
    const uint32_t* src_u32 = reinterpret_cast<const uint32_t*>(src_raw);
    uint32_t*       dst_u32 = reinterpret_cast<uint32_t*>(dst_raw);

    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wg.add(1);
            job_system->submit({[=, &wg, &kernel]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                for (int y = y0; y < y1; y++) {
                    int row_off = y * W;

                    for (int x = x0; x < x1; x++) {

                        float vz = z_raw[row_off + x];

                        glm::vec2 v_cam = compute_camera_velocity_canvas_fast(
                            x, y, vz, W, H, curr_vp, prev_vp, inv_curr_vp, curr_proj
                        );

                        glm::vec2 v_full = v_raw[row_off + x];
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

                        if (len < 0.001f || kernel.samples <= 1) {
                            dst_u32[row_off + x] = src_u32[row_off + x];
                            continue;
                        }

                        glm::vec2 dir = v_total / len;

                        float r = 0.0f, g = 0.0f, b = 0.0f;
                        float wsum = 0.0f;

                        for (int i = 0; i < kernel.samples; i++) {

                            float a     = kernel.a[i];
                            float wgt   = kernel.w[i];

                            glm::vec2 p = glm::vec2(float(x), float(y)) + dir * (a * len);

                            int sx = shs::Math::clamp((int)std::lround(p.x), 0, W - 1);
                            int sy = shs::Math::clamp((int)std::lround(p.y), 0, H - 1);

                            uint32_t u = src_u32[sy * W + sx];

                            float cr = float( (u      ) & 0xFF );
                            float cg = float( (u >>  8) & 0xFF );
                            float cb = float( (u >> 16) & 0xFF );

                            r += wgt * cr;
                            g += wgt * cg;
                            b += wgt * cb;
                            wsum += wgt;
                        }

                        if (wsum < 1e-6f) wsum = 1.0f;

                        int rr = (int)(r / wsum);
                        int gg = (int)(g / wsum);
                        int bb = (int)(b / wsum);

                        rr = shs::Math::clamp(rr, 0, 255);
                        gg = shs::Math::clamp(gg, 0, 255);
                        bb = shs::Math::clamp(bb, 0, 255);

                        dst_u32[row_off + x] = (uint32_t)rr | ((uint32_t)gg << 8) | ((uint32_t)bb << 16) | (0xFFu << 24);
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

// ==========================================
// SCENE STATE
// ==========================================

class DemoScene : public shs::AbstractSceneState
{
public:
    DemoScene(shs::Canvas* canvas, shs::Viewer* viewer, const shs::Texture2D* car_tex, const shs::AbstractSky* sky, const IBLResources* ibl)
    {
        this->canvas = canvas;
        this->viewer = viewer;
        this->sky    = sky;
        this->ibl    = ibl;

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

    shs::Canvas* canvas;
    shs::Viewer*      viewer;

    const shs::AbstractSky* sky = nullptr;
    const IBLResources* ibl = nullptr;

    FloorPlane*   floor;

    SubaruObject* car;
    MonkeyObject* monkey;

    std::vector<shs::AbstractObject3D*> scene_objects;
};

// ==========================================
// RENDERER SYSTEM (Shadow + Camera + MotionBlur + Skybox + IBL)
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
            shs::Color{20,20,25,255}
        );

        mb_out = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

        shadow = new shs::ShadowMap(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);

        has_prev_cam = false;
        prev_view    = glm::mat4(1.0f);
        prev_proj    = glm::mat4(1.0f);
    }

    ~RendererSystem()
    {
        delete rt;
        delete mb_out;
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

        glm::mat4 light_proj = shs::Math::ortho_lh_zo(L, R, B, T, zn, zf);
        glm::mat4 light_vp = light_proj * light_view;

        // -----------------------
        // PASS0: ShadowMap depth (XSIMD raster)
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
                            u.model    = glm::mat4(1.0f);
                            u.light_vp = light_vp;

                            const auto& fv = scene->floor->verts;
                            for (size_t i = 0; i < fv.size(); i += 3) {
                                glm::vec3 tri[3] = { fv[i], fv[i+1], fv[i+2] };

                                draw_triangle_tile_shadow_xsimd(
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
                            // Subaru
                            if (auto* car = dynamic_cast<SubaruObject*>(obj)) {
                                Uniforms u;
                                u.model    = car->get_world_matrix();
                                u.light_vp = light_vp;

                                const auto& v = car->geometry->triangles;
                                for (size_t i = 0; i < v.size(); i += 3) {
                                    glm::vec3 tri[3] = { v[i], v[i+1], v[i+2] };
                                    draw_triangle_tile_shadow_xsimd(
                                        *shadow,
                                        tri,
                                        [&u](const glm::vec3& p) { return shadow_vertex_shader(p, u); },
                                        t_min, t_max
                                    );
                                }
                            }

                            // Monkey
                            if (auto* mk = dynamic_cast<MonkeyObject*>(obj)) {
                                Uniforms u;
                                u.model    = mk->get_world_matrix();
                                u.light_vp = light_vp;

                                const auto& v = mk->geometry->triangles;
                                for (size_t i = 0; i < v.size(); i += 3) {
                                    glm::vec3 tri[3] = { v[i], v[i+1], v[i+2] };
                                    draw_triangle_tile_shadow_xsimd(
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
        // PASS1: Camera render -> RT_ColorDepthMotion
        // -----------------------
        rt->clear(shs::Color{20,20,25,255});

        // Skybox background
        if (scene->sky) {
            skybox_background_pass(rt->color, *scene->sky, *scene->viewer->camera, job_system, wg_sky);
        }

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

                        // Floor draw
                        {
                            Uniforms u;
                            u.model           = glm::mat4(1.0f);
                            u.view            = view;
                            u.mv              = u.view * u.model;
                            u.mvp             = proj   * u.mv;
                            u.prev_mvp        = u.mvp;
                            u.normal_mat      = glm::mat3(1.0f);

                            u.light_vp        = light_vp;
                            u.light_dir_world = LIGHT_DIR_WORLD;
                            u.camera_pos      = scene->viewer->position;
                            u.base_color      = shs::Color{ 120, 122, 128, 255 };
                            u.albedo          = nullptr;
                            u.use_texture     = false;
                            u.shadow          = shadow;

                            u.sky             = scene->sky;

                            // IBL
                            u.ibl             = scene->ibl;
                            u.ibl_ambient     = 0.30f;
                            u.ibl_refl        = 0.22f;
                            u.ibl_f0          = 0.04f;
                            u.ibl_refl_mix    = 0.10f;
                            u.shininess       = 32.0f;

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

                        // Objects draw
                        for (shs::AbstractObject3D* obj : scene->scene_objects)
                        {
                            // Subaru
                            if (auto* car = dynamic_cast<SubaruObject*>(obj))
                            {
                                glm::mat4 model = car->get_world_matrix();
                                glm::mat4 mvp   = proj * view * model;

                                glm::mat4 prev_mvp = mvp;
                                if (car->has_prev_mvp) prev_mvp = car->prev_mvp;

                                Uniforms u;
                                u.model           = model;
                                u.view            = view;
                                u.mv              = u.view * u.model;
                                u.mvp             = proj * u.mv;
                                u.prev_mvp        = prev_mvp;
                                u.normal_mat      = glm::transpose(glm::inverse(glm::mat3(u.model)));
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.camera_pos      = scene->viewer->position;
                                u.base_color      = shs::Color{200,200,200,255};
                                u.albedo          = car->albedo;
                                u.use_texture     = (car->albedo && car->albedo->valid());
                                u.shadow          = shadow;

                                u.sky             = scene->sky;

                                // IBL
                                u.ibl             = scene->ibl;
                                u.ibl_ambient     = 0.28f;
                                u.ibl_refl        = 0.38f;
                                u.ibl_f0          = 0.04f;
                                u.ibl_refl_mix    = 0.60f;
                                u.shininess       = 96.0f;

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

                            // Monkey
                            if (auto* mk = dynamic_cast<MonkeyObject*>(obj))
                            {
                                glm::mat4 model = mk->get_world_matrix();
                                glm::mat4 mvp   = proj * view * model;

                                glm::mat4 prev_mvp = mvp;
                                if (mk->has_prev_mvp) prev_mvp = mk->prev_mvp;

                                Uniforms u;
                                u.model           = model;
                                u.view            = view;
                                u.mv              = u.view * u.model;
                                u.mvp             = proj   * u.mv;
                                u.prev_mvp        = prev_mvp;
                                u.normal_mat      = glm::transpose(glm::inverse(glm::mat3(u.model)));
                                u.light_vp        = light_vp;
                                u.light_dir_world = LIGHT_DIR_WORLD;
                                u.camera_pos      = scene->viewer->position;
                                u.base_color      = shs::Color{ 180, 150, 95, 255 };
                                u.albedo          = nullptr;
                                u.use_texture     = false;
                                u.shadow          = shadow;

                                u.sky             = scene->sky;

                                // IBL
                                u.ibl             = scene->ibl;
                                u.ibl_ambient     = 0.30f;
                                u.ibl_refl        = 0.32f;
                                u.ibl_f0          = 0.04f;
                                u.ibl_refl_mix    = 0.35f;
                                u.shininess       = 48.0f;

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
        // PASS2: Combined Motion Blur (raw u32 path)
        // -----------------------
        glm::mat4 curr_view = scene->viewer->camera->view_matrix;
        glm::mat4 curr_proj = scene->viewer->camera->projection_matrix;

        if (!has_prev_cam) {
            prev_view    = curr_view;
            prev_proj    = curr_proj;
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
    shs::Canvas*         mb_out;

    shs::ShadowMap* shadow;

    shs::Job::WaitGroup wg_shadow;
    shs::Job::WaitGroup wg_cam;
    shs::Job::WaitGroup wg_mb;
    shs::Job::WaitGroup wg_sky;

    bool has_prev_cam;
    glm::mat4 prev_view;
    glm::mat4 prev_proj;
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

    // XSIMD мэдээлэл (CPU ISA)
    std::cout << "XSIMD arch: " << xsimd::default_arch::name()
              << " | batch<float>::size=" << xsimd::batch<float>::size << "\n";

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    IMG_Init(IMG_INIT_PNG | IMG_INIT_JPG);

    auto* job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window*   window   = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas* screen_canvas  = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    SDL_Surface* screen_surface = screen_canvas->create_sdl_surface();
    SDL_Texture* screen_texture = SDL_CreateTextureFromSurface(renderer, screen_surface);

    shs::Texture2D car_tex = shs::load_texture_sdl_image("./obj/subaru/SUBARU1_M.bmp", true);

    // Skybox cubemap load (LDR -> CubeMapSky (AbstractSky))
    shs::CubeMap ldr_cm;
    ldr_cm.face[0] = shs::load_texture_sdl_image("./images/skybox/water_scene/right.jpg",  true);
    ldr_cm.face[1] = shs::load_texture_sdl_image("./images/skybox/water_scene/left.jpg",   true);
    ldr_cm.face[2] = shs::load_texture_sdl_image("./images/skybox/water_scene/top.jpg",    true);
    ldr_cm.face[3] = shs::load_texture_sdl_image("./images/skybox/water_scene/bottom.jpg", true);
    ldr_cm.face[4] = shs::load_texture_sdl_image("./images/skybox/water_scene/front.jpg",  true);
    ldr_cm.face[5] = shs::load_texture_sdl_image("./images/skybox/water_scene/back.jpg",   true);

    shs::AbstractSky* active_sky = nullptr;
    if (!ldr_cm.valid()) {
        std::cout << "Warning: Skybox cubemap load failed (images/skybox/water_scene/*.jpg)" << std::endl;
    } else {
        // Use shared CubeMapSky (1.0f intensity for LDR look)
        active_sky = new shs::CubeMapSky(ldr_cm, 1.0f);
        std::cout << "STATUS : Using Shared CubeMapSky" << std::endl;
    }

    // ------------------------------------------------------
    // IBL precompute (програм эхлэхэд нэг удаа)
    // ------------------------------------------------------
    IBLResources ibl;
    const IBLResources* ibl_ptr = nullptr;

    if (ldr_cm.valid()) {
        std::cout << "STATUS : IBL precompute started...\n";

        ibl.env = cubemap_to_float_rgb01(ldr_cm);
        if (ibl.env.valid()) {

            std::cout << "STATUS : IBL diffuse irradiance building..."
                      << " | size=" << IBL_IRR_SIZE
                      << " | samples=" << IBL_IRR_SAMPLES << "\n";

            ibl.irradiance = build_irradiance_cubemap(ibl.env, IBL_IRR_SIZE, IBL_IRR_SAMPLES);

            // Specular prefilter нь өртөг өндөртэй тул base resolution cap хийнэ
            int specBase = std::min(IBL_SPEC_BASE_CAP, ibl.env.size);

            std::cout << "STATUS : IBL specular prefilter building..."
                      << " | base=" << specBase
                      << " | mips=" << IBL_SPEC_MIPCOUNT
                      << " | samples=" << IBL_SPEC_SAMPLES << "\n";

            ibl.spec = build_prefiltered_spec(ibl.env, specBase, IBL_SPEC_MIPCOUNT, IBL_SPEC_SAMPLES);
        }

        if (!ibl.valid()) {
            std::cout << "Warning: IBL precompute failed (falling back to direct lighting only).\n";
        } else {
            std::cout << "STATUS : IBL precompute done.\n";
            ibl_ptr = &ibl;
        }
    }

    shs::Viewer* viewer    = new shs::Viewer(glm::vec3(0.0f, 10.0f, -42.0f), 55.0f, CANVAS_WIDTH, CANVAS_HEIGHT);
    DemoScene*   scene     = new DemoScene(screen_canvas, viewer, &car_tex, active_sky, ibl_ptr);

    SystemProcessor* sys = new SystemProcessor(scene, job_system);

    bool exit        = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();
    bool is_dragging = false;
    int   frames     = 0;
    float fps_timer  = 0.0f;

    while (!exit)
    {
        Uint32 current_tick = SDL_GetTicks();
        float dt            = (current_tick - last_tick) / 1000.0f;
        last_tick           = current_tick;

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
                "IBL(irr+spec) + Shadow + MotionBlur + XSIMD | FPS: " + std::to_string(frames) +
                " | Threads: " + std::to_string(THREAD_COUNT) +
                " | Canvas: " + std::to_string(CANVAS_WIDTH) + "x" + std::to_string(CANVAS_HEIGHT) +
                " | Shadow: " + std::to_string(SHADOW_MAP_SIZE);
            SDL_SetWindowTitle(window, title.c_str());
            frames = 0;
            fps_timer = 0.0f;
        }
    }

    delete sys;
    delete scene;
    delete viewer;
    delete active_sky;

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
