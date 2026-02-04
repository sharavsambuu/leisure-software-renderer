/*

Volumetric Light Shafts (Screen-space) + PCSS Soft Shadows


Нарны чиглэлд тархсан агаарт (манан, тоос, уур) гэрлийн урд чиглэл (forward scattering) 
ойж харагдах үзэгдэл буюу light shafts / god rays-ийг screen-space ray marching аргаар 
ойролцоолох render pass туршилт

- Henyey–Greenstein phase
- depth-aware буюу объектын цаадах агаарыг тооцохгүй


Render pipeline

1. PASS-1: PBR + IBL + PCSS Shadows
   - HDR lighting
   - PCSS Soft Shadows (Blocker Search -> Penumbra -> Variable PCF)
   - Tonemap + Gamma
   - LDR sRGB framebuffer үүсгэнэ

2. PASS-2: Volumetric Light Shafts 
   - PASS-1-ийн LDR output дээр нэмэлт гэрэлтүүлэг хийнэ
   - tonemap нэмж хийхгүй



- Ray marching:
  Камерын харах чиглэлээр агаарт жижиг алхмуудаар (steps) урагшилж,
  нарны гэрэл тухайн цэг дээр хэр их сарниж байгааг хуримтлуулна.

- Phase function (Henyey–Greenstein):
  g параметрээр forward scattering-ийн хүчийг илэрхийлнэ.
  g -> 1.0  => нар руу харах үед тодрох гэрлийн багана
  g ~  0.85 -> бодит агаарт ойролцоо утга

- Extinction:
  sigma_t ≥ sigma_s
  - sigma_s : scattering
  - sigma_t : нийт шингээлт (scattering + absorption)

- Depth-aware termination:
  Туяаг тухайн пикселийн эхний surface (z-buffer) дээр зогсоно.
  Ингэснээр объектын ард байгаа агаараас гэрлийн нэвчилт харагдахгүй.


Screen-space ашиглагдсан шалтгаан

- Shadow map + voxel volumetrics-аас хамаагүй хөнгөн
- Гадаа scene буюу нар тэнгэр гэх мэт дээр сайн ажиллана


Сайжруулах зүйлс
- Нар кадрт байхгүй үед shafts харагдахгүй
- volumetric fog
- Temporal reprojection хэрэгжүүлэх  (jitter нэмэх боломжтой)




Параметрүүдийн тайлбар :

    base_density    : Агаарын ерөнхий нягт
    height_falloff  : Шингэрэх коэффициент
    sigma_s         : Scattering strength
    sigma_t         : Total extinction (>= sigma_s)
    g               : Forward scattering cone
    intensity       : Эцсийн нэмэх хүч
    min_dist        : Камераас эхлэх зай
    max_dist        : Ray march зогсох хамгийн их зай
    steps           : Ray marching алхмын тоо



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
#include <glm/gtc/constants.hpp>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

//#define WINDOW_WIDTH      800
//#define WINDOW_HEIGHT     600
#define WINDOW_WIDTH      800
#define WINDOW_HEIGHT     600
#define CANVAS_WIDTH      800
#define CANVAS_HEIGHT     600


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

// ------------------------------------------
// PCSS (SOFT SHADOWS) CONFIG
// ------------------------------------------
// Гэрлийн хэмжээ томрох тусам penumbra томрох буюу soft болно
static const float LIGHT_UV_RADIUS_BASE = 0.0035f;

// Blocker search radius
static const float PCSS_BLOCKER_SEARCH_RADIUS_TEXELS = 18.0f;

// Filter radius clamp (PCF-ийн final blur)
static const float PCSS_MIN_FILTER_RADIUS_TEXELS = 1.0f;
static const float PCSS_MAX_FILTER_RADIUS_TEXELS = 28.0f;

// Sample counts
static const int   PCSS_BLOCKER_SAMPLES = 12;
static const int   PCSS_PCF_SAMPLES     = 24;

static const float PCSS_EPSILON = 1e-5f;

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
// IBL PRECOMPUTE CONFIG (програм эхлэхэд нэг удаа)
// ------------------------------------------
static const int   IBL_IRR_SIZE      = 16;
static const int   IBL_IRR_SAMPLES   = 64;

static const int   IBL_SPEC_MIPCOUNT = 6;
static const int   IBL_SPEC_SAMPLES  = 16;
static const int   IBL_SPEC_BASE_CAP = 256;

// ------------------------------------------
// PBR CONFIG
// ------------------------------------------
static const float PBR_EXPOSURE      = 1.75f;   // brightness
static const float PBR_GAMMA         = 2.2f;
static const float PBR_MIN_ROUGHNESS = 0.04f;
static const float SKY_EXPOSURE      = 1.85f;

// ------------------------------------------
// LIGHT SHAFTS CONFIG
// ------------------------------------------
struct LightShaftParams
{
    bool  enable         = true;

    int   steps          = 40;        // Ray marching алхмын тоо
    float max_dist       = 110.0f;    // Ray march зогсох хамгийн их зай
    float min_dist       = 1.0f;      // Камераас эхлэх зай

    // Fog/scattering
    float base_density   = 0.18f;     // Агаарын ерөнхий нягт
    float height_falloff = 0.10f;     // Шингэрэх коэффициент
    
    // Dust & Noise тохиргоо
    float noise_scale    = 0.65f;     // Тоосны үүлний хэмжээ (Noise frequency)
    float noise_strength = 0.60f;     // Тоосны нягтын өөрчлөлт (0=жигд, 1=багцлагдсан)
    float jitter_amount  = 1.0f;      // Ray marching алхмыг санамсаргүйгээр зөрүүлэх (Grainy look)
    float ambient_strength = 0.08f;   // Сүүдэр доторх тоосны харагдах хэмжээ (Ambient scattering)

    float sigma_s        = 0.030f;    // Scattering strength
    float sigma_t        = 0.065f;    // Total extinction (>= sigma_s)

    // Henyey–Greenstein phase
    float g              = 0.82f;     // Forward scattering cone (өтгөн туяа мэт байлгахын тулд бага зэрэг өргөн утга хэрэгтэй)

    float intensity      = 0.35f;     // Эцсийн нэмэх хүч

    bool  use_shadow     = true;
    float shadow_bias    = 0.0045f;
    bool  shadow_pcf_2x2 = true;
};

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
static inline float saturate(float v) { return clamp01(v); }

static inline glm::vec3 color_to_srgb01(const shs::Color& c)
{
    return glm::vec3(float(c.r), float(c.g), float(c.b)) / 255.0f;
}

static inline shs::Color srgb01_to_color(const glm::vec3& c01)
{
    glm::vec3 c = glm::clamp(c01, 0.0f, 1.0f) * 255.0f;
    return shs::Color{ (uint8_t)c.x, (uint8_t)c.y, (uint8_t)c.z, 255 };
}

// sRGB -> Linear (pow)
static inline glm::vec3 srgb_to_linear(glm::vec3 srgb01)
{
    srgb01 = glm::clamp(srgb01, 0.0f, 1.0f);
    return glm::pow(srgb01, glm::vec3(PBR_GAMMA));
}

static inline glm::vec3 linear_to_srgb(glm::vec3 lin01)
{
    lin01 = glm::clamp(lin01, 0.0f, 1.0f);
    return glm::pow(lin01, glm::vec3(1.0f / PBR_GAMMA));
}

// tonemap (Reinhard)
static inline glm::vec3 tonemap_reinhard(glm::vec3 x)
{
    return x / (glm::vec3(1.0f) + x);
}

// ==========================================
// LH Ortho matrix (NDC z: 0..1)
// ==========================================

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

// ==========================================
// TEXTURE SAMPLER (nearest, returns sRGB color)
// ==========================================

static inline shs::Color sample_nearest_srgb(const shs::Texture2D &tex, glm::vec2 uv)
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
// SKYBOX CUBEMAP (6 faces) + sampler
// ==========================================

struct CubeMapLDR
{
    // 0:+X right, 1:-X left, 2:+Y top, 3:-Y bottom, 4:+Z front, 5:-Z back
    shs::Texture2D face[6];

    inline bool valid() const
    {
        for (int i = 0; i < 6; ++i) if (!face[i].valid()) return false;
        return true;
    }
};

static inline CubeMapLDR load_cubemap_water_scene(const std::string& folder)
{
    CubeMapLDR cm;
    cm.face[0] = shs::load_texture_sdl_image(folder + "/right.jpg",  true);
    cm.face[1] = shs::load_texture_sdl_image(folder + "/left.jpg",   true);
    cm.face[2] = shs::load_texture_sdl_image(folder + "/top.jpg",    true);
    cm.face[3] = shs::load_texture_sdl_image(folder + "/bottom.jpg", true);
    cm.face[4] = shs::load_texture_sdl_image(folder + "/front.jpg",  true);
    cm.face[5] = shs::load_texture_sdl_image(folder + "/back.jpg",   true);
    return cm;
}

// ==========================================
// IBL FLOAT CUBEMAP (LINEAR) + BILINEAR SAMPLING
// ==========================================

struct CubeMapLinear
{
    int size = 0; // square face
    std::vector<glm::vec3> face[6]; // rgb linear

    inline bool valid() const {
        if (size <= 0) return false;
        for (int i=0;i<6;i++) if ((int)face[i].size() != size*size) return false;
        return true;
    }

    inline const glm::vec3& at(int f, int x, int y) const {
        return face[f][(size_t)y * (size_t)size + (size_t)x];
    }
};

// LDR cubemap (sRGB) -> linear float cubemap
static inline CubeMapLinear cubemap_ldr_to_linear(const CubeMapLDR& cm)
{
    CubeMapLinear out;
    if (!cm.valid()) return out;

    out.size = cm.face[0].w;

    for (int f=0; f<6; ++f) {
        out.face[f].resize((size_t)out.size * (size_t)out.size);
        for (int y=0; y<out.size; ++y) {
            for (int x=0; x<out.size; ++x) {
                shs::Color c = cm.face[f].texels.at(x, y);
                glm::vec3 srgb01 = color_to_srgb01(c);
                out.face[f][(size_t)y*out.size + (size_t)x] = srgb_to_linear(srgb01);
            }
        }
    }
    return out;
}

static inline glm::vec3 sample_face_bilinear_linear(const CubeMapLinear& cm, int face, float u, float v)
{
    u = clamp01(u);
    v = clamp01(v);

    float fx = u * float(cm.size - 1);
    float fy = v * float(cm.size - 1);

    int x0   = clampi((int)std::floor(fx), 0, cm.size - 1);
    int y0   = clampi((int)std::floor(fy), 0, cm.size - 1);
    int x1   = clampi(x0 + 1, 0, cm.size - 1);
    int y1   = clampi(y0 + 1, 0, cm.size - 1);

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

// dir_world -> cubemap (face + uv) (LH +Z forward)
static inline glm::vec3 sample_cubemap_linear(const CubeMapLinear& cm, const glm::vec3& dir_world)
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
    float u  = 0.5f, v = 0.5f;

    if (ax >= ay && ax >= az) {
        // ±X
        if (d.x > 0.0f) { // +X
            face = 0;
            u = (-d.z / ax);
            v = ( d.y / ax);
        } else {          // -X
            face = 1;
            u = ( d.z / ax);
            v = ( d.y / ax);
        }
    } else if (ay >= ax && ay >= az) {
        // ±Y
        if (d.y > 0.0f) { // +Y
            face = 2;
            u = ( d.x / ay);
            v = (-d.z / ay);
        } else {          // -Y
            face = 3;
            u = ( d.x / ay);
            v = ( d.z / ay);
        }
    } else {
        // ±Z
        if (d.z > 0.0f) { // +Z front
            face = 4;
            u = ( d.x / az);
            v = ( d.y / az);
        } else {          // -Z back
            face = 5;
            u = (-d.x / az);
            v = ( d.y / az);
        }
    }

    // [-1,1] -> [0,1]
    u = 0.5f * (u + 1.0f);
    v = 0.5f * (v + 1.0f);

    return sample_face_bilinear_linear(cm, face, u, v);
}

// ==========================================
// IBL PRECOMPUTE: Irradiance + Prefiltered Specular Mips
// ==========================================

static inline glm::vec3 face_uv_to_dir(int face, float u, float v)
{
    // u,v [0,1] -> a,b [-1,1]
    float a = 2.0f * u - 1.0f;
    float b = 2.0f * v - 1.0f;

    glm::vec3 d(0.0f);

    // sample_cubemap_linear mapping-тэй 1:1 таарах ёстой
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
    glm::vec3 up = (std::abs(N.y) < 0.999f) ? glm::vec3(0,1,0) : glm::vec3(1,0,0);
    T            = glm::normalize(glm::cross(up, N));
    B            = glm::cross(N, T);
}

static inline CubeMapLinear build_env_irradiance(const CubeMapLinear& env_radiance, int outSize, int sampleCount)
{
    CubeMapLinear irr;
    irr.size = outSize;
    for (int f=0; f<6; ++f) irr.face[f].assign((size_t)outSize*(size_t)outSize, glm::vec3(0.0f));

    for (int f=0; f<6; ++f) {
        for (int y=0; y<outSize; ++y) {
            for (int x=0; x<outSize; ++x) {

                float u = (float(x) + 0.5f) / float(outSize);
                float v = (float(y) + 0.5f) / float(outSize);

                glm::vec3 N = face_uv_to_dir(f, u, v);
                glm::vec3 T,B;
                tangent_basis(N, T, B);

                glm::vec3 sum(0.0f);

                // deterministic random
                uint32_t seed = (uint32_t)(f*73856093u ^ x*19349663u ^ y*83492791u);
                auto rnd01 = [&seed]() {
                    seed = 1664525u * seed + 1013904223u;
                    return float(seed & 0x00FFFFFFu) / float(0x01000000u);
                };

                for (int i=0; i<sampleCount; ++i) {
                    float r1 = rnd01();
                    float r2 = rnd01();

                    glm::vec3 h = cosine_sample_hemisphere(r1, r2);
                    glm::vec3 L = glm::normalize(T*h.x + B*h.y + N*h.z);

                    sum += sample_cubemap_linear(env_radiance, L);
                }

                irr.face[f][(size_t)y*outSize + (size_t)x] = sum / float(sampleCount);
            }
        }
    }

    return irr;
}

//  PREFILTER
static inline float roughness_to_phong_exp(float rough)
{
    rough     = clamp01(rough);
    float r2  = std::max(1e-4f, rough*rough);
    float exp = (2.0f / r2) - 2.0f;
    return std::max(1.0f, exp);
}

static inline glm::vec3 phong_lobe_sample(float u1, float u2, float exp)
{
    float phi  = 6.2831853f * u1;
    float cosT = std::pow(1.0f - u2, 1.0f / (exp + 1.0f));
    float sinT = std::sqrt(std::max(0.0f, 1.0f - cosT*cosT));
    return glm::vec3(std::cos(phi)*sinT, std::sin(phi)*sinT, cosT);
}

struct PrefilteredSpecular
{
    std::vector<CubeMapLinear> mip; // mip[0]=sharp, mip[last]=blur

    inline bool valid() const { return !mip.empty() && mip[0].valid(); }
    inline int  mip_count() const { return (int)mip.size(); }
};

static inline PrefilteredSpecular build_env_prefiltered_specular(
    const CubeMapLinear& env_radiance,
    int baseSize,
    int mipCount,
    int samplesPerTexel)
{
    PrefilteredSpecular out;
    out.mip.resize(mipCount);

    for (int m=0; m<mipCount; ++m) {
        int sz = std::max(1, baseSize >> m);

        std::cout << "STATUS :   Env prefilter mip " << m << "/" << (mipCount - 1)
                  << " | size=" << sz
                  << " | samples=" << samplesPerTexel
                  << std::endl;

        out.mip[m].size = sz;
        for (int f=0; f<6; ++f) out.mip[m].face[f].assign((size_t)sz*(size_t)sz, glm::vec3(0.0f));

        float rough = float(m) / float(std::max(1, mipCount - 1));
        float exp   = roughness_to_phong_exp(rough);

        for (int f=0; f<6; ++f) {
            for (int y=0; y<sz; ++y) {
                for (int x=0; x<sz; ++x) {

                    float u = (float(x) + 0.5f) / float(sz);
                    float v = (float(y) + 0.5f) / float(sz);

                    glm::vec3 R = face_uv_to_dir(f, u, v);
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

                        sum += sample_cubemap_linear(env_radiance, L);
                    }

                    out.mip[m].face[f][(size_t)y*sz + (size_t)x] = sum / float(samplesPerTexel);
                }
            }
        }
    }

    return out;
}

static inline glm::vec3 sample_prefiltered_spec_trilinear(
    const PrefilteredSpecular& ps,
    const glm::vec3& dir_world,
    float lod)
{
    if (!ps.valid()) return glm::vec3(0.0f);

    float mmax = float(ps.mip_count() - 1);
    lod        = clampf(lod, 0.0f, mmax);

    int m0  = (int)std::floor(lod);
    int m1  = std::min(m0 + 1, ps.mip_count() - 1);
    float t = lod - float(m0);

    glm::vec3 c0 = sample_cubemap_linear(ps.mip[m0], dir_world);
    glm::vec3 c1 = sample_cubemap_linear(ps.mip[m1], dir_world);
    return glm::mix(c0, c1, t);
}

struct EnvIBL
{
    CubeMapLinear         env_radiance;            // linear
    CubeMapLinear         env_irradiance;          // diffuse convolved
    PrefilteredSpecular   env_prefiltered_spec;    // spec mip chain

    inline bool valid() const {
        return env_radiance.valid() && env_irradiance.valid() && env_prefiltered_spec.valid();
    }
};

// ==========================================
// PBR (GGX) FUNCTIONS
// ==========================================

namespace PBR
{
    static constexpr float PI = 3.14159265358979323846f;

    static inline glm::vec3 fresnel_schlick(glm::vec3 F0, float NoV)
    {
        NoV      = saturate(NoV);
        float x  = 1.0f - NoV;
        float x2 = x*x;
        float x5 = x2*x2*x;
        return F0 + (glm::vec3(1.0f) - F0) * x5;
    }

    static inline float ndf_ggx(float NoH, float alpha)
    {
        NoH      = saturate(NoH);
        float a2 = alpha * alpha;
        float d  = (NoH*NoH) * (a2 - 1.0f) + 1.0f;
        return a2 / (PI * d * d);
    }

    static inline float g_schlick_ggx(float NoV, float k)
    {
        NoV = saturate(NoV);
        return NoV / (NoV * (1.0f - k) + k);
    }

    static inline float g_smith(float NoV, float NoL, float roughness)
    {
        roughness = clampf(roughness, PBR_MIN_ROUGHNESS, 1.0f);

        // UE4 style k
        float r   = roughness + 1.0f;
        float k   = (r*r) / 8.0f;

        float gv  = g_schlick_ggx(NoV, k);
        float gl  = g_schlick_ggx(NoL, k);
        return gv * gl;
    }
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
// MOTION BUFFER (Canvas coords, pixels, +Y up)
// ==========================================

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

// ==========================================
// RT: Color + Depth(view_z) + Motion(full)
// ==========================================

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

    shs::Canvas  color;
    shs::ZBuffer depth;   // view_z
    MotionBuffer motion;  // v_full
};

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
        rotation_angle += 12.0f * dt;
        if (rotation_angle >= 360.0f) rotation_angle -= 360.0f;
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
        this->geometry = new ModelGeometry("./obj/monkey/monkey.rawobj");

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
// PBR MATERIAL + UNIFORMS + VARYINGS
// ==========================================

struct MaterialPBR
{
    shs::Color baseColor_srgb = shs::Color{200,200,200,255};
    float metallic  = 0.0f;
    float roughness = 0.5f;
    float ao        = 1.0f;
};

struct Uniforms
{
    glm::mat4 mvp;
    glm::mat4 prev_mvp;
    glm::mat4 model;
    glm::mat4 view;

    glm::mat4 mv;
    glm::mat3 normal_mat;

    glm::mat4 light_vp;

    glm::vec3 light_dir_world;
    glm::vec3 camera_pos;

    MaterialPBR mat;

    const shs::Texture2D *albedo = nullptr;
    bool use_texture             = false;

    const ShadowMap *shadow = nullptr;
    const CubeMapLDR *sky   = nullptr;
    const EnvIBL *ibl       = nullptr;

    float ibl_diffuse_intensity   = 0.30f;
    float ibl_specular_intensity  = 0.35f;
    float ibl_reflection_strength = 1.00f;
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

    glm::vec4 world_h = u.model * glm::vec4(aPos, 1.0f);
    out.world_pos     = glm::vec3(world_h);

    out.normal        = glm::normalize(u.normal_mat * aNormal);
    out.uv            = aUV;

    glm::vec4 view_pos = u.mv * glm::vec4(aPos, 1.0f);
    out.view_z = view_pos.z;

    return out;
}

// ==========================================
// SHADOW HELPERS + PCSS
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

    if (out_z_ndc < 0.0f || out_z_ndc > 1.0f) return false;

    out_uv.x = ndc.x * 0.5f + 0.5f;
    out_uv.y = 1.0f - (ndc.y * 0.5f + 0.5f);

    return true;
}

// PCSS-д зориулсан helper
static inline float shadow_sample_depth_uv(const ShadowMap& sm, glm::vec2 uv)
{
    // uv -> nearest depth
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f)
        return std::numeric_limits<float>::max();

    int x = (int)std::lround(uv.x * float(sm.w - 1));
    int y = (int)std::lround(uv.y * float(sm.h - 1));
    return sm.sample(x, y);
}

// Poisson disk offsets (2D) — fixed pattern
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

// Hash / rotate (per-pixel) — banding багасгах
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

// PCSS Soft Shadow Factor
static inline float pcss_shadow_factor(
    const ShadowMap& sm,
    const glm::vec2& uv,
    float z_receiver,
    float bias,
    int px, int py)
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
// PBR FRAGMENT SHADER (Direct GGX + IBL + PCSS)
// ==========================================

static shs::Color fragment_shader_pbr(const VaryingsFull& in, const Uniforms& u, int px, int py)
{
    glm::vec3 N = glm::normalize(in.normal);
    glm::vec3 V = glm::normalize(u.camera_pos - in.world_pos);
    glm::vec3 L = glm::normalize(-u.light_dir_world);
    glm::vec3 H = glm::normalize(V + L);

    float NoV   = glm::max(0.0f, glm::dot(N, V));
    float NoL   = glm::max(0.0f, glm::dot(N, L));
    float NoH   = glm::max(0.0f, glm::dot(N, H));

    glm::vec3 baseColor_linear(0.7f);

    if (u.use_texture && u.albedo && u.albedo->valid()) {
        shs::Color tc    = sample_nearest_srgb(*u.albedo, in.uv);
        baseColor_linear = srgb_to_linear(color_to_srgb01(tc));
    } else {
        baseColor_linear = srgb_to_linear(color_to_srgb01(u.mat.baseColor_srgb));
    }

    float metallic  = clamp01(u.mat.metallic);
    float roughness = clampf(u.mat.roughness, PBR_MIN_ROUGHNESS, 1.0f);
    float ao        = clamp01(u.mat.ao);

    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), baseColor_linear, metallic);

    glm::vec3 F  = PBR::fresnel_schlick(F0, NoV);
    F           *= glm::vec3(1.0f, 0.96f, 0.90f);

    glm::vec3 kd = (glm::vec3(1.0f) - F) * (1.0f - metallic);

    float alpha = roughness * roughness;
    float D     = PBR::ndf_ggx(NoH, alpha);
    float G     = PBR::g_smith(NoV, NoL, roughness);

    glm::vec3 direct_diffuse  = kd * baseColor_linear * (1.0f / PBR::PI);
    glm::vec3 direct_specular = (D * G) * F / glm::max(1e-6f, (4.0f * NoV * NoL));

    glm::vec3 direct_radiance = glm::vec3(3.0f);
    glm::vec3 direct          = (direct_diffuse + direct_specular) * direct_radiance * NoL;

    float shadow = 1.0f;
    if (u.shadow) {
        glm::vec2 suv;
        float sz;
        if (shadow_uvz_from_world(u.light_vp, in.world_pos, suv, sz)) {
            float slope = 1.0f - glm::clamp(glm::dot(N, L), 0.0f, 1.0f);
            float bias  = SHADOW_BIAS_BASE + SHADOW_BIAS_SLOPE * slope;
            // PCSS Implementation:
            shadow = pcss_shadow_factor(*u.shadow, suv, sz, bias, px, py);
        }
    }
    direct *= shadow;

    glm::vec3 ibl(0.0f);

    if (u.ibl && u.ibl->valid()) {

        glm::vec3 irradiance  = sample_cubemap_linear(u.ibl->env_irradiance, N);
        glm::vec3 diffuseIBL  = irradiance * baseColor_linear * kd;
        diffuseIBL           *= clamp01(u.ibl_diffuse_intensity);

        glm::vec3 R           = glm::reflect(-V, N);

        float lod             = roughness * float(u.ibl->env_prefiltered_spec.mip_count() - 1);
        glm::vec3 prefiltered = sample_prefiltered_spec_trilinear(u.ibl->env_prefiltered_spec, R, lod);

        glm::vec3 specIBL     = prefiltered * F;
        specIBL              *= clamp01(u.ibl_specular_intensity) * clamp01(u.ibl_reflection_strength);

        ibl                   = diffuseIBL + specIBL;
    }

    ibl *= ao;

    glm::vec3 color_linear = direct + ibl;

    color_linear        *= PBR_EXPOSURE;
    color_linear         = tonemap_reinhard(color_linear);
    glm::vec3 color_srgb = linear_to_srgb(color_linear);

    return srgb01_to_color(color_srgb);
}

// ==========================================
// SKYBOX BACKGROUND PASS (parallel tiled)
// ==========================================

static void skybox_background_pass(
    shs::Canvas& dst,
    const CubeMapLDR& sky,
    const shs::Camera3D& cam,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    if (!sky.valid()) return;

    int W = dst.get_width();
    int H = dst.get_height();

    float aspect       = float(W) / float(H);
    float tan_half_fov = std::tan(glm::radians(cam.field_of_view) * 0.5f);

    glm::vec3 forward  = glm::normalize(cam.direction_vector);
    glm::vec3 right    = glm::normalize(cam.right_vector);
    glm::vec3 up       = glm::normalize(cam.up_vector);

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    auto sample_cubemap_nearest_srgb01 = [&](const glm::vec3& dir_world) -> glm::vec3 {

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

        const shs::Texture2D& tex = sky.face[face];
        shs::Color c = sample_nearest_srgb(tex, glm::vec2(u, v));
        return color_to_srgb01(c);
    };

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

                        float fx    = (float(x) + 0.5f) / float(W);
                        float fy    = (float(y) + 0.5f) / float(H);

                        float ndc_x = fx * 2.0f - 1.0f;
                        float ndc_y = fy * 2.0f - 1.0f;

                        glm::vec3 dir =
                            forward +
                            right * (ndc_x * aspect * tan_half_fov) +
                            up    * (ndc_y * tan_half_fov);

                        dir = glm::normalize(dir);

                        glm::vec3 sky_srgb = sample_cubemap_nearest_srgb01(dir);
                        glm::vec3 sky_lin  = srgb_to_linear(sky_srgb);

                        sky_lin *= SKY_EXPOSURE;
                        sky_lin  = tonemap_reinhard(sky_lin);
                        glm::vec3 out_srgb = linear_to_srgb(sky_lin);

                        dst.draw_pixel(x, y, srgb01_to_color(out_srgb));
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
// CAMERA PASS RASTER (Color + Depth + Motion)
// (Near-plane clipping in clip-space: z >= 0)
// ==========================================

static inline glm::vec2 clip_to_screen_xy(const glm::vec4& clip, int w, int h)
{
    glm::vec3 s = shs::Canvas::clip_to_screen(clip, w, h);
    return glm::vec2(s.x, s.y);
}

static void draw_triangle_tile_color_depth_motion(
    RT_ColorDepthMotion& rt,
    const std::vector<glm::vec3>& tri_verts,
    const std::vector<glm::vec3>& tri_norms,
    const std::vector<glm::vec2>& tri_uvs,
    std::function<VaryingsFull(const glm::vec3&, const glm::vec3&, const glm::vec2&)> vs,
    std::function<shs::Color(const VaryingsFull&, int, int)> fs, // Updated to take px, py
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    int W = rt.color.get_width();
    int H = rt.color.get_height();

    auto lerp_vary = [](const VaryingsFull& a, const VaryingsFull& b, float t) -> VaryingsFull {
        VaryingsFull o;
        o.position      = a.position      + (b.position      - a.position     ) * t;
        o.prev_position = a.prev_position + (b.prev_position - a.prev_position) * t;
        o.world_pos     = a.world_pos     + (b.world_pos     - a.world_pos    ) * t;
        o.normal        = a.normal        + (b.normal        - a.normal       ) * t;
        o.uv            = a.uv            + (b.uv            - a.uv           ) * t;
        o.view_z        = a.view_z        + (b.view_z        - a.view_z       ) * t;
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
                for (int py = (int)bboxmax.y; py <= (int)bboxmin.y; py--) {
                    // bbox-ийг tile-аар аль хэдийн хязгаарласан, y ordering хамаагүй
                    (void)py;
                }
            }

            for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
                for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

                    glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                    if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                    float vz = bc.x * tv[0].view_z + bc.y * tv[1].view_z + bc.z * tv[2].view_z;

                    int cy = (H - 1) - py;

                    if (rt.depth.test_and_set_depth(px, cy, vz)) {

                        float w0 = tv[0].position.w;
                        float w1 = tv[1].position.w;
                        float w2 = tv[2].position.w;

                        float invw0 = (std::abs(w0) < 1e-6f) ? 0.0f : 1.0f / w0;
                        float invw1 = (std::abs(w1) < 1e-6f) ? 0.0f : 1.0f / w1;
                        float invw2 = (std::abs(w2) < 1e-6f) ? 0.0f : 1.0f / w2;

                        float invw_sum = bc.x * invw0 + bc.y * invw1 + bc.z * invw2;
                        if (invw_sum <= 1e-8f) continue;

                        VaryingsFull vin;
                        vin.position      = bc.x * tv[0].position      + bc.y * tv[1].position      + bc.z * tv[2].position;
                        vin.prev_position = bc.x * tv[0].prev_position + bc.y * tv[1].prev_position + bc.z * tv[2].prev_position;

                        vin.normal = glm::normalize(bc.x * tv[0].normal + bc.y * tv[1].normal + bc.z * tv[2].normal);

                        glm::vec3 wp_over_w =
                            bc.x * (tv[0].world_pos * invw0) +
                            bc.y * (tv[1].world_pos * invw1) +
                            bc.z * (tv[2].world_pos * invw2);
                        vin.world_pos = wp_over_w / invw_sum;

                        glm::vec2 uv_over_w =
                            bc.x * (tv[0].uv * invw0) +
                            bc.y * (tv[1].uv * invw1) +
                            bc.z * (tv[2].uv * invw2);
                        vin.uv = uv_over_w / invw_sum;

                        vin.view_z = vz;

                        glm::vec2 curr_s   = clip_to_screen_xy(vin.position, W, H);
                        glm::vec2 prev_s   = clip_to_screen_xy(vin.prev_position, W, H);
                        glm::vec2 v_screen = curr_s - prev_s;
                        glm::vec2 v_canvas = glm::vec2(v_screen.x, -v_screen.y);

                        float len = glm::length(v_canvas);
                        if (len > MB_MAX_PIXELS && len > 1e-6f) {
                            v_canvas *= (MB_MAX_PIXELS / len);
                        }
                        rt.motion.set(px, cy, v_canvas);

                        rt.color.draw_pixel_screen_space(px, py, fs(vin, px, py));
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
    float ndc_z = viewz_to_ndcz(view_z, curr_proj);

    glm::vec4 clip_curr(ndc_xy.x, ndc_xy.y, ndc_z, 1.0f);

    glm::vec4 world_h = inv_curr_viewproj * clip_curr;
    if (std::abs(world_h.w) < 1e-6f) return glm::vec2(0.0f);
    glm::vec3 world   = glm::vec3(world_h) / world_h.w;

    glm::vec4 prev_clip = prev_viewproj * glm::vec4(world, 1.0f);
    if (std::abs(prev_clip.w) < 1e-6f) return glm::vec2(0.0f);
    glm::vec3 prev_ndc  = glm::vec3(prev_clip) / prev_clip.w;

    int py_screen = (H - 1) - y;
    glm::vec2 curr_screen{ float(x), float(py_screen) };
    glm::vec2 prev_screen = ndc_to_screen_xy(prev_ndc, W, H);

    glm::vec2 v_screen    = curr_screen - prev_screen;
    glm::vec2 v_canvas    = glm::vec2(v_screen.x, -v_screen.y);
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
    const MotionBuffer& v_full_buf,
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

    glm::mat4 inv_curr_vp = glm::inverse(curr_vp);

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const shs::Color* src_raw  = src.buffer().raw();
    shs::Color*       dst_raw  = dst.buffer().raw();
    const float*      z_raw    = depth.buffer().raw();
    const glm::vec2*  v_raw    = v_full_buf.vel.data();

    wg.reset();

    for (int ty = 0; ty < rows; ty++) {
        for (int tx = 0; tx < cols; tx++) {

            wg.add(1);
            job_system->submit({[=, &wg]() {

                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                auto sample_fast = [&](int sx, int sy) -> shs::Color {
                    sx = clampi(sx, 0, W - 1);
                    sy = clampi(sy, 0, H - 1);
                    return src_raw[sy * W + sx];
                };

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

                        if (len < 0.001f || samples <= 1) {
                            dst_raw[row_off + x] = src_raw[row_off + x];
                            continue;
                        }

                        glm::vec2 dir = v_total / len;

                        float r = 0, g = 0, b = 0;
                        float wsum = 0.0f;

                        for (int i = 0; i < samples; i++) {
                            float t = (samples == 1) ? 0.0f : (float(i) / float(samples - 1));
                            float a = (t - 0.5f) * 2.0f;

                            glm::vec2 p = glm::vec2(float(x), float(y)) + dir * (a * len);

                            int sx = clampi((int)std::round(p.x), 0, W - 1);
                            int sy = clampi((int)std::round(p.y), 0, H - 1);

                            float wgt = 1.0f - std::abs(a);
                            shs::Color c = sample_fast(sx, sy);

                            r += wgt * float(c.r);
                            g += wgt * float(c.g);
                            b += wgt * float(c.b);
                            wsum += wgt;
                        }

                        if (wsum < 0.0001f) wsum = 1.0f;

                        dst_raw[row_off + x] = shs::Color{
                            (uint8_t)clampi((int)(r / wsum), 0, 255),
                            (uint8_t)clampi((int)(g / wsum), 0, 255),
                            (uint8_t)clampi((int)(b / wsum), 0, 255),
                            255
                        };
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

// ================================================================================================
// VOLUMETRIC LIGHT SHAFTS PASS (tiled, ray marching)
// ================================================================================================

// Canvas pixel -> world view ray direction (inv_vp ашиглана)
static inline glm::vec3 reconstruct_world_dir_from_pixel(
    int x, int y, int W, int H,
    const glm::mat4& inv_vp,
    const glm::vec3& cam_pos)
{
    // Canvas y -> Screen y
    int py_screen = (H - 1) - y;

    float fx = (float(x) + 0.5f) / float(W);
    float fy = (float(py_screen) + 0.5f) / float(H);

    float ndc_x = fx * 2.0f - 1.0f;
    float ndc_y = 1.0f - fy * 2.0f;

    // Far plane point (z=1)
    glm::vec4 clip_far(ndc_x, ndc_y, 1.0f, 1.0f);
    glm::vec4 world_far = inv_vp * clip_far;
    if (std::abs(world_far.w) < 1e-8f) return glm::vec3(0,0,1);

    world_far /= world_far.w;

    glm::vec3 dir = glm::vec3(world_far) - cam_pos;
    float len = glm::length(dir);
    if (len < 1e-6f) return glm::vec3(0,0,1);
    return dir / len;
}

// Height fog density
static inline float fog_density(const LightShaftParams& p, const glm::vec3& world_pos)
{
    float y = world_pos.y;
    float h = std::max(0.0f, y);
    float d = p.base_density * std::exp(-h * p.height_falloff);
    return d;
}

// Henyey–Greenstein phase (normalized constant-гүй хувилбар)
static inline float phase_hg(float cosTheta, float g)
{
    cosTheta    = clampf(cosTheta, -1.0f, 1.0f);
    g           = clampf(g, -0.95f, 0.95f);

    float gg    = g * g;
    float denom = std::pow(std::max(1e-4f, 1.0f + gg - 2.0f * g * cosTheta), 1.5f);
    return (1.0f - gg) / denom;
}

// Shadow raw sampling
static inline float shadow_sample_depth_raw(const float* shadow_raw, int sw, int sh, int x, int y)
{
    x = std::max(0, std::min(sw - 1, x));
    y = std::max(0, std::min(sh - 1, y));
    return shadow_raw[(size_t)y * (size_t)sw + (size_t)x];
}

static inline float shadow_compare_raw(
    const float* shadow_raw, int sw, int sh,
    glm::vec2 uv, float z_ndc, float bias)
{
    if (!shadow_raw || sw <= 0 || sh <= 0) return 1.0f;
    if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) return 1.0f;

    int x = (int)std::lround(uv.x * float(sw - 1));
    int y = (int)std::lround(uv.y * float(sh - 1));

    float d = shadow_sample_depth_raw(shadow_raw, sw, sh, x, y);
    if (d == std::numeric_limits<float>::max()) return 1.0f;

    return (z_ndc <= d + bias) ? 1.0f : 0.0f;
}

static inline float shadow_factor_pcf_2x2_raw(
    const float* shadow_raw, int sw, int sh,
    glm::vec2 uv, float z_ndc, float bias, bool enable_pcf)
{
    if (!enable_pcf) return shadow_compare_raw(shadow_raw, sw, sh, uv, z_ndc, bias);
    if (!shadow_raw || sw <= 0 || sh <= 0) return 1.0f;

    float fx = uv.x * float(sw - 1);
    float fy = uv.y * float(sh - 1);

    int x0 = std::max(0, std::min(sw - 1, (int)std::floor(fx)));
    int y0 = std::max(0, std::min(sh - 1, (int)std::floor(fy)));
    int x1 = std::max(0, std::min(sw - 1, x0 + 1));
    int y1 = std::max(0, std::min(sh - 1, y0 + 1));

    float d00 = shadow_sample_depth_raw(shadow_raw, sw, sh, x0, y0);
    float d10 = shadow_sample_depth_raw(shadow_raw, sw, sh, x1, y0);
    float d01 = shadow_sample_depth_raw(shadow_raw, sw, sh, x0, y1);
    float d11 = shadow_sample_depth_raw(shadow_raw, sw, sh, x1, y1);

    auto cmp = [&](float d) {
        if (d == std::numeric_limits<float>::max()) return 1.0f;
        return (z_ndc <= d + bias) ? 1.0f : 0.0f;
    };

    return 0.25f * (cmp(d00) + cmp(d10) + cmp(d01) + cmp(d11));
}

static inline float volumetric_shadow_visibility(
    const LightShaftParams& p,
    const float* shadow_raw, int sw, int sh,
    const glm::mat4& light_vp,
    const glm::vec3& world_pos)
{
    if (!p.use_shadow || !shadow_raw) return 1.0f;

    glm::vec2 uv; float z;
    if (!shadow_uvz_from_world(light_vp, world_pos, uv, z)) return 1.0f;

    // Volumetric shafts, PCF 2x2
    return shadow_factor_pcf_2x2_raw(shadow_raw, sw, sh, uv, z, p.shadow_bias, p.shadow_pcf_2x2);
}


// ------------------------------------------
// NOISE HELPER
// ------------------------------------------

// 3D Noise: Тоосны бөөгнөрөл (clumps) үүсгэхэд ашиглана.
// Sine долгионуудыг хольж бага зэргийн үүл маягтай бүтэц үүсгэнэ.
static inline float volumetric_cloud_noise(const glm::vec3& p, float scale)
{
    glm::vec3 s = p * scale;
    // вариац үүсгэхэд зориулсан синусойд холилт
    float n = std::sin(s.x) * std::cos(s.y) * std::sin(s.z + s.x * 0.5f);
    return n * 0.5f + 0.5f; // Map to [0..1]
}


static void light_shafts_pass(
    const shs::Canvas& src,
    const shs::ZBuffer& depth_vz,
    shs::Canvas& dst,

    const glm::vec3& cam_pos,
    const glm::mat4& inv_curr_vp,

    const glm::vec3& sun_dir_world,   // sun -> scene ray direction

    const glm::mat4& light_vp,
    const float* shadow_raw,
    int shadow_w,
    int shadow_h,

    const LightShaftParams& p,

    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    // Tint colors for atmosphere
    // Нарны тусгалтай хэсгийн өнгө (Light Shafts tint)
    const glm::vec3 shafts_tint = glm::vec3(0.92f, 0.96f, 1.00f);
    // Сүүдэр болон тоосны үндсэн өнгө (Ambient Dust tint)
    const glm::vec3 ambient_tint = glm::vec3(0.60f, 0.65f, 0.70f);

    const int W = src.get_width();
    const int H = src.get_height();

    const shs::Color* src_raw = src.buffer().raw();
    shs::Color*       dst_raw = dst.buffer().raw();
    const float*      z_raw   = depth_vz.buffer().raw();

    glm::vec3 sun_dir = -glm::normalize(sun_dir_world);

    const int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    auto process_tile = [&](int tx, int ty)
    {
        int x0 = tx * TILE_SIZE_X;
        int y0 = ty * TILE_SIZE_Y;
        int x1 = std::min(x0 + TILE_SIZE_X, W);
        int y1 = std::min(y0 + TILE_SIZE_Y, H);

        for (int y = y0; y < y1; ++y) {
            int row = y * W;

            for (int x = x0; x < x1; ++x) {

                shs::Color base_srgb = src_raw[row + x];

                if (!p.enable) {
                    dst_raw[row + x] = base_srgb;
                    continue;
                }

                float view_z = z_raw[row + x];

                // Depth мэдрэг termination логик
                float ray_max = p.max_dist;
                if (view_z != std::numeric_limits<float>::max()) {
                    ray_max = std::min(ray_max, std::max(p.min_dist, view_z - 0.25f)); // 0.25 world units offset
                }
                if (ray_max <= p.min_dist) {
                    dst_raw[row + x] = base_srgb;
                    continue;
                }

                glm::vec3 view_dir = reconstruct_world_dir_from_pixel(x, y, W, H, inv_curr_vp, cam_pos);

                float cosTheta = glm::dot(view_dir, sun_dir);
                float phase    = phase_hg(cosTheta, p.g);

                // Нарны эргэн тойронд гэрэлтүүлгийг төвлөрүүлэх (Gate function)
                float gate = saturate((cosTheta - 0.1f) / 0.9f);
                
                int steps = std::max(1, p.steps);
                float ds  = ray_max / float(steps);

                // JITTERING (Dithering)
                // Pixel бүрийн ray эхлэх цэгийг санамсаргүйгээр бага зэрэг шилжүүлнэ.
                // banding буюу зурааслаг харагдах эффектийг арилгаж, grain буюу тоосорхог байдалтай харагдуулна.
                uint32_t seed = (uint32_t)(x * 1973u ^ y * 9277u);
                float random_val = float(seed & 0xFFFF) / 65536.0f;
                float t = p.min_dist + (random_val * ds * p.jitter_amount);

                float Tm = 1.0f;
                glm::vec3 Ls(0.0f);

                for (int i = 0; i < steps; ++i) {
                    if (t >= ray_max) break;

                    glm::vec3 wp = cam_pos + view_dir * t;

                    // Суурь нягтралшил (Height Fog тооцолол)
                    float dens = fog_density(p, wp);

                    // Агаарын нягтыг 3D noise ашиглан өөрчилж, тоосны бөөгнөрөл мягтай юм үүсгэх.
                    if (dens > 1e-6f) {
                        float dust = volumetric_cloud_noise(wp, p.noise_scale);
                        // суурь нягтралшилийг noise-той холих
                        dens *= glm::mix(1.0f, dust, p.noise_strength);
                    }

                    if (dens > 1e-6f) {
                        float vis = volumetric_shadow_visibility(p, shadow_raw, shadow_w, shadow_h, light_vp, wp);

                        float sigma_s = p.sigma_s * dens;
                        float sigma_t = p.sigma_t * dens;

                        // Ambient + Direct Lighting
                        // Direct Light (vis > 0 үед харагдана)
                        float direct_light = phase * vis * gate; 

                        // Ambient Light (vis = 0 үед буюу сүүдэрт ч харагдана)
                        // ингэснээр сүүдэр дотор тоос байгаа юм шиг санагдуулна.
                        float ambient_light = p.ambient_strength;

                        // Нийлбэр гэрэлтүүлэг
                        float light_term = (direct_light * p.intensity) + ambient_light;
                        
                        // Scattering хуримтлуулах
                        float scatter = Tm * sigma_s * light_term * ds;

                        // Distance attenuation (холоос харахад манан хэтэрхий тоо байж болохгүй)
                        float dist01 = t / std::max(1e-3f, p.max_dist);
                        float dist_fall = 1.0f - (dist01 * dist01); 
                        scatter *= dist_fall;

                        // Tint холих
                        // Гэрэлтэй хэсэгт shafts_tint, сүүдэрт ambient_tint ашиглана.
                        glm::vec3 current_tint = glm::mix(ambient_tint, shafts_tint, vis);
                        
                        Ls += current_tint * scatter;

                        Tm *= std::exp(-sigma_t * ds);

                        if (Tm < 0.01f) break;
                    }

                    t += ds;
                }

                glm::vec3 base01  = color_to_srgb01(base_srgb);
                glm::vec3 baseLin = srgb_to_linear(base01);

                // volumetric оролцоог сайжруулах
                glm::vec3 outLin = baseLin + Ls;

                // Soft re-tonemap шалгалт
                outLin = outLin / (1.0f + outLin * 0.15f); 

                // LDR clamp
                outLin = glm::clamp(outLin, 0.0f, 1.0f);

                // Linear -> sRGB буцаана
                glm::vec3 outSrgb = linear_to_srgb(outLin);
                dst_raw[row + x]  = srgb01_to_color(outSrgb);
            }
        }
    };

    if (job_system) {
        wg.reset();
        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg.add(1);
                job_system->submit({[=, &wg]() {
                    process_tile(tx, ty);
                    wg.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }
        wg.wait();
    } else {
        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                process_tile(tx, ty);
            }
        }
    }
}


// ==========================================
// SCENE STATE
// ==========================================

class DemoScene : public shs::AbstractSceneState
{
public:
    DemoScene(shs::Canvas* canvas, Viewer* viewer, const shs::Texture2D* car_tex, const CubeMapLDR* sky, const EnvIBL* ibl)
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

    shs::Canvas*      canvas;
    Viewer*           viewer;

    const CubeMapLDR* sky = nullptr;
    const EnvIBL*     ibl = nullptr;

    FloorPlane*       floor;
    SubaruObject*     car;
    MonkeyObject*     monkey;

    std::vector<shs::AbstractObject3D*> scene_objects;
};

// ==========================================
// RENDERER SYSTEM (Shadow + Camera + Shafts + MotionBlur + Skybox + PBR + PCSS)
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(DemoScene* scene, shs::Job::ThreadedPriorityJobSystem* job_sys)
        : scene(scene), job_system(job_sys)
    {
        rt = new RT_ColorDepthMotion(
            CANVAS_WIDTH, CANVAS_HEIGHT,
            scene->viewer->camera->z_near,
            scene->viewer->camera->z_far,
            shs::Color{20,20,25,255}
        );

        shafts_out = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
        mb_out     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

        shadow = new ShadowMap(SHADOW_MAP_SIZE, SHADOW_MAP_SIZE);

        shafts_params = LightShaftParams(); // default
        shafts_params.enable        = true;
        shafts_params.steps         = 28;
        shafts_params.min_dist      = 1.0f;
        shafts_params.max_dist      = 110.0f;

        //shafts_params.base_density  = 0.18f;
        shafts_params.base_density *= 0.85f;
        shafts_params.height_falloff= 0.12f;

        shafts_params.sigma_s       = 0.030f;
        shafts_params.sigma_t       = 0.065f;   // >= sigma_s

        shafts_params.g             = 0.86f;    
        shafts_params.intensity     = 0.22f;

        shafts_params.use_shadow    = true;
        shafts_params.shadow_bias   = 0.0055f;
        shafts_params.shadow_pcf_2x2= true;


        has_prev_cam = false;
        prev_view    = glm::mat4(1.0f);
        prev_proj    = glm::mat4(1.0f);
    }

    ~RendererSystem()
    {
        delete rt;
        delete shafts_out;
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
        // PASS1: Camera render -> RT_ColorDepthMotion
        // -----------------------
        rt->clear(shs::Color{20,20,25,255});

        // Skybox background fill
        if (scene->sky && scene->sky->valid()) {
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

                        
                        // Floor (PBR хуванцарлаг)
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

                            u.shadow          = shadow;
                            u.sky             = scene->sky;
                            u.ibl             = scene->ibl;

                            u.mat.baseColor_srgb = shs::Color{120,122,128,255};
                            u.mat.metallic       = 0.00f;
                            u.mat.roughness      = 0.70f;
                            u.mat.ao             = 1.00f;

                            u.albedo          = nullptr;
                            u.use_texture     = false;

                            u.ibl_diffuse_intensity   = 0.55f;
                            u.ibl_specular_intensity  = 0.30f;
                            u.ibl_reflection_strength = 0.10f;

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

                                draw_triangle_tile_color_depth_motion(
                                    *rt,
                                    tri_v, tri_n, tri_uv,
                                    [&u](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
                                        return vertex_shader_full(p, n, uv, u);
                                    },
                                    [&u](const VaryingsFull& vin, int px, int py) {
                                        return fragment_shader_pbr(vin, u, px, py);
                                    },
                                    t_min, t_max
                                );
                            }
                        }

                        
                        for (shs::AbstractObject3D* obj : scene->scene_objects)
                        {
                            // Car
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

                                u.shadow          = shadow;
                                u.sky             = scene->sky;
                                u.ibl             = scene->ibl;

                                u.mat.baseColor_srgb = shs::Color{200,200,200,255};
                                u.mat.metallic       = 0.00f;
                                u.mat.roughness      = 0.22f;
                                u.mat.ao             = 1.00f;

                                u.albedo      = car->albedo;
                                u.use_texture = (car->albedo && car->albedo->valid());

                                u.ibl_diffuse_intensity   = 0.42f;
                                u.ibl_specular_intensity  = 0.65f;
                                u.ibl_reflection_strength = 0.85f;

                                const auto& v = car->geometry->triangles;
                                const auto& n = car->geometry->normals;
                                const auto& t = car->geometry->uvs;

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { t[i], t[i+1], t[i+2] };

                                    draw_triangle_tile_color_depth_motion(
                                        *rt,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin, int px, int py) {
                                            return fragment_shader_pbr(vin, u, px, py);
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

                                u.shadow          = shadow;
                                u.sky             = scene->sky;
                                u.ibl             = scene->ibl;

                                u.mat.baseColor_srgb = shs::Color{240,195,75,255};
                                u.mat.metallic       = 0.95f;
                                u.mat.roughness      = 0.20f;
                                u.mat.ao             = 1.00f;

                                u.albedo      = nullptr;
                                u.use_texture = false;

                                u.ibl_diffuse_intensity   = 0.25f;
                                u.ibl_specular_intensity  = 0.55f;
                                u.ibl_reflection_strength = 0.80f;

                                const auto& v = mk->geometry->triangles;
                                const auto& n = mk->geometry->normals;
                                static const glm::vec2 uv0(0.0f);

                                for (size_t i = 0; i < v.size(); i += 3) {
                                    std::vector<glm::vec3> tri_v = { v[i], v[i+1], v[i+2] };
                                    std::vector<glm::vec3> tri_n = { n[i], n[i+1], n[i+2] };
                                    std::vector<glm::vec2> tri_t = { uv0, uv0, uv0 };

                                    draw_triangle_tile_color_depth_motion(
                                        *rt,
                                        tri_v, tri_n, tri_t,
                                        [&u](const glm::vec3& p, const glm::vec3& nn, const glm::vec2& uv) {
                                            return vertex_shader_full(p, nn, uv, u);
                                        },
                                        [&u](const VaryingsFull& vin, int px, int py) {
                                            return fragment_shader_pbr(vin, u, px, py);
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
        // PASS1.5: Light Shafts
        // -----------------------
        {
            glm::mat4 curr_view = scene->viewer->camera->view_matrix;
            glm::mat4 curr_proj = scene->viewer->camera->projection_matrix;
            glm::mat4 inv_vp = glm::inverse(curr_proj * curr_view);

            // shadow raw pointer ашиглагдаж байгаа
            const float* shadow_raw = shadow->depth.raw();

            light_shafts_pass(
                rt->color,
                rt->depth,
                *shafts_out,
                scene->viewer->position,
                inv_vp,
                LIGHT_DIR_WORLD,
                light_vp,
                shadow_raw,
                shadow->w,
                shadow->h,
                shafts_params,
                job_system,
                wg_shafts
            );
        }

        // -----------------------
        // PASS2: Combined Motion Blur (source = shafts_out)
        // -----------------------
        glm::mat4 curr_view = scene->viewer->camera->view_matrix;
        glm::mat4 curr_proj = scene->viewer->camera->projection_matrix;

        if (!has_prev_cam) {
            prev_view    = curr_view;
            prev_proj    = curr_proj;
            has_prev_cam = true;
        }

        combined_motion_blur_pass(
            *shafts_out,
            rt->depth,
            rt->motion,
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

    RT_ColorDepthMotion* rt;
    shs::Canvas* shafts_out;
    shs::Canvas* mb_out;

    ShadowMap* shadow;

    LightShaftParams shafts_params;

    shs::Job::WaitGroup wg_shadow;
    shs::Job::WaitGroup wg_cam;
    shs::Job::WaitGroup wg_mb;
    shs::Job::WaitGroup wg_sky;
    shs::Job::WaitGroup wg_shafts;

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

    // Skybox cubemap load (LDR)
    CubeMapLDR sky = load_cubemap_water_scene("./images/skybox/water_scene");
    if (!sky.valid()) {
        std::cout << "Warning: Skybox cubemap load failed (images/skybox/water_scene/*.jpg)" << std::endl;
    }

    // IBL precompute (програм эхлэхэд нэг удаа)
    EnvIBL ibl;
    if (sky.valid()) {
        std::cout << "STATUS : IBL precompute started..." << std::endl;

        ibl.env_radiance = cubemap_ldr_to_linear(sky);
        if (ibl.env_radiance.valid()) {

            std::cout << "STATUS : IBL irradiance building..."
                      << " | size=" << IBL_IRR_SIZE
                      << " | samples=" << IBL_IRR_SAMPLES
                      << std::endl;

            ibl.env_irradiance = build_env_irradiance(ibl.env_radiance, IBL_IRR_SIZE, IBL_IRR_SAMPLES);

            int specBase = std::min(IBL_SPEC_BASE_CAP, ibl.env_radiance.size);

            std::cout << "STATUS : IBL specular prefilter building..."
                      << " | base=" << specBase
                      << " | mips=" << IBL_SPEC_MIPCOUNT
                      << " | samples=" << IBL_SPEC_SAMPLES
                      << std::endl;

            ibl.env_prefiltered_spec = build_env_prefiltered_specular(ibl.env_radiance, specBase, IBL_SPEC_MIPCOUNT, IBL_SPEC_SAMPLES);
        }

        if (!ibl.valid()) {
            std::cout << "Warning: IBL precompute failed (falling back to direct only)." << std::endl;
        } else {
            std::cout << "STATUS : IBL precompute done." << std::endl;
        }
    }

    // Scene
    Viewer*    viewer    = new Viewer(glm::vec3(0.0f, 10.0f, -42.0f), 55.0f);
    DemoScene* scene     = new DemoScene(screen_canvas, viewer, &car_tex, &sky, (ibl.valid() ? &ibl : nullptr));

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
                "PBR (GGX+IBL) + PCSS Soft Shadow + Shafts + MotionBlur | FPS: " + std::to_string(frames) +
                " | Threads: " + std::to_string(THREAD_COUNT) +
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
