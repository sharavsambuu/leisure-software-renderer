/*
    hello_glowing_star.cpp

    SHS - Glowing Golden Star Demo (CPU Software Renderer)
    - Pass 0: Render to RT_ColorDepthVelocitySpec (Color + Depth + Velocity + SpecMask)
    - Pass 1: Per-object Motion Blur (John Chapman style): rt.color + rt.velocity -> mb_out
    - Pass 2: DOF (Auto-focus + blur composite): mb_out + rt.depth -> dof_out
    - Pass 3: Specular Glow/Bloom (from spec mask): dof_out + rt.spec -> bloom_out
    - Pass 4: Pseudo Lens Flare (Chapman-ish): bloom_out -> flare_out
    - Composite: dof_out + bloom_out + flare_out -> final_out
    - Present: final_out -> SDL

    NOTE : (Coordinate convention)
    - Screen space origin: top-left (SDL), +Y down
    - shs::Canvas origin : bottom-left, +Y up
    - Depth & velocity stored in CANVAS coordinates (bottom-left) to avoid inverting bugs.

    References:
    - https://john-chapman-graphics.blogspot.com/2013/02/pseudo-lens-flare.html

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
// STAR CONFIG
// ===============================
static const glm::vec3 STAR_BASE_POS    = glm::vec3(0.0f, 4.0f, 18.0f);
static const float     STAR_SCALE       = 6.8f;
static const float     STAR_WOBBLE_AMP  = 7.2f;
static const float     STAR_WOBBLE_SPD  = 0.2f;
static const float     STAR_ROT_DEG_SPD = 25.0f;

// Golden base color
static const shs::Color STAR_COLOR = shs::Color{ 255, 215, 100, 255 };

// ===============================
// MOTION BLUR CONFIG
// ===============================
static const int   MB_SAMPLES    = 8;      // 6..12
static const float MB_STRENGTH   = 1.5f;   // 0.5..2.0
static const float MB_MAX_PIXELS = 30.0f;  // clamp in pixels (canvas coords)

// ===============================
// DOF CONFIG
// ===============================
static const bool ENABLE_DOF       = true;
static const int  BLUR_ITERATIONS  = 3;
static const int  AUTOFOCUS_RADIUS = 6;
static float      DOF_RANGE        = 22.0f;
static float      DOF_MAXBLUR      = 0.80f;

// ===============================
// BLOOM / GLOW CONFIG (SPEC-DRIVEN)
// ===============================
static const bool  ENABLE_BLOOM         = true;
static const float SPEC_GLOW_THRESHOLD  = 0.139f;  // 0..1, lower = more glow
static const float SPEC_GLOW_INTENSITY  = 13.25f;  // multiplier
static const int   BLOOM_BLUR_ITERS     = 10;      // keep small for CPU

// ===============================
// PSEUDO LENS FLARE CONFIG
// ===============================
static const bool  ENABLE_FLARE       = true;
static const int   FLARE_GHOSTS       = 3;
static const float FLARE_INTENSITY    = 0.55f;
static const float FLARE_HALO_INTENS  = 0.35f;
static const float FLARE_CHROMA_SHIFT = 0.8f;

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

static inline float luma_from_color(const shs::Color& c)
{
    float r = float(c.r) / 255.0f;
    float g = float(c.g) / 255.0f;
    float b = float(c.b) / 255.0f;
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

static inline shs::Color add_color_clamped(const shs::Color& a, const shs::Color& b)
{
    return shs::Color{
        (uint8_t)clampi(int(a.r) + int(b.r), 0, 255),
        (uint8_t)clampi(int(a.g) + int(b.g), 0, 255),
        (uint8_t)clampi(int(a.b) + int(b.b), 0, 255),
        255
    };
}

static inline shs::Color mul_color(const shs::Color& c, float k)
{
    return shs::Color{
        (uint8_t)clampi(int(float(c.r) * k), 0, 255),
        (uint8_t)clampi(int(float(c.g) * k), 0, 255),
        (uint8_t)clampi(int(float(c.b) * k), 0, 255),
        255
    };
}

// ==========================================
// STAR GEOMETRY (LOW POLY 5-POINT PRISM)
// - builds triangles + per-triangle face normals (duplicated vertices)
// ==========================================

static inline glm::vec3 face_normal(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
    return glm::normalize(glm::cross(b - a, c - a));
}

static void build_star_prism_lowpoly(
    std::vector<glm::vec3>& out_triangles,
    std::vector<glm::vec3>& out_normals,
    float R = 1.0f,
    float r = 0.62f,
    float thickness = 0.70f)
{
    out_triangles.clear();
    out_normals.clear();

    const float zf = +0.5f * thickness;
    const float zb = -0.5f * thickness;

    glm::vec3 ring[10];
    for (int i = 0; i < 10; i++) {
        float a = glm::radians(90.0f) + float(i) * glm::radians(36.0f); // 360/10
        float rad = (i % 2 == 0) ? R : r;

        ring[i] = glm::vec3(std::cos(a) * rad, std::sin(a) * rad, 0.0f);
    }

    glm::vec3 cf(0.0f, 0.0f, zf);
    glm::vec3 cb(0.0f, 0.0f, zb);

    auto push_tri = [&](const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) {
        out_triangles.push_back(a);
        out_triangles.push_back(b);
        out_triangles.push_back(c);

        glm::vec3 n = face_normal(a, b, c);
        out_normals.push_back(n);
        out_normals.push_back(n);
        out_normals.push_back(n);
    };

    // Front cap: 10 faceted triangles
    for (int i = 0; i < 10; i++) {
        int j = (i + 1) % 10;
        push_tri(cf, ring[i], ring[j]);
    }

    // Back cap: reverse winding so it faces outward
    for (int i = 0; i < 10; i++) {
        int j = (i + 1) % 10;
        push_tri(cb, ring[j], ring[i]);
    }
}


// ==========================================
// UNIFORMS & SHADERS (Blinn-Phong + Spec Mask)
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

struct VaryingsStar
{
    glm::vec4 position;      // current clip
    glm::vec4 prev_position; // previous clip
    glm::vec3 world_pos;
    glm::vec3 normal;
    float     view_z;
};

struct FragOut
{
    shs::Color color;
    float      spec01; // 0..1 (for glow/flare)
};

static VaryingsStar star_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    VaryingsStar out;
    out.position       = u.mvp * glm::vec4(aPos, 1.0f);
    out.prev_position  = u.prev_mvp * glm::vec4(aPos, 1.0f);

    out.world_pos      = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal         = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);

    glm::vec4 view_pos = u.view * u.model * glm::vec4(aPos, 1.0f);
    out.view_z         = view_pos.z; // +z forward in convention

    return out;
}

static FragOut star_fragment_shader(const VaryingsStar& in, const Uniforms& u)
{
    glm::vec3 norm     = glm::normalize(in.normal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - in.world_pos);

    float ambientStrength = 0.20f;
    glm::vec3 ambient     = ambientStrength * glm::vec3(1.0f);

    float diff        = glm::max(glm::dot(norm, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f);

    glm::vec3 halfwayDir   = glm::normalize(lightDir + viewDir);
    float specularStrength = 3.85f;
    float shininess        = 256.0f;

    float spec   = glm::pow(glm::max(glm::dot(norm, halfwayDir), 0.0f), shininess);
    float spec01 = clampf(specularStrength * spec, 0.0f, 1.0f);

    glm::vec3 objectColorVec = glm::vec3(u.color.r, u.color.g, u.color.b) / 255.0f;

    // Gold-ish spec tint (helps flare look warm)
    glm::vec3 specTint = glm::vec3(1.0f, 0.90f, 0.55f);

    glm::vec3 result =
        (ambient + diffuse) * (objectColorVec * 0.85f) +
        (spec01 * specTint);

    result = glm::clamp(result, 0.0f, 1.0f);

    FragOut out;
    out.color = shs::Color{
        (uint8_t)(result.r * 255),
        (uint8_t)(result.g * 255),
        (uint8_t)(result.b * 255),
        255
    };
    out.spec01 = spec01;
    return out;
}

// ==========================================
// RT: Color + Depth + Velocity + SpecMask
// ==========================================

struct RT_ColorDepthVelocitySpec
{
    RT_ColorDepthVelocitySpec(int W, int H, float zn, float zf, shs::Color clear_col)
        : color(W, H, clear_col),
          depth(W, H, zn, zf),
          velocity(W, H, glm::vec2(0.0f)),
          spec(W, H, 0.0f)
    {
        (void)zn; (void)zf;
        clear(clear_col);
    }

    inline void clear(shs::Color c)
    {
        color.buffer().clear(c);
        depth.clear();
        velocity.clear(glm::vec2(0.0f));
        spec.clear(0.0f);
    }

    inline int width() const  { return color.get_width(); }
    inline int height() const { return color.get_height(); }

    shs::Canvas            color;
    shs::ZBuffer           depth;
    shs::Buffer<glm::vec2> velocity; // canvas coords (x right, y up), pixels
    shs::Buffer<float>     spec;     // 0..1
};

// ==========================================
// RASTER: triangle -> color+depth+velocity+spec
// (velocity/spec stored in CANVAS coords)
// ==========================================

static inline glm::vec2 clip_to_screen_xy(const glm::vec4& clip, int w, int h)
{
    glm::vec3 s = shs::Canvas::clip_to_screen(clip, w, h); // screen coords (x right, y down)
    return glm::vec2(s.x, s.y);
}

static void draw_triangle_color_depth_velocity_spec(
    RT_ColorDepthVelocitySpec &rt,
    const std::vector<glm::vec3> &vertices,
    const std::vector<glm::vec3> &normals,
    std::function<VaryingsStar(const glm::vec3&, const glm::vec3&)> vertex_shader,
    std::function<FragOut(const VaryingsStar&)> fragment_shader)
{
    int w = rt.color.get_width();
    int h = rt.color.get_height();

    VaryingsStar vout[3];
    glm::vec3 screen_coords[3];

    for (int i = 0; i < 3; i++) {
        vout[i] = vertex_shader(vertices[i], normals[i]);
        screen_coords[i] = shs::Canvas::clip_to_screen(vout[i].position, w, h);
    }

    int max_x = w - 1;
    int max_y = h - 1;

    std::vector<glm::vec2> v2d = {
        glm::vec2(screen_coords[0]),
        glm::vec2(screen_coords[1]),
        glm::vec2(screen_coords[2])
    };

    glm::vec2 bboxmin(max_x, max_y);
    glm::vec2 bboxmax(0, 0);

    for (int i = 0; i < 3; i++) {
        bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, v2d[i]));
        bboxmax = glm::min(glm::vec2(max_x, max_y), glm::max(bboxmax, v2d[i]));
    }

    if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

    float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) - (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
    if (area <= 0) return; // backface cull 

    for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
        for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
            if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

            float z = bc.x * vout[0].view_z + bc.y * vout[1].view_z + bc.z * vout[2].view_z;

            // screen py -> canvas y
            int cy = (h - 1) - py;

            if (rt.depth.test_and_set_depth(px, cy, z)) {

                VaryingsStar interpolated;
                interpolated.normal  = glm::normalize(bc.x * vout[0].normal  + bc.y * vout[1].normal  + bc.z * vout[2].normal);
                interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                interpolated.view_z  = z;

                interpolated.position      = bc.x * vout[0].position      + bc.y * vout[1].position      + bc.z * vout[2].position;
                interpolated.prev_position = bc.x * vout[0].prev_position + bc.y * vout[1].prev_position + bc.z * vout[2].prev_position;

                // velocity from clip->screen (y down) -> canvas (y up)
                glm::vec2 curr_s   = clip_to_screen_xy(interpolated.position, w, h);
                glm::vec2 prev_s   = clip_to_screen_xy(interpolated.prev_position, w, h);
                glm::vec2 v_screen = curr_s - prev_s;                    // y down
                glm::vec2 v_canvas = glm::vec2(v_screen.x, -v_screen.y); // y up

                float len = glm::length(v_canvas);
                if (len > MB_MAX_PIXELS && len > 0.0001f) {
                    v_canvas = v_canvas * (MB_MAX_PIXELS / len);
                }

                rt.velocity.at(px, cy) = v_canvas;

                FragOut fo = fragment_shader(interpolated);
                rt.spec.at(px, cy) = fo.spec01;

                rt.color.draw_pixel_screen_space(px, py, fo.color);
            }
        }
    }
}

// ==========================================
// PASS 1: PER-OBJECT MOTION BLUR (post)
// ==========================================

static void motion_blur_pass(
    const shs::Canvas& src,
    const shs::Buffer<glm::vec2>& velocity,
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

                        glm::vec2 v = velocity.at(x, y) * strength;
                        float vlen = glm::length(v);

                        if (vlen < 0.001f || samples <= 1) {
                            dst.draw_pixel(x, y, src.get_color_at(x, y));
                            continue;
                        }

                        glm::vec2 dir = v / vlen;

                        float r = 0, g = 0, b = 0;
                        float wsum = 0.0f;

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
// AUTOFOCUS + DOF COMPOSITE
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
// PASS: SPECULAR GLOW / BLOOM (from spec mask)
// - builds a bright buffer tinted gold, blurs it, then add to base
// ==========================================

static void build_spec_bright_pass(
    const shs::Buffer<float>& spec01,
    shs::Canvas& bright_out,
    float threshold,
    float intensity,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    int W = bright_out.get_width();
    int H = bright_out.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wg.reset();

    const glm::vec3 gold = glm::vec3(1.0f, 0.88f, 0.45f);

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

                        float s = spec01.at(x, y); // already canvas coords
                        float v = (s - threshold) / std::max(1e-6f, (1.0f - threshold));
                        v = smoothstep01(v);
                        v *= intensity;

                        // Convert to RGB glow (tinted)
                        float r = 255.0f * gold.r * v;
                        float g = 255.0f * gold.g * v;
                        float b = 255.0f * gold.b * v;

                        bright_out.draw_pixel(x, y, color_from_rgbaf(r, g, b, 255.0f));
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

static void additive_composite_pass(
    const shs::Canvas& base,
    const shs::Canvas& add,
    shs::Canvas& out,
    float add_strength,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    int W = base.get_width();
    int H = base.get_height();

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
                        shs::Color a = base.get_color_at(x, y);
                        shs::Color b = mul_color(add.get_color_at(x, y), add_strength);
                        out.draw_pixel(x, y, add_color_clamped(a, b));
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

// ==========================================
// PASS: PSEUDO LENS FLARE (Chapman-ish, CPU)
// - driven purely by bright buffer (bloom source)
// ==========================================

static void pseudo_lens_flare_pass(
    const shs::Canvas& bright,
    shs::Canvas& flare_out,
    float intensity,
    float halo_intensity,
    float chroma_shift_px,
    shs::Job::ThreadedPriorityJobSystem* job_system,
    shs::Job::WaitGroup& wg)
{
    int W = bright.get_width();
    int H = bright.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    wg.reset();

    auto sample = [&](float fx, float fy) -> shs::Color {
        int sx = clampi((int)std::round(fx), 0, W - 1);
        int sy = clampi((int)std::round(fy), 0, H - 1);
        return bright.get_color_at(sx, sy);
    };

    const float cx = 0.5f * float(W);
    const float cy = 0.5f * float(H);

    // Ghost scales (tuned for a single bright star)
    float ghost_scales[FLARE_GHOSTS] = { 0.55f, 0.85f, 1.25f};

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

                        // Vector from center (pixel space, canvas coords)
                        float dx = float(x) - cx;
                        float dy = float(y) - cy;

                        // Normalized-ish for weights
                        float dist = std::sqrt(dx*dx + dy*dy);
                        float nd   = (dist / std::max(1.0f, std::min(cx, cy)));
                        float vign = 1.0f - smoothstep01(nd);

                        float rr = 0, gg = 0, bb = 0;

                        // Ghosts: sample mirrored across center, scaled by factors
                        for (int i = 0; i < FLARE_GHOSTS; i++) {
                            float s = ghost_scales[i];

                            float sx = cx - dx * s;
                            float sy = cy - dy * s;

                            // tiny chroma offsets along radial direction
                            glm::vec2 dir(0.0f);
                            if (dist > 1e-4f) dir = glm::vec2(dx, dy) / dist;

                            shs::Color cR = sample(sx + dir.x * chroma_shift_px, sy + dir.y * chroma_shift_px);
                            shs::Color cG = sample(sx, sy);
                            shs::Color cB = sample(sx - dir.x * chroma_shift_px, sy - dir.y * chroma_shift_px);

                            float w = (0.45f + 0.55f * (1.0f - float(i) / float(FLARE_GHOSTS)));
                            rr += w * float(cR.r);
                            gg += w * float(cG.g);
                            bb += w * float(cB.b);
                        }

                        // Halo: sample near a ring around center, aligned with dx/dy direction
                        {
                            float halo_scale = 0.35f;
                            float sx = cx - dx * halo_scale;
                            float sy = cy - dy * halo_scale;

                            shs::Color hC = sample(sx, sy);

                            // Ring-ish weight (stronger mid radius)
                            float ring = std::exp(-8.0f * (nd - 0.35f) * (nd - 0.35f));
                            rr += halo_intensity * ring * float(hC.r);
                            gg += halo_intensity * ring * float(hC.g);
                            bb += halo_intensity * ring * float(hC.b);
                        }

                        rr *= intensity * vign;
                        gg *= intensity * vign;
                        bb *= intensity * vign;

                        flare_out.draw_pixel(x, y, color_from_rgbaf(rr, gg, bb, 255.0f));
                    }
                }

                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }

    wg.wait();
}

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
// STAR OBJECT (prev_mvp for per-object velocity)
// ==========================================

class GlowingStarObject : public shs::AbstractObject3D
{
public:
    GlowingStarObject()
    {
        build_star_prism_lowpoly(triangles, normals, 1.0f, 0.45f, 0.22f);

        base_position = STAR_BASE_POS;
        position      = base_position;
        scale         = glm::vec3(STAR_SCALE);

        time_accum     = 0.0f;
        rotation_angle = 0.0f;

        has_prev_mvp = false;
        prev_mvp     = glm::mat4(1.0f);
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

        float y = std::sin(time_accum * STAR_WOBBLE_SPD) * STAR_WOBBLE_AMP;
        position = base_position + glm::vec3(0.0f, y, 0.0f);

        rotation_angle += STAR_ROT_DEG_SPD * delta_time;
        if (rotation_angle > 360.0f) rotation_angle -= 360.0f;
    }

    void render() override {}

    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;

    glm::vec3 base_position;
    glm::vec3 position;
    glm::vec3 scale;

    float time_accum;
    float rotation_angle;

    bool      has_prev_mvp;
    glm::mat4 prev_mvp;
};

// ==========================================
// SYSTEMS
// ==========================================

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(Viewer* viewer, GlowingStarObject* star) : viewer(viewer), star(star) {}
    void process(float delta_time) override
    {
        viewer->update();
        star->update(delta_time);
    }
private:
    Viewer* viewer;
    GlowingStarObject* star;
};

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(Viewer* viewer, GlowingStarObject* star, shs::Job::ThreadedPriorityJobSystem* job_sys, RT_ColorDepthVelocitySpec* rt)
        : viewer(viewer), star(star), job_system(job_sys), rt(rt)
    {
        light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));
    }

    void process(float delta_time) override
    {
        (void)delta_time;

        rt->clear(shs::Color{20, 20, 25, 255});

        glm::mat4 view = viewer->camera->view_matrix;
        glm::mat4 proj = viewer->camera->projection_matrix;

        glm::mat4 model = star->get_world_matrix();
        glm::mat4 mvp   = proj * view * model;

        glm::mat4 prev_mvp = mvp;
        if (star->has_prev_mvp) prev_mvp = star->prev_mvp;

        Uniforms uniforms;
        uniforms.model      = model;
        uniforms.view       = view;
        uniforms.mvp        = mvp;
        uniforms.prev_mvp   = prev_mvp;
        uniforms.light_dir  = light_direction;
        uniforms.camera_pos = viewer->position;
        uniforms.color      = STAR_COLOR;

        // Render all triangles (single object; keep it simple, no tiling needed here)
        for (size_t i = 0; i < star->triangles.size(); i += 3)
        {
            std::vector<glm::vec3> tri_verts = { star->triangles[i], star->triangles[i+1], star->triangles[i+2] };
            std::vector<glm::vec3> tri_norms = { star->normals[i],   star->normals[i+1],   star->normals[i+2]   };

            draw_triangle_color_depth_velocity_spec(
                *rt,
                tri_verts,
                tri_norms,
                [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                    return star_vertex_shader(p, n, uniforms);
                },
                [&uniforms](const VaryingsStar& v) {
                    return star_fragment_shader(v, uniforms);
                }
            );
        }

        // Commit prev_mvp after render
        glm::mat4 view2 = viewer->camera->view_matrix;
        glm::mat4 proj2 = viewer->camera->projection_matrix;

        glm::mat4 model2 = star->get_world_matrix();
        star->prev_mvp = proj2 * view2 * model2;
        star->has_prev_mvp = true;
    }

private:
    Viewer* viewer;
    GlowingStarObject* star;
    shs::Job::ThreadedPriorityJobSystem* job_system;
    RT_ColorDepthVelocitySpec* rt;

    glm::vec3 light_direction;
};

class SystemProcessor
{
public:
    SystemProcessor(Viewer* viewer, GlowingStarObject* star, shs::Job::ThreadedPriorityJobSystem* job_sys, RT_ColorDepthVelocitySpec* rt)
    {
        command_processor = new shs::CommandProcessor();
        logic_system      = new LogicSystem(viewer, star);
        renderer_system   = new RendererSystem(viewer, star, job_sys, rt);
    }

    ~SystemProcessor()
    {
        delete command_processor;
        delete logic_system;
        delete renderer_system;
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

    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas *screen_canvas  = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *screen_surface = screen_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, screen_surface);

    Viewer *viewer = new Viewer(glm::vec3(0.0f, 6.0f, -28.0f), 50.0f);
    GlowingStarObject *star = new GlowingStarObject();

    // RT (Pass 0)
    RT_ColorDepthVelocitySpec rt_scene(CANVAS_WIDTH, CANVAS_HEIGHT, viewer->camera->z_near, viewer->camera->z_far, shs::Color{20,20,25,255});

    // Pass 1 output (motion blur)
    shs::Canvas mb_out(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    // DOF buffers
    shs::Canvas sharp_copy(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas blur_ping (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas blur_pong (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});
    shs::Canvas dof_out   (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    // Bloom buffers
    shs::Canvas bright_spec(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0,0,0,255});
    shs::Canvas bloom_ping (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0,0,0,255});
    shs::Canvas bloom_pong (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0,0,0,255});
    shs::Canvas bloom_out  (CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0,0,0,255});

    // Flare buffers
    shs::Canvas flare_out(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{0,0,0,255});

    // Final composite
    shs::Canvas final_out(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20,20,25,255});

    SystemProcessor *sys = new SystemProcessor(viewer, star, job_system, &rt_scene);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();
    bool is_dragging = false;

    shs::Job::WaitGroup wg_mb;
    shs::Job::WaitGroup wg_blur;
    shs::Job::WaitGroup wg_dof;
    shs::Job::WaitGroup wg_spec;
    shs::Job::WaitGroup wg_bloom;
    shs::Job::WaitGroup wg_flare;
    shs::Job::WaitGroup wg_comp;

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

        // Pass 1: motion blur
        motion_blur_pass(rt_scene.color, rt_scene.velocity, mb_out, MB_SAMPLES, MB_STRENGTH, job_system, wg_mb);

        // Pass 2: DOF
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

        // Pass 3: Specular bloom (from rt_scene.spec)
        if (ENABLE_BLOOM)
        {
            build_spec_bright_pass(rt_scene.spec, bright_spec, SPEC_GLOW_THRESHOLD, SPEC_GLOW_INTENSITY, job_system, wg_spec);

            // blur bright_spec -> bloom_pong
            bloom_pong.buffer() = bright_spec.buffer();

            for (int i = 0; i < BLOOM_BLUR_ITERS; i++)
            {
                gaussian_blur_pass(bloom_pong, bloom_ping, true,  job_system, wg_bloom);
                gaussian_blur_pass(bloom_ping, bloom_pong, false, job_system, wg_bloom);
            }

            bloom_out.buffer() = bloom_pong.buffer();
        }
        else
        {
            bloom_out.buffer().clear(shs::Color{0,0,0,255});
        }

        // Pass 4: Pseudo lens flare driven by bloom_out (already bright-only)
        if (ENABLE_FLARE)
        {
            pseudo_lens_flare_pass(
                bloom_out,
                flare_out,
                FLARE_INTENSITY,
                FLARE_HALO_INTENS,
                FLARE_CHROMA_SHIFT,
                job_system,
                wg_flare
            );
        }
        else
        {
            flare_out.buffer().clear(shs::Color{0,0,0,255});
        }

        // Composite: dof_out + bloom_out + flare_out
        // (two additive composites to reuse existing function)
        additive_composite_pass(dof_out, bloom_out, final_out, 1.0f, job_system, wg_comp);
        additive_composite_pass(final_out, flare_out, final_out, 1.0f, job_system, wg_comp);

        // Present
        screen_canvas->buffer() = final_out.buffer();
        shs::Canvas::copy_to_SDLSurface(screen_surface, screen_canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    delete sys;
    delete star;
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
