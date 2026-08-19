#define SDL_MAIN_HANDLED

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include <cstdint>
#include <iomanip>
#include <sstream>

#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

#include "shs_renderer.hpp"

// ============================================================================
// CONFIGURATION & CONSTANTS
// ============================================================================
static const int WINDOW_WIDTH             = 1280;
static const int WINDOW_HEIGHT            = 720;
static const int CANVAS_WIDTH             = 1280;
static const int CANVAS_HEIGHT            = 720;

static const int THREAD_COUNT             = 16;
static const int TILE_SIZE_X              = 80;
static const int TILE_SIZE_Y              = 80;

static const float Z_NEAR                 = 1.0f;    // Crisp depth precision (avoids near-plane precision collapse)
static const float Z_FAR                  = 2000.0f;
// Lowest landing-gear vertex is y=-0.65 in airplane-local space.
static const float PLANE_GROUND_CLEARANCE = 0.67f;
static const float THROTTLE_STEP          = 0.05f;  // per mouse-wheel notch

static const glm::vec3 LIGHT_DIR_WORLD    = glm::normalize(glm::vec3(0.4f, -0.8f, 0.45f));

// ============================================================================
// CLEAN LOW-POLY TRIANGLE STRUCTURES
// ============================================================================
struct LowPolyTriangle {
    glm::vec3  p0;
    glm::vec3  p1;
    glm::vec3  p2;
    shs::Color color;
    float      depth_bias;

    LowPolyTriangle(glm::vec3 a, glm::vec3 b, glm::vec3 c, shs::Color col, float bias = 0.0f)
        : p0(a), p1(b), p2(c), color(col), depth_bias(bias) {}
};

// ============================================================================
// PROCEDURAL LOW-POLY SYNTHETIC TERRAIN (Clean 24x24 grid)
// ============================================================================
class SyntheticTerrain {
public:
    static float get_height(float x, float z) {
        // Keep the full runway and its safety margin genuinely flat.
        // The runway itself occupies X=[-16,16], Z=[-220,220].
        float dx    = std::max(0.0f, std::abs(x) - 25.0f);
        float dz    = std::max(0.0f, std::abs(z) - 240.0f);
        float dist  = std::sqrt(dx * dx + dz * dz);
        float blend = glm::clamp(dist / 60.0f, 0.0f, 1.0f);
        blend       = blend * blend * (3.0f - 2.0f * blend); // smoothstep

        // Clean low-frequency rolling terrain
        float h1 = std::sin(x * 0.006f) * std::cos(z * 0.006f) * 40.0f;
        float h2 = std::sin(x * 0.015f + 1.0f) * std::cos(z * 0.015f + 0.5f) * 18.0f;

        // Distant perimeter hills
        float r = std::sqrt(x * x + z * z);
        float perimeter = (r > 380.0f) ? (r - 380.0f) * 0.18f * (r - 380.0f) * 0.02f : 0.0f;

        return (h1 + h2 + perimeter) * blend;
    }

    static std::vector<LowPolyTriangle> build_mesh() {
        std::vector<LowPolyTriangle> tris;
        const int   GRID = 24;
        const float SIZE = 1600.0f;
        const float STEP = SIZE / (float)GRID;
        const float HALF = SIZE * 0.5f;

        shs::Color grass_c1 = shs::Color{85, 155, 65, 255};
        shs::Color grass_c2 = shs::Color{70, 140, 55, 255};
        shs::Color hill_c   = shs::Color{135, 125, 110, 255};
        shs::Color runway_c = shs::Color{50, 55, 60, 255};
        shs::Color stripe_c = shs::Color{240, 240, 240, 255};

        for (int iz = 0; iz < GRID; ++iz) {
            float z0 = -HALF + (float)iz * STEP;
            float z1 = z0 + STEP;

            for (int ix = 0; ix < GRID; ++ix) {
                float x0 = -HALF + (float)ix * STEP;
                float x1 = x0 + STEP;

                float y00 = get_height(x0, z0);
                float y10 = get_height(x1, z0);
                float y01 = get_height(x0, z1);
                float y11 = get_height(x1, z1);

                glm::vec3 p00(x0, y00, z0);
                glm::vec3 p10(x1, y10, z0);
                glm::vec3 p01(x0, y01, z1);
                glm::vec3 p11(x1, y11, z1);

                float avg_y    = (y00 + y10 + y01 + y11) * 0.25f;
                shs::Color col = ((ix + iz) % 2 == 0) ? grass_c1 : grass_c2;
                if (avg_y > 28.0f) col = hill_c;

                // Top-facing (+Y) winding. SHS uses LH +Z forward, but the
                // world-space facet normal still follows cross(e1, e2).
                tris.emplace_back(p00, p11, p10, col, 0.0f);
                tris.emplace_back(p00, p01, p11, col, 0.0f);
            }
        }

        // Dedicated Runway with explicit depth bias (prevents coplanar Z-fighting)
        float rw_w   = 16.0f;
        float rw_len = 220.0f;
        float rw_y   = 0.08f;

        // Runway main asphalt (depth bias brings it reliably in front of terrain)
        tris.emplace_back(glm::vec3(-rw_w, rw_y, -rw_len), glm::vec3( rw_w, rw_y, rw_len), glm::vec3( rw_w, rw_y, -rw_len), runway_c, -0.0005f);
        tris.emplace_back(glm::vec3(-rw_w, rw_y, -rw_len), glm::vec3(-rw_w, rw_y,  rw_len), glm::vec3( rw_w, rw_y,  rw_len), runway_c, -0.0005f);

        // Centerline stripes (depth bias brings it reliably in front of runway)
        for (float z = -rw_len + 15.0f; z < rw_len - 15.0f; z += 35.0f) {
            float sw   = 0.8f;
            float slen = 16.0f;
            float sy   = 0.12f;
            tris.emplace_back(glm::vec3(-sw, sy, z), glm::vec3( sw, sy, z + slen), glm::vec3( sw, sy, z), stripe_c, -0.0010f);
            tris.emplace_back(glm::vec3(-sw, sy, z), glm::vec3(-sw, sy, z + slen), glm::vec3( sw, sy, z + slen), stripe_c, -0.0010f);
        }

        return tris;
    }
};

// ============================================================================
// LOW-POLY AIRPLANE MESH (~60 clean triangles)
// ============================================================================
class SyntheticAirplane {
public:
    static std::vector<LowPolyTriangle> build_mesh() {
        std::vector<LowPolyTriangle> tris;

        shs::Color red    = shs::Color{225, 45, 40, 255};
        shs::Color white  = shs::Color{240, 240, 245, 255};
        shs::Color glass  = shs::Color{70, 160, 235, 255};
        shs::Color yellow = shs::Color{245, 205, 40, 255};
        shs::Color dark   = shs::Color{45, 45, 50, 255};

        auto add_quad = [&](glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, shs::Color c) {
            tris.emplace_back(v0, v1, v2, c, 0.0f);
            tris.emplace_back(v0, v2, v3, c, 0.0f);
        };

        auto add_box = [&](glm::vec3 center, glm::vec3 size, shs::Color c_top, shs::Color c_side, shs::Color c_bot) {
            glm::vec3 h    = size * 0.5f;
            glm::vec3 p000 = center + glm::vec3(-h.x, -h.y, -h.z);
            glm::vec3 p100 = center + glm::vec3( h.x, -h.y, -h.z);
            glm::vec3 p110 = center + glm::vec3( h.x,  h.y, -h.z);
            glm::vec3 p010 = center + glm::vec3(-h.x,  h.y, -h.z);
            glm::vec3 p001 = center + glm::vec3(-h.x, -h.y,  h.z);
            glm::vec3 p101 = center + glm::vec3( h.x, -h.y,  h.z);
            glm::vec3 p111 = center + glm::vec3( h.x,  h.y,  h.z);
            glm::vec3 p011 = center + glm::vec3(-h.x,  h.y,  h.z);

            add_quad(p001, p101, p111, p011, c_side); // Front (+Z)
            add_quad(p100, p000, p010, p110, c_side); // Back (-Z)
            add_quad(p010, p011, p111, p110, c_top ); // Top (+Y)
            add_quad(p000, p100, p101, p001, c_bot ); // Bottom (-Y)
            add_quad(p100, p110, p111, p101, c_side); // Right (+X)
            add_quad(p000, p001, p011, p010, c_side); // Left (-X)
        };

        // 1. Fuselage Core
        add_box(glm::vec3(0, 0.4f, 0.0f), glm::vec3(0.85f, 0.75f, 3.6f), red, red, white);

        // 2. Nose Cone
        glm::vec3 nose_tip(0.0f, 0.35f, 2.65f);
        glm::vec3 nf0(-0.42f, 0.05f, 1.8f), nf1( 0.42f, 0.05f, 1.8f);
        glm::vec3 nf2( 0.42f, 0.75f, 1.8f), nf3(-0.42f, 0.75f, 1.8f);
        tris.emplace_back(nf3, nf2, nose_tip, yellow, 0.0f);
        tris.emplace_back(nf0, nose_tip, nf1, white,  0.0f);
        tris.emplace_back(nf2, nf1, nose_tip, red,    0.0f);
        tris.emplace_back(nf0, nf3, nose_tip, red,    0.0f);

        // 3. Cockpit Canopy
        add_box(glm::vec3(0.0f, 0.92f, 0.3f), glm::vec3(0.55f, 0.40f, 1.3f), glass, glass, glass);

        // 4. Main Wings
        add_box(glm::vec3( 2.7f, 0.35f, 0.2f), glm::vec3(4.6f, 0.08f, 1.2f), red, red, white);
        add_box(glm::vec3(-2.7f, 0.35f, 0.2f), glm::vec3(4.6f, 0.08f, 1.2f), red, red, white);
        // Wingtips
        add_box(glm::vec3( 5.05f, 0.42f, 0.2f), glm::vec3(0.12f, 0.25f, 1.1f), yellow, yellow, yellow);
        add_box(glm::vec3(-5.05f, 0.42f, 0.2f), glm::vec3(0.12f, 0.25f, 1.1f), yellow, yellow, yellow);

        // 5. Tail Vertical Fin
        add_box(glm::vec3(0.0f, 1.10f, -1.65f), glm::vec3(0.12f, 1.25f, 1.0f), yellow, yellow, yellow);

        // 6. Tail Horizontal Elevators
        add_box(glm::vec3( 1.05f, 0.42f, -1.75f), glm::vec3(1.9f, 0.08f, 0.75f), red, red, white);
        add_box(glm::vec3(-1.05f, 0.42f, -1.75f), glm::vec3(1.9f, 0.08f, 0.75f), red, red, white);

        // 7. Landing Gear Struts & Wheels
        add_box(glm::vec3(-1.0f, -0.2f,  0.4f ), glm::vec3(0.08f, 0.5f , 0.08f), dark, dark, dark);
        add_box(glm::vec3(-1.0f, -0.5f,  0.4f ), glm::vec3(0.18f, 0.3f , 0.3f ), dark, dark, dark);
        add_box(glm::vec3( 1.0f, -0.2f,  0.4f ), glm::vec3(0.08f, 0.5f , 0.08f), dark, dark, dark);
        add_box(glm::vec3( 1.0f, -0.5f,  0.4f ), glm::vec3(0.18f, 0.3f , 0.3f ), dark, dark, dark);
        add_box(glm::vec3(0.0f, -0.15f, -1.75f), glm::vec3(0.06f, 0.35f, 0.06f), dark, dark, dark);
        add_box(glm::vec3(0.0f, -0.35f, -1.75f), glm::vec3(0.12f, 0.18f, 0.18f), dark, dark, dark);

        return tris;
    }

    static std::vector<LowPolyTriangle> build_propeller() {
        std::vector<LowPolyTriangle> tris;
        shs::Color prop_c = shs::Color{30, 30, 30, 255};
        shs::Color tip_c  = shs::Color{245, 205, 40, 255};

        auto add_blade = [&](float angle) {
            float c = std::cos(angle);
            float s = std::sin(angle);
            glm::vec3 ax(c, s, 0);
            glm::vec3 ay(-s, c, 0);

            glm::vec3 p0 = ax * 0.15f - ay * 0.06f;
            glm::vec3 p1 = ax * 1.05f - ay * 0.04f;
            glm::vec3 p2 = ax * 1.05f + ay * 0.04f;
            glm::vec3 p3 = ax * 0.15f + ay * 0.06f;

            tris.emplace_back(p0, p1, p2, prop_c, 0.0f);
            tris.emplace_back(p0, p2, p3, tip_c,  0.0f);
        };

        add_blade(0.0f);
        add_blade(glm::pi<float>());

        glm::vec3 tip(0, 0, 0.3f);
        int segs = 6;
        float r = 0.16f;
        for (int i = 0; i < segs; ++i) {
            float a0 = (float)i * glm::two_pi<float>() / (float)segs;
            float a1 = (float)(i + 1) * glm::two_pi<float>() / (float)segs;
            glm::vec3 p0(std::cos(a0) * r, std::sin(a0) * r, 0.0f);
            glm::vec3 p1(std::cos(a1) * r, std::sin(a1) * r, 0.0f);
            tris.emplace_back(p0, p1, tip, tip_c, 0.0f);
        }

        return tris;
    }
};

// ============================================================================
// STYLIZED LOW-POLY CLOUDS (~40 clean triangles per cloud)
// ============================================================================
class SyntheticClouds {
public:
    static std::vector<LowPolyTriangle> build_cloud_field() {
        std::vector<LowPolyTriangle> tris;
        shs::Color c_top  = shs::Color{245, 250, 255, 255};
        shs::Color c_side = shs::Color{220, 230, 245, 255};
        shs::Color c_bot  = shs::Color{185, 200, 220, 255};

        auto add_cloud_block = [&](glm::vec3 center, glm::vec3 size) {
            glm::vec3 h = size * 0.5f;
            glm::vec3 p000 = center + glm::vec3(-h.x, -h.y, -h.z);
            glm::vec3 p100 = center + glm::vec3( h.x, -h.y, -h.z);
            glm::vec3 p110 = center + glm::vec3( h.x,  h.y, -h.z);
            glm::vec3 p010 = center + glm::vec3(-h.x,  h.y, -h.z);
            glm::vec3 p001 = center + glm::vec3(-h.x, -h.y,  h.z);
            glm::vec3 p101 = center + glm::vec3( h.x, -h.y,  h.z);
            glm::vec3 p111 = center + glm::vec3( h.x,  h.y,  h.z);
            glm::vec3 p011 = center + glm::vec3(-h.x,  h.y,  h.z);

            tris.emplace_back(p001, p101, p111, c_side, 0.0f); tris.emplace_back(p001, p111, p011, c_side, 0.0f);
            tris.emplace_back(p100, p000, p010, c_side, 0.0f); tris.emplace_back(p100, p010, p110, c_side, 0.0f);
            tris.emplace_back(p010, p011, p111, c_top,  0.0f); tris.emplace_back(p010, p111, p110, c_top,  0.0f);
            tris.emplace_back(p000, p100, p101, c_bot,  0.0f); tris.emplace_back(p000, p101, p001, c_bot,  0.0f);
            tris.emplace_back(p100, p110, p111, c_side, 0.0f); tris.emplace_back(p100, p111, p101, c_side, 0.0f);
            tris.emplace_back(p000, p001, p011, c_side, 0.0f); tris.emplace_back(p000, p011, p010, c_side, 0.0f);
        };

        auto add_cloud_cluster = [&](glm::vec3 c, float s) {
            add_cloud_block(c, glm::vec3(80, 20, 50) * s);
            add_cloud_block(c + glm::vec3(25, 12, 10) * s, glm::vec3(55, 24, 40) * s);
            add_cloud_block(c + glm::vec3(-30, 8, -10) * s, glm::vec3(50, 18, 45) * s);
        };

        add_cloud_cluster(glm::vec3(-250, 140,  300), 1.0f);
        add_cloud_cluster(glm::vec3( 350, 160,  150), 1.2f);
        add_cloud_cluster(glm::vec3(-400, 150, -250), 1.1f);
        add_cloud_cluster(glm::vec3( 180, 170, -400), 0.9f);
        add_cloud_cluster(glm::vec3( 450, 130,  450), 1.3f);

        return tris;
    }
};

// ============================================================================
// FLIGHT PHYSICS
// ============================================================================
struct PlaneState {
    glm::vec3 position       = glm::vec3(0.0f, PLANE_GROUND_CLEARANCE, -140.0f);
    glm::quat orientation    = glm::quat(glm::vec3(0.0f, 0.0f, 0.0f));
    glm::vec3 velocity       = glm::vec3(0.0f);
    float throttle           = 0.0f;
    float prop_angle         = 0.0f;

    float pitch_input        = 0.0f;
    float roll_input         = 0.0f;
    float yaw_input          = 0.0f;
    bool  brakes             = false;

    bool  is_on_ground       = true;
    float airspeed           = 0.0f;
    float altitude           = 0.0f;

    void update(float dt) {
        glm::vec3 forward = orientation * glm::vec3(0, 0, 1);
        glm::vec3 up      = orientation * glm::vec3(0, 1, 0);

        airspeed = glm::dot(velocity, forward);
        altitude = position.y - SyntheticTerrain::get_height(position.x, position.z);

        float authority = glm::clamp(std::abs(airspeed) / 22.0f, 0.15f, 1.2f);
        if (is_on_ground) authority = std::min(authority, 0.45f);

        float pitch_rate = pitch_input * 1.5f * authority;
        float roll_rate  = roll_input  * 2.6f * authority;
        float yaw_rate   = yaw_input   * 1.0f * authority;

        glm::quat q_pitch = glm::angleAxis(pitch_rate * dt, glm::vec3(1, 0, 0));
        glm::quat q_yaw   = glm::angleAxis(yaw_rate * dt,   glm::vec3(0, 1, 0));
        glm::quat q_roll  = glm::angleAxis(roll_rate * dt,  glm::vec3(0, 0, -1));

        orientation = glm::normalize(orientation * (q_yaw * q_pitch * q_roll));

        forward = orientation * glm::vec3(0, 0, 1);
        up      = orientation * glm::vec3(0, 1, 0);

        const float MAX_THRUST = 110.0f;
        const float DRAG_COEFF = 0.032f;
        const float LIFT_COEFF = 0.70f;
        const float MASS       = 1.0f;

        glm::vec3 thrust_force  = forward * (throttle * MAX_THRUST);
        glm::vec3 drag_force    = -velocity * (DRAG_COEFF * glm::length(velocity));
        float forward_speed_sq  = std::max(0.0f, airspeed * airspeed);
        glm::vec3 lift_force    = up * (LIFT_COEFF * forward_speed_sq * 0.08f);
        glm::vec3 gravity_force = glm::vec3(0, -9.81f * MASS, 0);

        glm::vec3 total_force   = thrust_force + drag_force + lift_force + gravity_force;
        velocity += (total_force / MASS) * dt;

        float rpm = (throttle * 90.0f + 10.0f);
        prop_angle += rpm * dt * glm::two_pi<float>();
        if (prop_angle > glm::two_pi<float>()) prop_angle -= glm::two_pi<float>();

        position += velocity * dt;

        float ground_y = SyntheticTerrain::get_height(position.x, position.z);
        if (position.y <= ground_y + PLANE_GROUND_CLEARANCE) {
            position.y   = ground_y + PLANE_GROUND_CLEARANCE;
            is_on_ground = true;
            if (velocity.y < 0.0f) velocity.y = 0.0f;

            float friction = brakes ? 8.0f : 1.2f;
            velocity.x -= velocity.x * friction * dt;
            velocity.z -= velocity.z * friction * dt;

            glm::vec3 euler = glm::eulerAngles(orientation);
            euler.z *= std::max(0.0f, 1.0f - 5.0f * dt);
            orientation = glm::quat(euler);
        } else {
            is_on_ground = false;
        }
    }

    void reset() {
        position     = glm::vec3(0.0f, PLANE_GROUND_CLEARANCE, -140.0f);
        orientation  = glm::quat(glm::vec3(0.0f));
        velocity     = glm::vec3(0.0f);
        throttle     = 0.0f;
        is_on_ground = true;
    }
};

// ============================================================================
// CAMERA CONTROLLER
// ============================================================================
enum class CameraMode { CHASE_CAM = 0, COCKPIT_CAM, FREE_ORBIT };

struct FlightCamera {
    CameraMode mode    = CameraMode::CHASE_CAM;
    glm::vec3 position = glm::vec3(0.0f, 5.0f, -165.0f);
    glm::vec3 target   = glm::vec3(0.0f, 0.0f, -140.0f);
    float orbit_yaw    = 0.0f;
    float orbit_pitch  = 12.0f;
    float orbit_dist   = 12.0f;

    void update(const PlaneState& plane, float dt) {
        glm::vec3 p_forward = plane.orientation * glm::vec3(0, 0, 1);

        if (mode == CameraMode::CHASE_CAM) {
            target = plane.position + glm::vec3(0, 0.6f, 0);

            float rad_yaw   = glm::radians(orbit_yaw);
            float rad_pitch = glm::radians(orbit_pitch);

            glm::vec3 offset_dir = -p_forward * std::cos(rad_yaw) + (plane.orientation * glm::vec3(1,0,0)) * std::sin(rad_yaw);
            offset_dir.y += std::sin(rad_pitch);
            offset_dir = glm::normalize(offset_dir);

            glm::vec3 desired_pos = target + offset_dir * orbit_dist + glm::vec3(0, 1.8f, 0);
            float lerp_factor = 1.0f - std::exp(-8.0f * dt);
            position = glm::mix(position, desired_pos, lerp_factor);

        } else if (mode == CameraMode::COCKPIT_CAM) {
            position = plane.position + plane.orientation * glm::vec3(0, 0.82f, 0.4f);
            target   = position + p_forward * 50.0f;

        } else if (mode == CameraMode::FREE_ORBIT) {
            target = plane.position;
            float ry = glm::radians(orbit_yaw);
            float rp = glm::radians(orbit_pitch);
            glm::vec3 off(
                std::sin(ry) * std::cos(rp) * orbit_dist,
                std::sin(rp) * orbit_dist,
                -std::cos(ry) * std::cos(rp) * orbit_dist
            );
            position = target + off;
        }
    }
};

// ============================================================================
// HIGH PRECISION PERSPECTIVE RASTERIZER
// ============================================================================
static inline glm::vec4 clip_to_screen_vec4(const glm::vec4& clip, int W, int H) {
    float inv_w = 1.0f / clip.w;
    glm::vec3 ndc = glm::vec3(clip) * inv_w;
    glm::vec4 s;
    s.x = (ndc.x + 1.0f) * 0.5f * (float)(W - 1);
    s.y = (1.0f - ndc.y) * 0.5f * (float)(H - 1);
    s.z = ndc.z;
    s.w = inv_w; // Store 1/w for perspective-correct interpolation
    return s;
}

static void rasterize_perspective_triangle_tile(
    shs::Canvas& canvas, shs::ZBuffer& z_buffer,
    const glm::vec4& sc0, const glm::vec4& sc1, const glm::vec4& sc2,
    shs::Color lit_color, float depth_bias,
    glm::ivec2 tile_min, glm::ivec2 tile_max)
{
    glm::vec2 v0(sc0.x, sc0.y);
    glm::vec2 v1(sc1.x, sc1.y);
    glm::vec2 v2(sc2.x, sc2.y);

    // Screen space backface culling
    float area = (v1.x - v0.x) * (v2.y - v0.y) - (v1.y - v0.y) * (v2.x - v0.x);
    if (!shs::Raster::is_front_facing_screen(area, shs::Raster::FrontFace::CounterClockwise)) return;

    glm::vec2 bboxmin = glm::max(glm::vec2(tile_min), glm::min(v0, glm::min(v1, v2)));
    glm::vec2 bboxmax = glm::min(glm::vec2(tile_max), glm::max(v0, glm::max(v1, v2)));

    if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

    std::vector<glm::vec2> v2d = { v0, v1, v2 };

    int min_x = (int)bboxmin.x; int max_x = (int)bboxmax.x;
    int min_y = (int)bboxmin.y; int max_y = (int)bboxmax.y;


    for (int py = min_y; py <= max_y; ++py) {
        for (int px = min_x; px <= max_x; ++px) {
            glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2((float)px + 0.5f, (float)py + 0.5f), v2d);
            if (bc.x < 0.0f || bc.y < 0.0f || bc.z < 0.0f) continue;

            float interp_z = shs::Raster::interpolate_ndc_depth(bc, sc0.z, sc1.z, sc2.z);

            // Apply depth bias
            float final_z = interp_z + depth_bias;

            if (final_z < -1.0f || final_z > 1.0f) continue;

            if (z_buffer.test_and_set_depth_screen_space(px, py, final_z)) {
                canvas.draw_pixel_screen_space(px, py, lit_color);
            }
        }
    }
}

// ============================================================================
// CLEAN SKY PASS
// ============================================================================
static void render_sky_pass(shs::Canvas& canvas, const glm::mat4& view, const glm::mat4& proj,
                            shs::Job::ThreadedPriorityJobSystem* job_sys, shs::Job::WaitGroup& wg)
{
    int W = canvas.get_width();
    int H = canvas.get_height();

    int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
    int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    glm::mat4 inv_vp = glm::inverse(proj * view);

    wg.reset();

    for (int ty = 0; ty < rows; ++ty) {
        for (int tx = 0; tx < cols; ++tx) {
            wg.add(1);
            job_sys->submit({[&canvas, tx, ty, W, H, inv_vp, &wg]() {
                int x0 = tx * TILE_SIZE_X;
                int y0 = ty * TILE_SIZE_Y;
                int x1 = std::min(x0 + TILE_SIZE_X, W);
                int y1 = std::min(y0 + TILE_SIZE_Y, H);

                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        float ndc_x = ((float)x + 0.5f) / (float)W * 2.0f - 1.0f;
                        float ndc_y = 1.0f - ((float)y + 0.5f) / (float)H * 2.0f;

                        glm::vec4 p_near = inv_vp * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
                        glm::vec4 p_far  = inv_vp * glm::vec4(ndc_x, ndc_y,  1.0f, 1.0f);
                        glm::vec3 ray_dir = glm::normalize(glm::vec3(p_far / p_far.w) - glm::vec3(p_near / p_near.w));

                        glm::vec3 zenith_col  = glm::vec3(0.18f, 0.48f, 0.95f);
                        glm::vec3 horizon_col = glm::vec3(0.68f, 0.85f, 0.98f);
                        glm::vec3 ground_col  = glm::vec3(0.40f, 0.50f, 0.38f);

                        glm::vec3 sky_color;
                        if (ray_dir.y >= 0.0f) {
                            sky_color = glm::mix(horizon_col, zenith_col, std::pow(ray_dir.y, 0.6f));
                        } else {
                            sky_color = glm::mix(horizon_col, ground_col, std::min(1.0f, -ray_dir.y * 2.5f));
                        }

                        float sun_dot = glm::dot(ray_dir, -LIGHT_DIR_WORLD);
                        if (sun_dot > 0.9985f) {
                            sky_color = glm::vec3(1.2f, 1.15f, 0.95f);
                        } else if (sun_dot > 0.985f) {
                            float glow = (sun_dot - 0.985f) / (0.9985f - 0.985f);
                            sky_color = glm::mix(sky_color, glm::vec3(1.0f, 0.95f, 0.8f), glow * 0.7f);
                        }

                        canvas.draw_pixel_screen_space(x, y, shs::rgb01_to_color(sky_color));
                    }
                }
                wg.done();
            }, shs::Job::PRIORITY_HIGH});
        }
    }
    wg.wait();
}

// ============================================================================
// HUD OVERLAY
// ============================================================================
static void draw_hud(shs::Canvas& canvas, const PlaneState& plane) {
    shs::Color hud_col = shs::Color{40, 230, 100, 255};

    int cx = canvas.get_width() / 2;
    int cy = canvas.get_height() / 2;

    shs::Canvas::draw_line(canvas, cx - 16, cy, cx - 4, cy, hud_col);
    shs::Canvas::draw_line(canvas, cx + 4, cy, cx + 16, cy, hud_col);
    shs::Canvas::draw_line(canvas, cx, cy - 8, cx, cy + 8, hud_col);

    int bar_x = 40;
    int bar_y = 50;
    int bar_h = 140;
    int bar_w = 14;
    shs::Canvas::draw_line(canvas, bar_x, bar_y, bar_x + bar_w, bar_y, hud_col);
    shs::Canvas::draw_line(canvas, bar_x, bar_y + bar_h, bar_x + bar_w, bar_y + bar_h, hud_col);
    shs::Canvas::draw_line(canvas, bar_x, bar_y, bar_x, bar_y + bar_h, hud_col);
    shs::Canvas::draw_line(canvas, bar_x + bar_w, bar_y, bar_x + bar_w, bar_y + bar_h, hud_col);

    int fill_h = (int)(plane.throttle * (float)bar_h);
    for (int y = bar_y + 1; y < bar_y + fill_h; ++y) {
        for (int x = bar_x + 1; x < bar_x + bar_w; ++x) {
            canvas.draw_pixel(x, y, hud_col);
        }
    }
}

// ============================================================================
// MAIN APPLICATION
// ============================================================================
int main(int argc, char* argv[]) {
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
        std::cerr << "SDL_Init Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow(
        "SHS Renderer - Flight Simulator Demo [WASD/Arrows: Steer, Wheel: Throttle, Space: Brakes, C: Cam, Mouse: Orbit]",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT,
        SDL_WINDOW_SHOWN
    );

    SDL_Renderer* sdl_renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture* screen_texture = SDL_CreateTexture(
        sdl_renderer,
        SDL_PIXELFORMAT_RGBA32,
        SDL_TEXTUREACCESS_STREAMING,
        CANVAS_WIDTH, CANVAS_HEIGHT
    );
    SDL_Surface* screen_surface = SDL_CreateRGBSurfaceWithFormat(0, CANVAS_WIDTH, CANVAS_HEIGHT, 32, SDL_PIXELFORMAT_RGBA32);

    shs::Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{30, 35, 45, 255});
    shs::ZBuffer z_buffer(CANVAS_WIDTH, CANVAS_HEIGHT, -1.0f, 1.0f);

    shs::Job::ThreadedPriorityJobSystem job_system(THREAD_COUNT);
    shs::Job::WaitGroup wg_render;
    shs::Job::WaitGroup wg_sky;

    std::cout << "Generating Low-Poly Synthetic World..." << std::endl;
    std::vector<LowPolyTriangle> terrain_tris = SyntheticTerrain::build_mesh();
    std::vector<LowPolyTriangle> plane_tris   = SyntheticAirplane::build_mesh();
    std::vector<LowPolyTriangle> prop_tris    = SyntheticAirplane::build_propeller();
    std::vector<LowPolyTriangle> cloud_tris   = SyntheticClouds::build_cloud_field();

    std::cout << "World Polygons: " 
              << (terrain_tris.size() + plane_tris.size() + prop_tris.size() + cloud_tris.size()) 
              << " triangles total." << std::endl;

    PlaneState plane;
    FlightCamera camera;

    bool quit = false;
    SDL_Event e;
    Uint32 last_tick = SDL_GetTicks();
    bool is_mouse_dragging = false;

    int frame_count = 0;
    float fps_timer = 0.0f;

    while (!quit) {
        Uint32 current_tick = SDL_GetTicks();
        float dt = (current_tick - last_tick) / 1000.0f;
        last_tick = current_tick;
        if (dt > 0.1f) dt = 0.1f;

        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;

            if (e.type == SDL_MOUSEBUTTONDOWN && e.button.button == SDL_BUTTON_LEFT) is_mouse_dragging = true;
            if (e.type == SDL_MOUSEBUTTONUP && e.button.button == SDL_BUTTON_LEFT)   is_mouse_dragging = false;

            if (e.type == SDL_MOUSEMOTION && is_mouse_dragging) {
                camera.orbit_yaw   += (float)e.motion.xrel * 0.25f;
                camera.orbit_pitch -= (float)e.motion.yrel * 0.25f;
                camera.orbit_pitch = glm::clamp(camera.orbit_pitch, -35.0f, 85.0f);
            }

            if (e.type == SDL_MOUSEWHEEL) {
                // SDL reports natural-scrolling devices as flipped. Normalize it
                // so wheel-up always increases thrust and wheel-down decreases it.
                float wheel_delta = (e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED)
                    ? -(float)e.wheel.y
                    :  (float)e.wheel.y;
                plane.throttle = glm::clamp(
                    plane.throttle + wheel_delta * THROTTLE_STEP,
                    0.0f,
                    1.0f
                );
            }

            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = true;
                if (e.key.keysym.sym == SDLK_r) plane.reset();
                if (e.key.keysym.sym == SDLK_c) {
                    camera.mode = (CameraMode)(((int)camera.mode + 1) % 3);
                }
            }
        }

        const Uint8* keys = SDL_GetKeyboardState(NULL);

        plane.pitch_input = 0.0f;
        if (keys[SDL_SCANCODE_S] || keys[SDL_SCANCODE_DOWN]) plane.pitch_input += 1.0f;
        if (keys[SDL_SCANCODE_W] || keys[SDL_SCANCODE_UP])   plane.pitch_input -= 1.0f;

        plane.roll_input = 0.0f;
        if (keys[SDL_SCANCODE_A] || keys[SDL_SCANCODE_LEFT])  plane.roll_input -= 1.0f;
        if (keys[SDL_SCANCODE_D] || keys[SDL_SCANCODE_RIGHT]) plane.roll_input += 1.0f;

        plane.yaw_input = 0.0f;
        if (keys[SDL_SCANCODE_Q]) plane.yaw_input -= 1.0f;
        if (keys[SDL_SCANCODE_E]) plane.yaw_input += 1.0f;

        plane.brakes = keys[SDL_SCANCODE_SPACE] != 0;

        // Update dynamics & camera
        plane.update(dt);
        camera.update(plane, dt);

        // Clear Z-Buffer
        z_buffer.clear();

        // Matrices (Z_NEAR = 1.0f gives 5x higher Z precision than 0.2f)
        glm::vec3 camera_up(0.0f, 1.0f, 0.0f);
        if (camera.mode == CameraMode::COCKPIT_CAM) {
            // A cockpit must bank with the aircraft rather than using a fixed horizon.
            camera_up = glm::normalize(plane.orientation * glm::vec3(0, 1, 0));
            glm::vec3 camera_forward = glm::normalize(camera.target - camera.position);
            if (std::abs(glm::dot(camera_forward, camera_up)) > 0.98f) {
                camera_up = glm::vec3(0.0f, 1.0f, 0.0f);
            }
        }
        glm::mat4 view = glm::lookAtLH(camera.position, camera.target, camera_up);
        glm::mat4 proj = glm::perspectiveLH_NO(glm::radians(60.0f), (float)CANVAS_WIDTH / (float)CANVAS_HEIGHT, Z_NEAR, Z_FAR);
        glm::mat4 vp   = proj * view;

        // 1. Sky Pass
        render_sky_pass(canvas, view, proj, &job_system, wg_sky);

        // Assemble transformed world batches
        struct ProcessedTriangle {
            glm::vec4 c0, c1, c2; // Clip coordinates
            shs::Color lit_color;
            float depth_bias;
        };

        std::vector<ProcessedTriangle> active_tris;
        active_tris.reserve(terrain_tris.size() + cloud_tris.size() + plane_tris.size() + prop_tris.size());

        auto process_batch = [&](const std::vector<LowPolyTriangle>& batch_tris, const glm::mat4& model) {
            glm::mat4 mvp = vp * model;

            for (const auto& tri : batch_tris) {
                // Compute facet normal in World space
                glm::vec3 w0 = glm::vec3(model * glm::vec4(tri.p0, 1.0f));
                glm::vec3 w1 = glm::vec3(model * glm::vec4(tri.p1, 1.0f));
                glm::vec3 w2 = glm::vec3(model * glm::vec4(tri.p2, 1.0f));

                glm::vec3 e1 = w1 - w0;
                glm::vec3 e2 = w2 - w0;
                glm::vec3 N = glm::cross(e1, e2);
                float len = glm::length(N);
                if (len < 1e-6f) continue;
                N /= len;

                // Directional Sun lighting + sky ambient
                glm::vec3 L = -LIGHT_DIR_WORLD;
                float NdotL = std::max(0.0f, glm::dot(N, L));
                float diffuse = NdotL * 0.75f + 0.25f;
                float ambient = std::max(0.0f, N.y) * 0.25f + 0.15f;

                glm::vec3 base_col = glm::vec3(tri.color.r, tri.color.g, tri.color.b) / 255.0f;
                glm::vec3 lit_rgb  = base_col * (diffuse * glm::vec3(1.0f, 0.98f, 0.92f) + ambient * glm::vec3(0.55f, 0.75f, 1.0f));

                // Atmospheric Distance Fog
                float dist = glm::length(camera.position - (w0 + w1 + w2) * 0.333f);
                float fog = glm::clamp((dist - 200.0f) / 750.0f, 0.0f, 0.85f);
                lit_rgb = glm::mix(lit_rgb, glm::vec3(0.68f, 0.85f, 0.98f), fog);

                shs::Color final_c = shs::rgb01_to_color(lit_rgb);

                glm::vec4 c0 = mvp * glm::vec4(tri.p0, 1.0f);
                glm::vec4 c1 = mvp * glm::vec4(tri.p1, 1.0f);
                glm::vec4 c2 = mvp * glm::vec4(tri.p2, 1.0f);

                active_tris.push_back({ c0, c1, c2, final_c, tri.depth_bias });
            }
        };

        // Process terrain & clouds
        process_batch(terrain_tris, glm::mat4(1.0f));
        process_batch(cloud_tris,   glm::mat4(1.0f));

        // Process airplane
        if (camera.mode != CameraMode::COCKPIT_CAM) {
            glm::mat4 plane_model = glm::translate(glm::mat4(1.0f), plane.position) * glm::mat4_cast(plane.orientation);
            process_batch(plane_tris, plane_model);

            glm::mat4 prop_rot   = glm::rotate(glm::mat4(1.0f), plane.prop_angle, glm::vec3(0, 0, 1));
            glm::mat4 prop_model = plane_model * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.35f, 2.68f)) * prop_rot;
            process_batch(prop_tris, prop_model);
        }

        // Tiled Multi-threaded Rasterization
        int W = canvas.get_width();
        int H = canvas.get_height();
        int cols = (W + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (H + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wg_render.reset();

        for (int ty = 0; ty < rows; ++ty) {
            for (int tx = 0; tx < cols; ++tx) {
                wg_render.add(1);
                job_system.submit({[&, tx, ty, W, H]() {
                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, W) - 1,
                                     std::min((ty + 1) * TILE_SIZE_Y, H) - 1);

                    for (const auto& tri : active_tris) {
                        const shs::Raster::FrustumClipPolygon poly =
                            shs::Raster::clip_triangle_to_frustum(tri.c0, tri.c1, tri.c2);
                        if (poly.count < 3) continue;

                        glm::vec4 s0 = clip_to_screen_vec4(poly.vertices[0], W, H);

                        for (int i = 1; i + 1 < poly.count; ++i) {
                            glm::vec4 s1 = clip_to_screen_vec4(poly.vertices[i], W, H);
                            glm::vec4 s2 = clip_to_screen_vec4(poly.vertices[i + 1], W, H);
                            rasterize_perspective_triangle_tile(canvas, z_buffer, s0, s1, s2, tri.lit_color, tri.depth_bias, t_min, t_max);
                        }
                    }

                    wg_render.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wg_render.wait();

        // 2. HUD Overlay
        draw_hud(canvas, plane);

        // --------------------------------------------------------------------
        // PRESENTATION
        // --------------------------------------------------------------------
        shs::Canvas::copy_to_SDLSurface(screen_surface, &canvas);
        SDL_UpdateTexture(screen_texture, NULL, screen_surface->pixels, screen_surface->pitch);
        SDL_RenderClear(sdl_renderer);
        SDL_RenderCopy(sdl_renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(sdl_renderer);

        frame_count++;
        fps_timer += dt;
        if (fps_timer >= 0.5f) {
            std::ostringstream ss;
            ss << "Low-Poly Flight Demo | FPS: " << (int)((float)frame_count / fps_timer)
               << " | Tris: " << active_tris.size()
               << " | Spd: " << std::fixed << std::setprecision(1) << (plane.airspeed * 3.6f) << " km/h"
               << " | Alt: " << (int)plane.altitude << " m"
               << " | Thr: " << (int)(plane.throttle * 100.0f) << "%"
               << " | [WASD/Arrows: Steer, Wheel: Throttle, Space: Brakes, C: Cam, Mouse: Orbit]";
            SDL_SetWindowTitle(window, ss.str().c_str());
            frame_count = 0;
            fps_timer = 0.0f;
        }
    }

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(screen_surface);
    SDL_DestroyRenderer(sdl_renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
