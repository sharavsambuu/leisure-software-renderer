#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: ibl.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Shared IBL data types and precompute/sampling helpers
            for environment irradiance + prefiltered specular chains.
*/

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace shs
{
    struct CubeMapLinear
    {
        int size = 0;
        std::vector<glm::vec3> face[6];

        bool valid() const
        {
            if (size <= 0) return false;
            for (int i = 0; i < 6; ++i)
            {
                if ((int)face[i].size() != size * size) return false;
            }
            return true;
        }

        const glm::vec3& at(int f, int x, int y) const
        {
            return face[f][(size_t)y * (size_t)size + (size_t)x];
        }
    };

    struct PrefilteredSpecular
    {
        std::vector<CubeMapLinear> mip{};

        bool valid() const { return !mip.empty() && mip[0].valid(); }
        int mip_count() const { return (int)mip.size(); }
    };

    struct EnvIBL
    {
        CubeMapLinear env_irradiance{};
        PrefilteredSpecular env_prefiltered_spec{};

        bool valid() const
        {
            return env_irradiance.valid() && env_prefiltered_spec.valid();
        }
    };

    inline glm::vec3 face_uv_to_dir(int face, float u, float v)
    {
        const float a = 2.0f * u - 1.0f;
        const float b = 2.0f * v - 1.0f;
        glm::vec3 d(0.0f);
        switch (face)
        {
            case 0: d = glm::vec3(1.0f, b, -a); break;  // +X
            case 1: d = glm::vec3(-1.0f, b, a); break;  // -X
            case 2: d = glm::vec3(a, 1.0f, -b); break;  // +Y
            case 3: d = glm::vec3(a, -1.0f, b); break;  // -Y
            case 4: d = glm::vec3(a, b, 1.0f); break;   // +Z
            case 5: d = glm::vec3(-a, b, -1.0f); break; // -Z
            default: d = glm::vec3(0.0f, 0.0f, 1.0f); break;
        }
        return glm::normalize(d);
    }

    inline void tangent_basis(const glm::vec3& n, glm::vec3& t, glm::vec3& b)
    {
        const glm::vec3 up = (std::abs(n.y) < 0.999f) ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        t = glm::normalize(glm::cross(up, n));
        b = glm::cross(n, t);
    }

    inline glm::vec3 cosine_sample_hemisphere(float u1, float u2)
    {
        const float r = std::sqrt(u1);
        const float phi = 6.2831853f * u2;
        const float x = r * std::cos(phi);
        const float y = r * std::sin(phi);
        const float z = std::sqrt(std::max(0.0f, 1.0f - u1));
        return glm::vec3(x, y, z);
    }

    template<typename TSkyLike>
    CubeMapLinear build_env_irradiance(const TSkyLike& sky, int out_size, int sample_count)
    {
        CubeMapLinear irr{};
        irr.size = out_size;
        for (int f = 0; f < 6; ++f)
        {
            irr.face[f].assign((size_t)out_size * (size_t)out_size, glm::vec3(0.0f));
        }

        for (int f = 0; f < 6; ++f)
        {
            for (int y = 0; y < out_size; ++y)
            {
                for (int x = 0; x < out_size; ++x)
                {
                    const float u = (float(x) + 0.5f) / float(out_size);
                    const float v = (float(y) + 0.5f) / float(out_size);
                    const glm::vec3 n = face_uv_to_dir(f, u, v);
                    glm::vec3 t{}, b{};
                    tangent_basis(n, t, b);

                    glm::vec3 sum(0.0f);
                    uint32_t seed = (uint32_t)(f * 73856093u ^ x * 19349663u ^ y * 83492791u);
                    auto rnd01 = [&seed]() {
                        seed = 1664525u * seed + 1013904223u;
                        return float(seed & 0x00ffffffu) / float(0x01000000u);
                    };

                    for (int i = 0; i < sample_count; ++i)
                    {
                        const float r1 = rnd01();
                        const float r2 = rnd01();
                        const glm::vec3 h = cosine_sample_hemisphere(r1, r2);
                        const glm::vec3 l = glm::normalize(t * h.x + b * h.y + n * h.z);
                        sum += sky.sample(l);
                    }
                    irr.face[f][(size_t)y * (size_t)out_size + (size_t)x] = sum / float(sample_count);
                }
            }
        }
        return irr;
    }

    inline float roughness_to_phong_exp(float roughness)
    {
        roughness = std::clamp(roughness, 0.0f, 1.0f);
        const float r2 = std::max(1e-4f, roughness * roughness);
        const float exp = (2.0f / r2) - 2.0f;
        return std::max(1.0f, exp);
    }

    inline glm::vec3 phong_lobe_sample(float u1, float u2, float exp)
    {
        const float phi = 6.2831853f * u1;
        const float cos_t = std::pow(1.0f - u2, 1.0f / (exp + 1.0f));
        const float sin_t = std::sqrt(std::max(0.0f, 1.0f - cos_t * cos_t));
        return glm::vec3(std::cos(phi) * sin_t, std::sin(phi) * sin_t, cos_t);
    }

    template<typename TSkyLike>
    PrefilteredSpecular build_env_prefiltered_specular(
        const TSkyLike& sky,
        int base_size,
        int mip_count,
        int samples_per_texel)
    {
        PrefilteredSpecular out{};
        out.mip.resize((size_t)mip_count);

        for (int m = 0; m < mip_count; ++m)
        {
            const int sz = std::max(1, base_size >> m);
            out.mip[(size_t)m].size = sz;
            for (int f = 0; f < 6; ++f)
            {
                out.mip[(size_t)m].face[f].assign((size_t)sz * (size_t)sz, glm::vec3(0.0f));
            }

            const float roughness = float(m) / float(std::max(1, mip_count - 1));
            const float exp = roughness_to_phong_exp(roughness);

            for (int f = 0; f < 6; ++f)
            {
                for (int y = 0; y < sz; ++y)
                {
                    for (int x = 0; x < sz; ++x)
                    {
                        const float u = (float(x) + 0.5f) / float(sz);
                        const float v = (float(y) + 0.5f) / float(sz);
                        const glm::vec3 r = face_uv_to_dir(f, u, v);
                        glm::vec3 t{}, b{};
                        tangent_basis(r, t, b);

                        glm::vec3 sum(0.0f);
                        uint32_t seed = (uint32_t)(m * 2654435761u ^ f * 97531u ^ x * 31337u ^ y * 1337u);
                        auto rnd01 = [&seed]() {
                            seed = 1664525u * seed + 1013904223u;
                            return float(seed & 0x00ffffffu) / float(0x01000000u);
                        };

                        for (int i = 0; i < samples_per_texel; ++i)
                        {
                            const float r1 = rnd01();
                            const float r2 = rnd01();
                            const glm::vec3 s = phong_lobe_sample(r1, r2, exp);
                            const glm::vec3 l = glm::normalize(t * s.x + b * s.y + r * s.z);
                            sum += sky.sample(l);
                        }

                        out.mip[(size_t)m].face[f][(size_t)y * (size_t)sz + (size_t)x] = sum / float(samples_per_texel);
                    }
                }
            }
        }

        return out;
    }

    inline glm::vec3 sample_face_bilinear_linear_vec(const CubeMapLinear& cm, int face, float u, float v)
    {
        u = std::clamp(u, 0.0f, 1.0f);
        v = std::clamp(v, 0.0f, 1.0f);
        const float fx = u * float(cm.size - 1);
        const float fy = v * float(cm.size - 1);
        const int x0 = std::clamp((int)std::floor(fx), 0, cm.size - 1);
        const int y0 = std::clamp((int)std::floor(fy), 0, cm.size - 1);
        const int x1 = std::clamp(x0 + 1, 0, cm.size - 1);
        const int y1 = std::clamp(y0 + 1, 0, cm.size - 1);
        const float tx = fx - float(x0);
        const float ty = fy - float(y0);
        const glm::vec3& c00 = cm.at(face, x0, y0);
        const glm::vec3& c10 = cm.at(face, x1, y0);
        const glm::vec3& c01 = cm.at(face, x0, y1);
        const glm::vec3& c11 = cm.at(face, x1, y1);
        const glm::vec3 cx0 = glm::mix(c00, c10, tx);
        const glm::vec3 cx1 = glm::mix(c01, c11, tx);
        return glm::mix(cx0, cx1, ty);
    }

    inline glm::vec3 sample_cubemap_linear_vec(const CubeMapLinear& cm, const glm::vec3& direction_ws)
    {
        if (!cm.valid()) return glm::vec3(0.0f);
        glm::vec3 d = direction_ws;
        const float len = glm::length(d);
        if (len < 1e-8f) return glm::vec3(0.0f);
        d /= len;

        const float ax = std::abs(d.x);
        const float ay = std::abs(d.y);
        const float az = std::abs(d.z);
        int face = 0;
        float u = 0.5f;
        float v = 0.5f;

        if (ax >= ay && ax >= az)
        {
            if (d.x > 0.0f) { face = 0; u = (-d.z / ax); v = (d.y / ax); }
            else            { face = 1; u = ( d.z / ax); v = (d.y / ax); }
        }
        else if (ay >= ax && ay >= az)
        {
            if (d.y > 0.0f) { face = 2; u = ( d.x / ay); v = (-d.z / ay); }
            else            { face = 3; u = ( d.x / ay); v = ( d.z / ay); }
        }
        else
        {
            if (d.z > 0.0f) { face = 4; u = ( d.x / az); v = (d.y / az); }
            else            { face = 5; u = (-d.x / az); v = (d.y / az); }
        }

        u = 0.5f * (u + 1.0f);
        v = 0.5f * (v + 1.0f);
        return sample_face_bilinear_linear_vec(cm, face, u, v);
    }

    inline glm::vec3 sample_prefiltered_spec_trilinear(
        const PrefilteredSpecular& ps,
        const glm::vec3& direction_ws,
        float lod)
    {
        if (!ps.valid()) return glm::vec3(0.0f);
        const float mmax = float(ps.mip_count() - 1);
        lod = std::clamp(lod, 0.0f, mmax);
        const int m0 = (int)std::floor(lod);
        const int m1 = std::min(m0 + 1, ps.mip_count() - 1);
        const float t = lod - float(m0);
        const glm::vec3 c0 = sample_cubemap_linear_vec(ps.mip[(size_t)m0], direction_ws);
        const glm::vec3 c1 = sample_cubemap_linear_vec(ps.mip[(size_t)m1], direction_ws);
        return glm::mix(c0, c1, t);
    }
}
