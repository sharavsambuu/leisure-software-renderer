/*

    shs_renderer.hpp

    Үндсэн хийсвэрлэлт болон бүтэц:

    1. Buffer<T> (ColorBuffer, DepthBuffer)
       - Санах ойд өгөгдлийг шугаман байдлаар хадгалах ерөнхий класс (T төрлөөр).
       - Canvas болон ZBuffer нь дотроо энэхүү Buffer<T>-ийг ашиглана.

    2. RenderTarget
       - Олон төрлийн буфферийг (Color, Depth, Motion, г.м) нэгтгэсэн бүтэц.
       - Рендер хийх үед эдгээр буфферууд руу зэрэг бичих боломжийг олгоно.

    3. Texture2D
       - Зураг болон текстур өгөгдлийг хадгалах бүтэц.
       - (0,0) цэг нь зүүн доод буланд байрлана.

    4. AbstractSky
       - Тэнгэр болон орчны гэрэлтүүлгийг тооцоолох ерөнхий интерфэйс.
       - Текстур (CubeMapSky) болон Математик (Procedural/Analytic) хувилбаруудтай.

    
    
    Координатын системийн тохиролцоо:

    Model -> World -> View -> Projection -> NDC -> Screen -> shs::Canvas

    1. Model Space (3D):
       - Эхлэл цэг (0,0,0) нь тухайн моделийн төв.
       - X: Баруун, Y: Дээш, Z: Урагшаа (LH)

    2. World Space (3D):
       - Бүх моделиудын нэгдсэн орон зай.
       - X: Баруун, Y: Дээш, Z: Урагшаа

    3. View Space (3D):
       - Камерын байрлал (0,0,0) дээр байна.
       - X: Баруун, Y: Дээш, Z: Урагшаа

    4. Projection/Clip/NDC Space:
       - [-1.0, 1.0] утгын мужид дүрслэгдэнэ.
       - X: Баруун, Y: Дээш, Z: Урагшаа (normalize хийсний дараа)

    5. Screen Space (2D):
       - Эхлэл цэг (0,0) нь Зүүн-Дээд буланд.
       - X: Зүүнээс баруун тийш.
       - Y: Дээд талаас доош.
       - Растеризаци энэ шатанд хийгдэнэ.

    6. shs::Canvas Space (2D):
       - Эхлэл цэг (0,0) нь Зүүн-Доод буланд.
       - X: Зүүнээс баруун тийш.
       - Y: Доод талаас дээш.
       - Screen Space -> Canvas Space хөрвүүлэлт: y_canvas = HEIGHT - y_screen

    
*/

#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <queue>
#include <list>
#include <optional>
#include <atomic>
#include <utility>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <limits>

// External Libraries
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>

namespace shs
{
    // ==========================================
    // ӨНГӨНИЙ ОГТОРГУЙ ХӨРВҮҮЛЭЛТ (sRGB <-> Linear)
    // ==========================================

    // sRGB -ээс Linear руу хөрвүүлэх (Гамма залруулга)
    static inline glm::vec3 srgb_to_linear(glm::vec3 srgb01, float gamma = 2.2f)
    {
        return glm::pow(glm::clamp(srgb01, 0.0f, 1.0f), glm::vec3(gamma));
    }

    // Linear -ээс sRGB руу хөрвүүлэх (Дэлгэцэнд харуулахын өмнө)
    static inline glm::vec3 linear_to_srgb(glm::vec3 lin01, float gamma = 2.2f)
    {
        return glm::pow(glm::clamp(lin01, 0.0f, 1.0f), glm::vec3(1.0f / gamma));
    }

    // ==========================================
    // ҮНДСЭН ӨГӨГДЛИЙН БҮТЦҮҮД (Basic Structures)
    // ==========================================

    struct Color
    {
        uint8_t r;
        uint8_t g;
        uint8_t b;
        uint8_t a;
    };

    struct RawTriangle
    {
        glm::vec3 v1;
        glm::vec3 v2;
        glm::vec3 v3;
    };

    // Shader хооронд дамжих өгөгдөл (Varyings)
    // Vertex Shader-ээс Fragment Shader руу дөхөлт хийгдэж хийгдэж очно.
    struct Varyings {
        glm::vec4 position;        // Clip-space байрлал
        glm::vec4 prev_position;   // Өмнөх фреймийн clip-space байрлал (Velocity тооцоход хэрэгтэй)
        glm::vec3 normal;          // Нормал вектор
        glm::vec3 world_pos;       // World-space байрлал
        glm::vec2 uv;              // Текстурын координат
        float view_z;              // Камерын Z зай (Depth)
    };

    class Pixel
    {
    public:
        Pixel() : color{0, 0, 0, 255} {}
        Pixel(shs::Color color) : color(color) {}
        Pixel(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : color{r, g, b, a} {}

        void change_color(shs::Color color) { this->color = color; }
        shs::Color get_color() const { return this->color; }

        static shs::Pixel red_pixel()   { return shs::Pixel{255, 0, 0, 255}; }
        static shs::Pixel green_pixel() { return shs::Pixel{0, 255, 0, 255}; }
        static shs::Pixel blue_pixel()  { return shs::Pixel{0, 0, 255, 255}; }
        static shs::Pixel black_pixel() { return shs::Pixel{0, 0, 0, 255}; }
        static shs::Pixel white_pixel() { return shs::Pixel{255, 255, 255, 255}; }
        static shs::Pixel random_pixel() {
            return shs::Pixel{
                (uint8_t)(rand() % 256),
                (uint8_t)(rand() % 256),
                (uint8_t)(rand() % 256),
                255};
        }

    private:
        shs::Color color;
    };

    // ==========================================
    // БУФФЕР ХИЙСВЭРЛЭЛТ (Buffer<T>)
    // - Санах ойд өгөгдлийг 2D (w*h) хэлбэрээр хадгалах ерөнхий класс.
    // - Өнгө, Гүн (Depth), эсвэл өөр дурын төрлийн өгөгдөл хадгалж болно.
    // ==========================================

    template<typename T>
    struct Buffer
    {
        int w, h;
        std::vector<T> data;

        Buffer() : w(0), h(0) {}

        Buffer(int width, int height, const T& clear_value)
            : w(width), h(height), data((size_t)width * (size_t)height, clear_value) {}

        inline int width()  const { return w; }
        inline int height() const { return h; }

        // Хязгаар дотор байгаа эсэхийг шалгах
        inline bool in_bounds(int x, int y) const {
            return (x >= 0 && x < w && y >= 0 && y < h);
        }

        // 2D -> 1D индекс рүү хөрвүүлэх
        inline int idx(int x, int y) const { return y * w + x; }

        // Пиксел рүү хандах (Unsafe буюу хурдан)
        inline T& at(int x, int y) { return data[(size_t)idx(x,y)]; }
        inline const T& at(int x, int y) const { return data[(size_t)idx(x,y)]; }

        // 1D байдлаар хандах (Optional)
        inline T& at(int i) { return data[(size_t)i]; }
        inline const T& at(int i) const { return data[(size_t)i]; }

        inline void clear(const T& v) { std::fill(data.begin(), data.end(), v); }
        inline size_t size() const { return data.size(); }

        inline T* raw() { return data.data(); }
        inline const T* raw() const { return data.data(); }
    };

    // Ойлгомжтой болгох үүднээс нэршил өгөх (Aliases)
    using ColorBuffer = Buffer<shs::Color>;
    using DepthBuffer = Buffer<float>;


    // ==========================================
    // ТЕКСТУР БОЛОН BLIT ҮЙЛДЛҮҮД (Texture2D + Blit)
    // - Texture2D координатын систем: (0,0) нь зүүн доод буланд, +Y дээш.
    // - SDL surface (top-left) -> Texture2D (bottom-left) хөрвүүлэлт хийх шаардлагатай.
    // ==========================================

    namespace Tex
    {
        enum BlendMode : int {
            BLEND_NONE  = 0,
            BLEND_ALPHA = 1,
        };

        enum FilterMode : int {
            FILTER_NEAREST  = 0,
            FILTER_BILINEAR = 1,
        };
    }

    struct Texture2D
    {
        int w = 0;
        int h = 0;
        shs::Buffer<shs::Color> texels;

        Texture2D() {}
        Texture2D(int width, int height, shs::Color clear = {0,0,0,0})
            : w(width), h(height), texels(width, height, clear) {}

        inline int width()  const { return w; }
        inline int height() const { return h; }

        inline bool valid() const { return (w > 0 && h > 0 && texels.size() > 0); }

        inline shs::Color get(int x, int y) const {
            if (!texels.in_bounds(x,y)) return {0,0,0,0};
            return texels.at(x,y);
        }
    };

    // ==========================================
    // ТЭНГЭР БОЛОН ОРЧНЫ ГЭРЭЛТҮҮЛЭГ (Sky & Environment)
    // ==========================================

    struct CubeMap
    {
        // 0:+X баруун, 1:-X зүүн, 2:+Y дээд, 3:-Y доод, 4:+Z урд, 5:-Z хойд
        shs::Texture2D face[6];

        inline bool valid() const
        {
            for (int i = 0; i < 6; ++i) if (!face[i].valid()) return false;
            return true;
        }
    };

    // Орчны гэрэлтүүлэг болон тэнгэрийг төлөөлөх суурь класс
    class AbstractSky
    {
    public:
        virtual ~AbstractSky() {}
        // Өгөгдсөн чиглэлд тохирох өнгийг буцаана (Radiance/Color)
        virtual glm::vec3 sample(const glm::vec3& direction) const = 0;
    };

    // Текстурт суурилсан энгийн Skybox (+ Intensity control + Bilinear)
    class CubeMapSky : public AbstractSky
    {
    public:
        CubeMapSky(const CubeMap& cm, float intensity = 1.0f) 
            : cm_(cm), intensity_(intensity) {}

        // Bilinear өнгө холих функц (sRGB -> Linear хөрвүүлэлт хийх)
        glm::vec3 sample_face_bilinear(const shs::Texture2D& tex, float u, float v) const
        {
            u = glm::clamp(u, 0.0f, 1.0f);
            v = glm::clamp(v, 0.0f, 1.0f);

            float fx = u * float(tex.w - 1);
            float fy = v * float(tex.h - 1);

            int x0 = (int)std::floor(fx);
            int y0 = (int)std::floor(fy);
            int x1 = std::min(x0 + 1, tex.w - 1);
            int y1 = std::min(y0 + 1, tex.h - 1);

            float tx = fx - float(x0);
            float ty = fy - float(y0);

            // 4 хөрш пикселийг унших (sRGB)
            shs::Color c00 = tex.get(x0, y0);
            shs::Color c10 = tex.get(x1, y0);
            shs::Color c01 = tex.get(x0, y1);
            shs::Color c11 = tex.get(x1, y1);

            // sRGB -> Linear 
            auto to_lin = [](const shs::Color& c) {
                glm::vec3 s = glm::vec3(c.r, c.g, c.b) / 255.0f;
                return srgb_to_linear(s);
            };

            glm::vec3 v00 = to_lin(c00);
            glm::vec3 v10 = to_lin(c10);
            glm::vec3 v01 = to_lin(c01);
            glm::vec3 v11 = to_lin(c11);

            // Bilinear дөхөлт
            glm::vec3 vx0 = glm::mix(v00, v10, tx);
            glm::vec3 vx1 = glm::mix(v01, v11, tx);
            return glm::mix(vx0, vx1, ty);
        }

        glm::vec3 sample(const glm::vec3& direction) const override
        {
            if (!cm_.valid()) return glm::vec3(0.0f);

            glm::vec3 d = direction;
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

            const shs::Texture2D& tex = cm_.face[face];
            
            // Bilinear
            return sample_face_bilinear(tex, u, v) * intensity_;
        }

    private:
        CubeMap cm_;
        float   intensity_;
    };

    // Математик тооцоололд суурилсан тэнгэр (Procedural)
    class ProceduralSky : public AbstractSky
    {
    public:
        ProceduralSky(glm::vec3 sun_dir = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f)))
            : sun_direction(sun_dir) {}

        glm::vec3 sample(const glm::vec3& dir) const override
        {
            glm::vec3 d = glm::normalize(dir);
            float t = glm::clamp(d.y * 0.5f + 0.5f, 0.0f, 1.0f);
            
            // Тэнгэрийн градиент (Zenith -> Horizon)
            glm::vec3 zenith_color  = glm::vec3(0.05f, 0.2f, 0.5f);
            glm::vec3 horizon_color = glm::vec3(0.3f, 0.6f, 1.0f);
            glm::vec3 sky_color     = glm::mix(horizon_color, zenith_color, t);
            
            // Нарны диск
            float sun_dot = glm::dot(d, -glm::normalize(sun_direction));
            if (sun_dot > 0.9998f) {
                sky_color = glm::vec3(15.0f); // Нарны хурц гэрэл
            } else if (sun_dot > 0.9990f) {
                float glow = (sun_dot - 0.9990f) / (0.9998f - 0.9990f);
                sky_color = glm::mix(sky_color, glm::vec3(10.0f, 8.0f, 4.0f), glow);
            }
            
            return sky_color;
        }

        glm::vec3 sun_direction;
    };

    // Аналитик тэнгэр (Агаар мандлын сарнилт болон нарны туяаг загварчилсан)
    class AnalyticSky : public AbstractSky
    {
    public:
        AnalyticSky(glm::vec3 sun_dir = glm::normalize(glm::vec3(0.4668f, -0.3487f, 0.8127f)))
            : sun_direction(sun_dir) {}

        glm::vec3 sample(const glm::vec3& dir) const override
        {
            glm::vec3 d = glm::normalize(dir);
            glm::vec3 s = -glm::normalize(sun_direction); // Нарны зүг

            float cosTheta = glm::clamp(d.y, -1.0f, 1.0f);
            float cosGamma = glm::dot(d, s);

            // 1. Тэнгэрийн үндсэн өнгө болон газрын өнгө (Zenith, Horizon, Ground)
            // PBR-д орчны гэрэл хангалттай байхын тулд өнгөний хүчийг нэмэв
            glm::vec3 zenith_color  = glm::vec3(0.30f, 0.70f, 1.70f);
            glm::vec3 horizon_color = glm::vec3(1.30f, 1.64f, 2.00f);
            glm::vec3 ground_base   = glm::vec3(0.30f, 0.26f, 0.22f);
            
            // Нарны өндрөөс хамаарч өнгө хувиргах (Sun height influence)
            float sun_height = glm::clamp(s.y, 0.0f, 1.0f);
            zenith_color  = glm::mix(glm::vec3(0.02f, 0.02f, 0.05f), zenith_color,  sun_height);
            horizon_color = glm::mix(glm::vec3(0.4f, 0.15f, 0.05f),  horizon_color, sun_height);
            
            // Газарт тэнгэр болон нарны туяанаас бага зэрэг өнгө нэмэх
            glm::vec3 ground_color = glm::mix(ground_base * 0.5f, ground_base, sun_height);
            ground_color += horizon_color * 0.05f; // Ambient reflection from horizon

            // Тэнгэр ба газрын шилжилтийг зөөлрүүлэх (Continuous smooth gradient)
            float sky_ground_factor = glm::smoothstep(-0.15f, 0.15f, cosTheta);
            
            // Дээд талын тэнгэрийн уусалт (Zenith to Horizon)
            glm::vec3 upper_sky = glm::mix(zenith_color, horizon_color, std::pow(1.0f - glm::max(0.0f, cosTheta), 3.0f));
            
            // Доод талын газрын уусалт: Тэнгэрийн хаяа руу дөхөх тусам агаар мандлын манан (haze) нэмэгдэнэ
            float haze_factor = std::pow(1.0f + glm::min(0.0f, cosTheta), 4.0f);
            glm::vec3 lower_sky = glm::mix(ground_color, horizon_color * 0.5f, haze_factor);
            
            glm::vec3 sky_color = glm::mix(lower_sky, upper_sky, sky_ground_factor);

            // 2. Нарны туяа болон гэрэлтэлт (Mie Scattering approx)
            // Sun Glow (Нарны эргэн тойрон дахь зөөлөн туяа)
            float glow_strength = std::pow(glm::clamp(cosGamma, 0.0f, 1.0f), 12.0f);
            glm::vec3 sun_glow_color = glm::vec3(1.0f, 0.8f, 0.4f) * 4.0f * sun_height;
            sky_color += sun_glow_color * glow_strength;

            // Sun Disk (Нарны биет)
            if (cosGamma > 0.9998f) {
                sky_color = glm::vec3(4.0f, 3.5f, 3.0f); // Нарны өнгийг илүү тэнцвэртэй болгов
            } else if (cosGamma > 0.9992f) {
                float edge = (cosGamma - 0.9992f) / (0.9998f - 0.9992f);
                sky_color = glm::mix(sky_color, glm::vec3(3.0f, 2.5f, 1.5f), edge);
            }

            return sky_color;
        }

        glm::vec3 sun_direction;
    };

    // ==========================================
    // CORE CLASSES
    // ==========================================

    class ZBuffer
    {
    public:
        ZBuffer(int width, int height, float z_near, float z_far)
            : width_(width), height_(height), z_near(z_near), z_far(z_far),
              depth_(width, height, std::numeric_limits<float>::max())
        {}

        bool test_and_set_depth(int x, int y, float depth)
        {
            if (x < 0 || x >= width_ || y < 0 || y >= height_) return false;
            float &d = depth_.at(x, y);
            if (depth < d)
            {
                d = depth;
                return true;
            }
            return false;
        }

        void clear()
        {
            depth_.clear(std::numeric_limits<float>::max());
        }

        // Compatibility + future migration helpers
        inline int get_width()  const { return width_; }
        inline int get_height() const { return height_; }

        inline DepthBuffer& buffer() { return depth_; }
        inline const DepthBuffer& buffer() const { return depth_; }

        inline float get_depth_at(int x, int y) const {
            if (!depth_.in_bounds(x,y)) return std::numeric_limits<float>::max();
            return depth_.at(x,y);
        }

    private:
        int   width_;
        int   height_;
        float z_near;
        float z_far;

        DepthBuffer depth_;
    };

    class Canvas
    {
    public:
        Canvas(int width, int height)
            : width_(width), height_(height),
              color_(width, height, shs::Color{0, 0, 0, 255})
        {}

        Canvas(int width, int height, shs::Pixel bg_pixel)
            : width_(width), height_(height),
              color_(width, height, bg_pixel.get_color())
        {}

        Canvas(int width, int height, shs::Color bg_color)
            : width_(width), height_(height),
              color_(width, height, bg_color)
        {}

        ~Canvas() {}

        int get_width() const { return width_; }
        int get_height() const { return height_; }

        shs::Color get_color_at(int x, int y) {
            if (!color_.in_bounds(x,y)) return {0,0,0,0};
            return color_.at(x,y);
        }

        shs::Color get_color_at(int x, int y) const {
            if (!color_.in_bounds(x,y)) return {0,0,0,0};
            return color_.at(x,y);
        }

        inline void draw_pixel(int x, int y, shs::Color color)
        {
            if (color_.in_bounds(x,y))
                color_.at(x,y) = color;
        }

        void draw_pixel(int x, int y, shs::Pixel pixel) {
            draw_pixel(x, y, pixel.get_color());
        }

        inline void draw_pixel_screen_space(int x_screen, int y_screen, shs::Color color)
        {
            int y_canvas = (height_ - 1) - y_screen;
            draw_pixel(x_screen, y_canvas, color);
        }

        // --- STATIC HELPER METHODS ---

        static void draw_pixel(shs::Canvas &canvas, int x, int y, shs::Color color) {
            canvas.draw_pixel(x, y, color);
        }

        static void draw_pixel(shs::Canvas &canvas, int x, int y, shs::Pixel pixel) {
            canvas.draw_pixel(x, y, pixel.get_color());
        }

        static void fill_pixel(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Pixel pixel)
        {
            for (int x = x0; x < x1; x++) {
                for (int y = y0; y < y1; y++) {
                    canvas.draw_pixel(x, y, pixel.get_color());
                }
            }
        }

        static void fill_random_pixel(shs::Canvas &canvas, int x0, int y0, int x1, int y1) {
            for (int x = x0; x < x1; x++) {
                for (int y = y0; y < y1; y++) {
                    canvas.draw_pixel(x, y, shs::Pixel::random_pixel());
                }
            }
        }

        static void draw_line(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Pixel pixel)
        {
            shs::Color color = pixel.get_color();
            bool steep = false;
            if (std::abs(x0 - x1) < std::abs(y0 - y1)) {
                std::swap(x0, y0); std::swap(x1, y1);
                steep = true;
            }
            if (x0 > x1) {
                std::swap(x0, x1); std::swap(y0, y1);
            }
            int dx = x1 - x0;
            int dy = y1 - y0;
            int derror2 = std::abs(dy) * 2;
            int error2 = 0;
            int y = y0;
            for (int x = x0; x <= x1; x++) {
                if (steep) canvas.draw_pixel(y, x, color);
                else       canvas.draw_pixel(x, y, color);
                error2 += derror2;
                if (error2 > dx) {
                    y += (y1 > y0 ? 1 : -1);
                    error2 -= dx * 2;
                }
            }
        }

        static void draw_circle_poly(shs::Canvas &canvas, int cx, int cy, int r, int segments, shs::Pixel pixel) {
            if (r <= 0) return;
            if (segments < 3) segments = 3;

            shs::Color color = pixel.get_color();

            const double two_pi = 6.283185307179586;
            const double step   = two_pi / (double)segments;

            int x_prev = cx + (int)std::lround((double)r * std::cos(0.0));
            int y_prev = cy + (int)std::lround((double)r * std::sin(0.0));

            for (int i = 1; i <= segments; i++) {
                double a = step * (double)i;
                int x = cx + (int)std::lround((double)r * std::cos(a));
                int y = cy + (int)std::lround((double)r * std::sin(a));

                shs::Canvas::draw_line(canvas, x_prev, y_prev, x, y, shs::Pixel(color));

                x_prev = x;
                y_prev = y;
            }
        }


        inline static glm::vec3 barycentric_coordinate(const glm::vec2 &P, const std::vector<glm::vec2> &triangle_vertices)
        {
            const glm::vec2 &A = triangle_vertices[0];
            const glm::vec2 &B = triangle_vertices[1];
            const glm::vec2 &C = triangle_vertices[2];

            glm::vec2 v0 = B - A;
            glm::vec2 v1 = C - A;
            glm::vec2 v2 = P - A;

            float d00 = glm::dot(v0, v0);
            float d01 = glm::dot(v0, v1);
            float d11 = glm::dot(v1, v1);
            float d20 = glm::dot(v2, v0);
            float d21 = glm::dot(v2, v1);

            float denom = d00 * d11 - d01 * d01;
            if (std::abs(denom) < 1e-5) return glm::vec3(-1, 1, 1);

            float v = (d11 * d20 - d01 * d21) / denom;
            float w = (d00 * d21 - d01 * d20) / denom;
            float u = 1.0f - v - w;

            //return glm::vec3(v, w, u);
            return glm::vec3(u, v, w);
        }

        // ======================================================
        // LEGACY COMPATIBILITY HELPERS 
        // ======================================================

        static void draw_triangle(
            shs::Canvas &canvas,
            std::vector<glm::vec2> &vertices_screen,
            shs::Pixel pixel)
        {
            int max_x = canvas.get_width() - 1;
            int max_y = canvas.get_height() - 1;
            glm::vec2 bboxmin(max_x, max_y);
            glm::vec2 bboxmax(0, 0);

            for (int i = 0; i < 3; i++) {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, vertices_screen[i]));
                bboxmax = glm::min(glm::vec2(max_x, max_y), glm::max(bboxmax, vertices_screen[i]));
            }

            glm::ivec2 p;
            for (p.x = (int)bboxmin.x; p.x <= (int)bboxmax.x; p.x++) {
                for (p.y = (int)bboxmin.y; p.y <= (int)bboxmax.y; p.y++) {
                    glm::vec3 bc_screen = barycentric_coordinate(glm::vec2(p.x, p.y), vertices_screen);
                    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
                    canvas.draw_pixel_screen_space(p.x, p.y, pixel.get_color());
                }
            }
        }

        static void draw_triangle_color_approximation(
            shs::Canvas &canvas,
            std::vector<glm::vec2> &vertices_screen,
            std::vector<glm::vec3> &colors)
        {
            int max_x = canvas.get_width() - 1;
            int max_y = canvas.get_height() - 1;
            glm::vec2 bboxmin(max_x, max_y);
            glm::vec2 bboxmax(0, 0);

            for (int i = 0; i < 3; i++) {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, vertices_screen[i]));
                bboxmax = glm::min(glm::vec2(max_x, max_y), glm::max(bboxmax, vertices_screen[i]));
            }

            glm::ivec2 p;
            for (p.x = (int)bboxmin.x; p.x <= (int)bboxmax.x; p.x++) {
                for (p.y = (int)bboxmin.y; p.y <= (int)bboxmax.y; p.y++) {
                    glm::vec3 bc = barycentric_coordinate(glm::vec2(p.x, p.y), vertices_screen);
                    if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                    glm::vec3 interpolated_color = bc.x * colors[0] + bc.y * colors[1] + bc.z * colors[2];
                    glm::vec4 rescaled = glm::clamp(glm::vec4(interpolated_color, 1.0f), 0.0f, 1.0f) * 255.0f;

                    canvas.draw_pixel_screen_space(p.x, p.y, shs::Color{
                        (uint8_t)rescaled.x, (uint8_t)rescaled.y, (uint8_t)rescaled.z, (uint8_t)rescaled.w});
                }
            }
        }

        static void draw_triangle_flat_shading(
            shs::Canvas &canvas,
            shs::ZBuffer &z_buffer,
            std::vector<glm::vec3> &vertices_screen,
            std::vector<glm::vec3> &view_space_normals,
            glm::vec3 &light_direction_in_view_space)
        {
            int max_x = canvas.get_width() - 1;
            int max_y = canvas.get_height() - 1;

            std::vector<glm::vec2> v2d;
            v2d.reserve(3);
            glm::vec2 bboxmin(max_x, max_y);
            glm::vec2 bboxmax(0, 0);

            for(const auto& v : vertices_screen) v2d.push_back(glm::vec2(v.x, v.y));

            for (int i = 0; i < 3; i++) {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, v2d[i]));
                bboxmax = glm::min(glm::vec2(max_x, max_y), glm::max(bboxmax, v2d[i]));
            }

            float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) -
                         (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
            if (area <= 0) return;

            glm::vec3 normal    = glm::normalize(view_space_normals[0]);
            glm::vec3 light_dir = glm::normalize(light_direction_in_view_space);
            float ambient_strength = 0.15f;
            float diffuse_strength = glm::max(0.0f, glm::dot(normal, light_dir));
            float total_intensity = ambient_strength + diffuse_strength;
            if (total_intensity > 1.0f) total_intensity = 1.0f;

            shs::Color color_to_draw = {
                (uint8_t)(255 * total_intensity),
                (uint8_t)(255 * total_intensity),
                (uint8_t)(255 * total_intensity),
                255
            };

            glm::ivec2 p;
            for (p.x = (int)bboxmin.x; p.x <= (int)bboxmax.x; p.x++)
            {
                for (p.y = (int)bboxmin.y; p.y <= (int)bboxmax.y; p.y++)
                {
                    glm::vec3 bc = barycentric_coordinate(glm::vec2(p.x + 0.5f, p.y + 0.5f), v2d);
                    if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                    float z = bc.x * vertices_screen[0].z + bc.y * vertices_screen[1].z + bc.z * vertices_screen[2].z;
                    if (z_buffer.test_and_set_depth(p.x, p.y, z))
                    {
                        canvas.draw_pixel_screen_space(p.x, p.y, color_to_draw);
                    }
                }
            }
        }

        // PROGRAMMABLE PIPELINE: Vertex болон Fragment Shader ашиглан зурах функц
        static void draw_triangle_pipeline(
            shs::Canvas  &canvas,
            shs::ZBuffer &z_buffer,
            const std::vector<glm::vec3> &vertices,
            const std::vector<glm::vec3> &normals,
            std::function<shs::Varyings(const glm::vec3& pos, const glm::vec3& norm)> vertex_shader,
            std::function<shs::Color(const shs::Varyings& varying)> fragment_shader)
        {
            int max_x = canvas.get_width() - 1;
            int max_y = canvas.get_height() - 1;

            // [VERTEX STAGE]
            shs::Varyings vout[3];
            glm::vec3     screen_coords[3];

            for (int i = 0; i < 3; i++) {
                vout[i]          = vertex_shader(vertices[i], normals[i]);
                screen_coords[i] = clip_to_screen(vout[i].position, canvas.get_width(), canvas.get_height());
            }

            // [RASTER PREP] Bounding box болон Back-face culling
            glm::vec2 bboxmin(max_x, max_y);
            glm::vec2 bboxmax(0, 0);
            std::vector<glm::vec2> v2d = { glm::vec2(screen_coords[0]), glm::vec2(screen_coords[1]), glm::vec2(screen_coords[2]) };

            for (int i = 0; i < 3; i++) {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, v2d[i]));
                bboxmax = glm::min(glm::vec2(max_x, max_y), glm::max(bboxmax, v2d[i]));
            }

            float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) - (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
            if (area <= 0) return;

            // [FRAGMENT STAGE]
            for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
                for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {

                    glm::vec3 bc = barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                    if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                    // Z-Buffer test
                    float z = bc.x * screen_coords[0].z + bc.y * screen_coords[1].z + bc.z * screen_coords[2].z;
                    if (z_buffer.test_and_set_depth(px, py, z)) {

                        // Varyings интерполяци хийх
                        shs::Varyings interpolated;
                        interpolated.normal    = glm::normalize(bc.x * vout[0].normal    + bc.y * vout[1].normal    + bc.z * vout[2].normal);
                        interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;

                        // Fragment shader-ийн эцсийн өнгө
                        shs::Color pixel_color = fragment_shader(interpolated);
                        canvas.draw_pixel_screen_space(px, py, pixel_color);
                    }
                }
            }
        }

        inline static glm::vec3 clip_to_screen(const glm::vec4 &clip_coord, int screen_width, int screen_height)
        {
            glm::vec3 ndc_coord = glm::vec3(clip_coord) / clip_coord.w;

            glm::vec3 screen_coord;
            // Map NDC [-1,+1] -> pixel [0..W-1], [0..H-1] (origin top-left, y down)
            screen_coord.x = (ndc_coord.x + 1.0f) * 0.5f * float(screen_width  - 1);
            screen_coord.y = (1.0f - ndc_coord.y) * 0.5f * float(screen_height - 1);

            // Keep z in NDC (LH typically 0..1)
            screen_coord.z = ndc_coord.z;

            return screen_coord;
        }

        static void copy_to_SDLSurface(SDL_Surface *surface, shs::Canvas *canvas)
        {
            if (!surface || !canvas) return;

            uint8_t* target_pixels = (uint8_t*)surface->pixels;
            int pitch = surface->pitch;
            int w = canvas->get_width();
            int h = canvas->get_height();

            for (int y = 0; y < h; ++y) {
                // SDL y is top-down, Canvas is bottom-up
                int sdl_y = (h - 1 - y);
                uint32_t* row = (uint32_t*)(target_pixels + sdl_y * pitch);

                for (int x = 0; x < w; ++x) {
                    shs::Color c = canvas->get_color_at(x, y);
                    row[x] = SDL_MapRGBA(surface->format, c.r, c.g, c.b, c.a);
                }
            }
        }

        SDL_Surface *create_sdl_surface()
        {
            #if SDL_BYTEORDER == SDL_BIG_ENDIAN
                return SDL_CreateRGBSurface(0, width_, height_, 32, 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff);
            #else
                return SDL_CreateRGBSurface(0, width_, height_, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
            #endif
        }

        void save_png(const std::string &filename) {
            (void)filename;
            std::cout << "Warning: PNG save not implemented in header-only version." << std::endl;
        }

        // Аажмаар шилжихэд хэрэгтэй 
        inline ColorBuffer& buffer() { return color_; }
        inline const ColorBuffer& buffer() const { return color_; }

    private:
        int width_;
        int height_;
        ColorBuffer color_;
    };

    // ==========================================
    // RENDER TARGET ТӨРЛҮҮД
    // ==========================================
    struct RT_Color {
        shs::Canvas color;

        RT_Color(int w, int h, shs::Color clear = {0,0,0,255})
            : color(w, h, clear) {}

        inline void clear(shs::Color c) {
            color.buffer().clear(c);
        }
    };

    struct RT_ColorDepth {
        shs::Canvas  color;
        shs::ZBuffer depth;

        RT_ColorDepth(int w, int h, float zn, float zf, shs::Color clear = {0,0,0,255})
            : color(w, h, clear),
            depth(w, h, zn, zf) {}

        inline void clear(shs::Color c) {
            color.buffer().clear(c);
            depth.clear();
        }
    };

    struct RT_ColorDepthVelocity {
        shs::Canvas            color;
        shs::ZBuffer           depth;
        shs::Buffer<glm::vec2> velocity; // Canvas-space velocity (pixels, +Y up)

        RT_ColorDepthVelocity(int w, int h, float zn, float zf, shs::Color clear = {0,0,0,255})
            : color(w, h, clear),
            depth(w, h, zn, zf),
            velocity(w, h, glm::vec2(0.0f)) {}

        inline void clear(shs::Color c) {
            color.buffer().clear(c);
            depth.clear();
            velocity.clear(glm::vec2(0.0f));
        }
    };

    // RENDER TARGET (Color + Depth багцалсан үндсэн төрөл)
    using RenderTarget = shs::RT_ColorDepth; 


    // SDL_image ашиглан зураг уншиж Texture2D болгох
    // flip_y=true бол SDL (top-left) -> Texture (bottom-left) хөрвүүлэлт хийнэ.
    static inline shs::Texture2D load_texture_sdl_image(const std::string &path, bool flip_y = true)
    {
        SDL_Surface *loaded = IMG_Load(path.c_str());
        if (!loaded) {
            std::cout << "IMG_Load failed: " << path << " | " << IMG_GetError() << std::endl;
            return shs::Texture2D();
        }

        SDL_Surface *converted = SDL_ConvertSurfaceFormat(loaded, SDL_PIXELFORMAT_RGBA32, 0);
        SDL_FreeSurface(loaded);
        if (!converted) {
            std::cout << "SDL_ConvertSurfaceFormat failed: " << path << " | " << SDL_GetError() << std::endl;
            return shs::Texture2D();
        }

        int w = converted->w;
        int h = converted->h;

        shs::Texture2D tex(w, h, shs::Color{0,0,0,0});

        uint8_t *src_pixels = (uint8_t*)converted->pixels;
        int pitch = converted->pitch;

        for (int y = 0; y < h; ++y) {
            int ty = flip_y ? (h - 1 - y) : y;

            uint32_t *row = (uint32_t*)(src_pixels + y * pitch);
            for (int x = 0; x < w; ++x) {
                uint8_t r,g,b,a;
                SDL_GetRGBA(row[x], converted->format, &r, &g, &b, &a);
                tex.texels.at(x, ty) = shs::Color{r,g,b,a};
            }
        }

        SDL_FreeSurface(converted);
        return tex;
    }

    // Тоон утгыг хязгаарлах (Integer clamp)
    static inline int clamp_i(int v, int lo, int hi) {
        return (v < lo) ? lo : (v > hi ? hi : v);
    }

    // Альфа холилтын функц (Alpha blend)
    static inline shs::Color alpha_blend(const shs::Color &dst, const shs::Color &src, uint8_t opacity)
    {
        // final_alpha = src.a * opacity / 255
        uint32_t a = (uint32_t)src.a * (uint32_t)opacity;
        uint32_t fa = a / 255; // 0..255

        if (fa == 0) return dst;
        if (fa == 255) return shs::Color{src.r, src.g, src.b, 255};

        uint32_t inv = 255 - fa;

        shs::Color out;
        out.r = (uint8_t)((src.r * fa + dst.r * inv) / 255);
        out.g = (uint8_t)((src.g * fa + dst.g * inv) / 255);
        out.b = (uint8_t)((src.b * fa + dst.b * inv) / 255);
        out.a = 255;
        return out;
    }

    // ЗУРАГ ХУУЛАХ (Blit)
    // - src_w/src_h == -1 => бүтэн texture ашиглана
    // - dst_w/dst_h <= 0  => 1:1 (dst хэмжээ = src хэмжээ)
    // - blend/filter нь optional
    static inline void image_blit(
        shs::Canvas &dst,
        const shs::Texture2D &src,
        int dst_x, int dst_y,

        int src_x = 0, int src_y = 0, int src_w = -1, int src_h = -1,
        int dst_w = -1, int dst_h = -1,

        uint8_t opacity = 255,
        int blend_mode = shs::Tex::BLEND_ALPHA,
        int filter_mode = shs::Tex::FILTER_NEAREST
    ){
        (void)filter_mode; // nearest only

        if (!src.valid()) return;

        // src rect default
        if (src_w < 0) src_w = src.w;
        if (src_h < 0) src_h = src.h;

        // clamp src rect inside texture
        if (src_x < 0) { src_w += src_x; src_x = 0; }
        if (src_y < 0) { src_h += src_y; src_y = 0; }
        if (src_x + src_w > src.w) src_w = src.w - src_x;
        if (src_y + src_h > src.h) src_h = src.h - src_y;
        if (src_w <= 0 || src_h <= 0) return;

        // dst size default = 1:1
        if (dst_w <= 0) dst_w = src_w;
        if (dst_h <= 0) dst_h = src_h;

        // dst clipping
        int x0 = dst_x;
        int y0 = dst_y;
        int x1 = dst_x + dst_w;
        int y1 = dst_y + dst_h;

        int clip_x0 = 0;
        int clip_y0 = 0;
        int clip_x1 = dst.get_width();
        int clip_y1 = dst.get_height();

        int draw_x0 = clamp_i(x0, clip_x0, clip_x1);
        int draw_y0 = clamp_i(y0, clip_y0, clip_y1);
        int draw_x1 = clamp_i(x1, clip_x0, clip_x1);
        int draw_y1 = clamp_i(y1, clip_y0, clip_y1);

        if (draw_x0 >= draw_x1 || draw_y0 >= draw_y1) return;

        // nearest scale mapping
        // sx = src_x + ( (dx - dst_x) * src_w ) / dst_w
        // sy = src_y + ( (dy - dst_y) * src_h ) / dst_h
        for (int y = draw_y0; y < draw_y1; ++y) {
            int sy = src_y + (int)(((int64_t)(y - dst_y) * (int64_t)src_h) / (int64_t)dst_h);
            if (sy < src_y) sy = src_y;
            if (sy >= src_y + src_h) sy = src_y + src_h - 1;

            for (int x = draw_x0; x < draw_x1; ++x) {
                int sx = src_x + (int)(((int64_t)(x - dst_x) * (int64_t)src_w) / (int64_t)dst_w);
                if (sx < src_x) sx = src_x;
                if (sx >= src_x + src_w) sx = src_x + src_w - 1;

                shs::Color sc = src.texels.at(sx, sy);
                if (blend_mode == shs::Tex::BLEND_NONE || sc.a == 255) {
                    if (opacity == 255) dst.draw_pixel(x, y, sc);
                    else {
                        shs::Color dc = dst.get_color_at(x, y);
                        shs::Color s2 = sc; s2.a = (uint8_t)((uint32_t)sc.a * (uint32_t)opacity / 255);
                        dst.draw_pixel(x, y, alpha_blend(dc, s2, 255));
                    }
                } else {
                    shs::Color dc = dst.get_color_at(x, y);
                    dst.draw_pixel(x, y, alpha_blend(dc, sc, opacity));
                }
            }
        }
    }

    // ==========================================
    // 3D КАМЕР, ОБЬЕКТ БОЛОН ЗАГВАРУУД
    // ==========================================

    class Camera3D
    {
    public:
        Camera3D() {
            this->width  = 10.0; this->height = 10.0;
            this->z_near = 0.1f; this->z_far = 1000.0f;
            this->field_of_view    = 45.0f;
            this->horizontal_angle = 0.0f; this->vertical_angle = 0.0f;
            this->position         = glm::vec3(0.0, 0.0, -5.0);
            this->direction_vector = glm::vec3(0.0, 0.0, 1.0);
            update();
        }

        void update() {
            this->direction_vector = glm::vec3(
                cos(glm::radians(this->vertical_angle)) * sin(glm::radians(this->horizontal_angle)),
                sin(glm::radians(this->vertical_angle)),
                cos(glm::radians(this->vertical_angle)) * cos(glm::radians(this->horizontal_angle)));

            this->direction_vector  = glm::normalize(this->direction_vector);
            glm::vec3 world_up      = glm::vec3(0.0f, 1.0f, 0.0f);
            this->right_vector      = glm::normalize(glm::cross(world_up, this->direction_vector));
            this->up_vector         = glm::normalize(glm::cross(this->direction_vector, this->right_vector));
            this->projection_matrix = glm::perspectiveLH(glm::radians(this->field_of_view), 4.0f/3.0f, this->z_near, this->z_far);
            this->view_matrix       = glm::lookAtLH(this->position, this->position + this->direction_vector, this->up_vector);
        }

        glm::mat4 view_matrix;
        glm::mat4 projection_matrix;
        glm::vec3 position, direction_vector, right_vector, up_vector;
        float horizontal_angle, vertical_angle;
        float width, height, field_of_view, z_near, z_far;
    };

    class AbstractObject3D {
    public:
        virtual ~AbstractObject3D() {}
        virtual void update(float delta_time) = 0;
        virtual void render() = 0;
        virtual glm::mat4 get_world_matrix() = 0;
    };

    class AbstractSceneState {
    public:
        virtual ~AbstractSceneState() {}
        virtual void process() = 0;
    };

    class AbstractSystem {
    public:
        virtual ~AbstractSystem() {}
        virtual void process(float delta_time) = 0;
    };

    // ==========================================
    // Command Pattern
    // ==========================================

    class Command {
    public:
        virtual ~Command() {}
        virtual void execute() = 0;
    };

    class MoveForwardCommand : public shs::Command {
    public:
        MoveForwardCommand(glm::vec3 &pos, glm::vec3 dir, float spd, float dt)
            : position(pos), direction(dir), speed(spd), delta_time(dt) {}
        void execute() override { this->position += this->direction * this->speed * delta_time; }
    private:
        glm::vec3 &position; glm::vec3 direction; float speed, delta_time;
    };

    class MoveBackwardCommand : public shs::Command {
    public:
        MoveBackwardCommand(glm::vec3 &pos, glm::vec3 dir, float spd, float dt)
            : position(pos), direction(dir), speed(spd), delta_time(dt) {}
        void execute() override { this->position -= this->direction * this->speed * delta_time; }
    private:
        glm::vec3 &position; glm::vec3 direction; float speed, delta_time;
    };

    class MoveRightCommand : public shs::Command {
    public:
        MoveRightCommand(glm::vec3 &pos, glm::vec3 right, float spd, float dt)
            : position(pos), right_vector(right), speed(spd), delta_time(dt) {}
        void execute() override { this->position += this->right_vector * this->speed * this->delta_time; }
    private:
        glm::vec3 &position; glm::vec3 right_vector; float speed, delta_time;
    };

    class MoveLeftCommand : public shs::Command {
    public:
        MoveLeftCommand(glm::vec3 &pos, glm::vec3 right, float spd, float dt)
            : position(pos), right_vector(right), speed(spd), delta_time(dt) {}
        void execute() override { this->position -= this->right_vector * this->speed * this->delta_time; }
    private:
        glm::vec3 &position; glm::vec3 right_vector; float speed, delta_time;
    };

    class CommandProcessor {
    public:
        void add_command(shs::Command *new_command) { this->commands.push(new_command); }
        void process() {
            while (!this->commands.empty()) {
                shs::Command *command = this->commands.front();
                command->execute();
                delete command;
                this->commands.pop();
            }
        }
    private:
        std::queue<shs::Command *> commands;
    };

    // ==========================================
    // ТУСЛАХ КЛАССУУД
    // ==========================================
    namespace Util {
        class Obj3DFile {
        public:
            static std::vector<shs::RawTriangle> read_triangles(const std::string& file_path) {
                std::vector<shs::RawTriangle> triangles;
                std::vector<glm::vec3> temp_vertices;
                std::ifstream objFile(file_path.c_str());
                if (!objFile.is_open()) return {};
                std::string line;
                while (std::getline(objFile, line)) {
                    std::stringstream iss(line);
                    std::string type; iss >> type;
                    if (type == "v") {
                        glm::vec3 vertex; iss >> vertex.x >> vertex.y >> vertex.z;
                        temp_vertices.push_back(vertex);
                    } else if (type == "f") {
                        unsigned int v1, v2, v3; std::string s1, s2, s3;
                        auto parse_index = [](const std::string& s) {
                            return std::stoi(s.substr(0, s.find('/')));
                        };
                        iss >> s1 >> s2 >> s3;
                        v1 = parse_index(s1); v2 = parse_index(s2); v3 = parse_index(s3);
                        shs::RawTriangle triangle;
                        triangle.v1 = temp_vertices[v1 - 1];
                        triangle.v2 = temp_vertices[v2 - 1];
                        triangle.v3 = temp_vertices[v3 - 1];
                        triangles.push_back(triangle);
                    }
                }
                return triangles;
            }
        };

        template <typename T>
        class ThreadSafePriorityQueue {
        public:
             void push(const T &value) {
                 std::lock_guard<std::mutex> lock(mutex_);
                 auto it = queue_.begin();
                 while(it != queue_.end()) {
                     if (it->second < value.second) break;
                     ++it;
                 }
                 queue_.insert(it, value);
             }
             std::optional<T> pop() {
                 std::lock_guard<std::mutex> lock(mutex_);
                 if(queue_.empty()) return std::nullopt;
                 T val = queue_.front();
                 queue_.pop_front();
                 return val;
             }
             bool empty() {
                 std::lock_guard<std::mutex> lock(mutex_);
                 return queue_.empty();
             }
        private:
             std::list<T> queue_;
             std::mutex mutex_;
        };
    }

    // ==========================================
    // ОЛОН THREAD АЖИЛЛУУЛАХ СИСТЕМ (Job System)
    // ==========================================

    namespace Job
    {
        static const int PRIORITY_LOW    = 5;
        static const int PRIORITY_NORMAL = 15;
        static const int PRIORITY_HIGH   = 30;

        struct JobEntry {
            std::function<void()> task;
            int priority;

            // max-heap comparator (Өндөр урьтамжтай нь эхэлж гарна)
            bool operator<(const JobEntry& other) const {
                return priority < other.priority;
            }
        };

        class AbstractJobSystem {
        public:
            virtual ~AbstractJobSystem() {}
            virtual void submit(std::pair<std::function<void()>, int> task) = 0;
            std::atomic<bool> is_running{true};
        };

        class ThreadedPriorityJobSystem : public AbstractJobSystem
        {
        public:
            ThreadedPriorityJobSystem(int concurrency_count)
            {
                thread_count = (concurrency_count <= 0) ? 1 : concurrency_count;
                q.reserve(4096);

                for (int i = 0; i < thread_count; ++i) {
                    workers.emplace_back([this] {
                        for (;;) {
                            JobEntry job;

                            {
                                std::unique_lock<std::mutex> lock(q_mtx);
                                q_cv.wait(lock, [this] {
                                    return !is_running.load(std::memory_order_relaxed) || !q.empty();
                                });

                                if (!is_running.load(std::memory_order_relaxed) && q.empty())
                                    return;

                                std::pop_heap(q.begin(), q.end());
                                job = std::move(q.back());
                                q.pop_back();
                            }

                            job.task();
                        }
                    });
                }

                std::cout << "STATUS : Job System started with " << thread_count << " threads." << std::endl;
            }

            ~ThreadedPriorityJobSystem()
            {
                is_running.store(false, std::memory_order_relaxed);
                q_cv.notify_all();
                for (auto &t : workers) if (t.joinable()) t.join();
            }

            void submit(std::pair<std::function<void()>, int> task) override
            {
                {
                    std::lock_guard<std::mutex> lock(q_mtx);
                    q.push_back({std::move(task.first), task.second});
                    std::push_heap(q.begin(), q.end());
                }
                q_cv.notify_one();
            }

            void submit(std::function<void()> task)
            {
                submit({std::move(task), PRIORITY_NORMAL});
            }

        private:
            int thread_count = 1;
            std::vector<std::thread> workers;

            std::mutex q_mtx;
            std::condition_variable q_cv;
            std::vector<JobEntry> q;
        };

        class WaitGroup
        {
        public:
            WaitGroup() : counter(0) {}

            inline void reset()
            {
                counter.store(0, std::memory_order_relaxed);
            }

            inline void add(int n = 1)
            {
                counter.fetch_add(n, std::memory_order_relaxed);
            }

            inline void done()
            {
                if (counter.fetch_sub(1, std::memory_order_release) == 1) {
                    std::lock_guard<std::mutex> lock(mtx);
                    cv.notify_all();
                }
            }

            inline void wait()
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this] {
                    return counter.load(std::memory_order_acquire) == 0;
                });
            }

        private:
            std::atomic<int> counter;
            std::mutex mtx;
            std::condition_variable cv;
        };

        using ThreadedLocklessPriorityJobSystem = ThreadedPriorityJobSystem;
        using ThreadedLocklessJobSystem         = ThreadedPriorityJobSystem;
        using ThreadedJobSystem                 = ThreadedPriorityJobSystem;
    }

} // namespace shs
