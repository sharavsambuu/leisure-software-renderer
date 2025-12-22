/*
    CUSTOM SOFTWARE RENDERER (SHS) 
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
    // BASIC DATA STRUCTURES
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

    // Shader-үүд хооронд дамжих өгөгдлийн бүтэц (Interpolated values)
    struct Varyings {
        glm::vec4 position;
        glm::vec3 normal;
        glm::vec3 world_pos;
        glm::vec2 uv;
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
    // CORE CLASSES
    // ==========================================

    class ZBuffer
    {
    public:
        ZBuffer(int width, int height, float z_near, float z_far) 
            : width(width), height(height), z_near(z_near), z_far(z_far)
        {
            this->depth_buffer.resize(width * height, std::numeric_limits<float>::max());
        }

        bool test_and_set_depth(int x, int y, float depth)
        {
            if (x < 0 || x >= width || y < 0 || y >= height) return false;
            int index = y * width + x;
            if (depth < this->depth_buffer[index])
            {
                this->depth_buffer[index] = depth;
                return true;
            }
            return false;
        }
        
        void clear()
        {
            std::fill(this->depth_buffer.begin(), this->depth_buffer.end(), std::numeric_limits<float>::max());
        }

    private:
        std::vector<float> depth_buffer;
        int width;
        int height;
        float z_near;
        float z_far;
    };

    class Canvas
    {
    public:
        Canvas(int width, int height) : width(width), height(height)
        {
            this->canvas.resize(width * height, shs::Color{0, 0, 0, 255});
        }
        Canvas(int width, int height, shs::Pixel bg_pixel) : width(width), height(height)
        {
             this->canvas.resize(width * height, bg_pixel.get_color());
        }
        Canvas(int width, int height, shs::Color bg_color) : width(width), height(height)
        {
             this->canvas.resize(width * height, bg_color);
        }

        ~Canvas() {}

        int get_width() const { return this->width; }
        int get_height() const { return this->height; }

        shs::Color get_color_at(int x, int y) {
            if (x < 0 || x >= width || y < 0 || y >= height) return {0,0,0,0};
            return this->canvas[y * width + x];
        }

        inline void draw_pixel(int x, int y, shs::Color color)
        {
            if (x >= 0 && x < width && y >= 0 && y < height)
            {
                this->canvas[y * width + x] = color;
            }
        }
        
        void draw_pixel(int x, int y, shs::Pixel pixel) {
            draw_pixel(x, y, pixel.get_color());
        }

        inline void draw_pixel_screen_space(int x_screen, int y_screen, shs::Color color)
        {
            int y_canvas = (height - 1) - y_screen;
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

            return glm::vec3(v, w, u); 
        }

        static void draw_triangle(shs::Canvas &canvas, std::vector<glm::vec2> &vertices_screen, shs::Pixel pixel)
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
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++) {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++) {
                    glm::vec3 bc_screen = barycentric_coordinate(glm::vec2(p.x, p.y), vertices_screen);
                    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0) continue;
                    canvas.draw_pixel_screen_space(p.x, p.y, pixel.get_color());
                }
            }
        }

        static void draw_triangle_color_approximation(shs::Canvas &canvas, std::vector<glm::vec2> &vertices_screen, std::vector<glm::vec3> &colors)
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
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++) {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++) {
                    glm::vec3 bc = barycentric_coordinate(glm::vec2(p.x, p.y), vertices_screen);
                    if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                    glm::vec3 interpolated_color = bc.x * colors[0] + bc.y * colors[1] + bc.z * colors[2];
                    glm::vec4 rescaled = glm::clamp(glm::vec4(interpolated_color, 1.0f), 0.0f, 1.0f) * 255.0f;

                    canvas.draw_pixel_screen_space(p.x, p.y, shs::Color{
                        (uint8_t)rescaled.x, (uint8_t)rescaled.y, (uint8_t)rescaled.z, (uint8_t)rescaled.w});
                }
            }
        }

        static void draw_triangle_flat_shading(shs::Canvas &canvas, shs::ZBuffer &z_buffer, std::vector<glm::vec3> &vertices_screen, std::vector<glm::vec3> &view_space_normals, glm::vec3 &light_direction_in_view_space)
        {
            int max_x = canvas.get_width() - 1;
            int max_y = canvas.get_height() - 1;

            std::vector<glm::vec2> v2d;
            v2d.reserve(3);
            glm::vec2 bboxmin(max_x, max_y);
            glm::vec2 bboxmax(0, 0);

            for(const auto& v : vertices_screen) {
                v2d.push_back(glm::vec2(v.x, v.y));
            }
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
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++)
            {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++)
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
            shs::Canvas &canvas, 
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
            screen_coord.x = (ndc_coord.x + 1.0f) * 0.5f * screen_width;
            screen_coord.y = (1.0f - ndc_coord.y) * 0.5f * screen_height;
            screen_coord.z = clip_coord.w; 
            return screen_coord;
        }

        static void copy_to_SDLSurface(SDL_Surface *surface, shs::Canvas *canvas)
        {
            if (!surface || !canvas) return;
            uint32_t* pixels = (uint32_t*)surface->pixels;
            int w = canvas->get_width();
            int h = canvas->get_height();

            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    shs::Color c = canvas->get_color_at(x, y);
                    int sdl_y = (h - 1 - y);
                    int sdl_index = sdl_y * w + x;
                    pixels[sdl_index] = SDL_MapRGBA(surface->format, c.r, c.g, c.b, c.a);
                }
            }
        }

        SDL_Surface *create_sdl_surface()
        {
            #if SDL_BYTEORDER == SDL_BIG_ENDIAN
                return SDL_CreateRGBSurface(0, width, height, 32, 0xff000000, 0x00ff0000, 0x0000ff00, 0x000000ff);
            #else
                return SDL_CreateRGBSurface(0, width, height, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
            #endif
        }
        
        void save_png(const std::string &filename) {
             (void)filename; 
             std::cout << "Warning: PNG save not implemented in header-only version." << std::endl;
        }

    private:
        std::vector<shs::Color> canvas;
        int width;
        int height;
    };

    // ==========================================
    // 3D SCENE CLASSES
    // ==========================================

    class Camera3D
    {
    public:
        Camera3D() {
            this->width = 10.0; this->height = 10.0;
            this->z_near = 0.1f; this->z_far = 1000.0f;
            this->field_of_view = 45.0f;
            this->horizontal_angle = 0.0f; this->vertical_angle = 0.0f;
            this->position = glm::vec3(0.0, 0.0, -5.0); 
            this->direction_vector = glm::vec3(0.0, 0.0, 1.0); 
            update();
        }

        void update() {
            this->direction_vector = glm::vec3(
                cos(glm::radians(this->vertical_angle)) * sin(glm::radians(this->horizontal_angle)),
                sin(glm::radians(this->vertical_angle)),
                cos(glm::radians(this->vertical_angle)) * cos(glm::radians(this->horizontal_angle)));
            
            this->direction_vector = glm::normalize(this->direction_vector);
            glm::vec3 world_up = glm::vec3(0.0f, 1.0f, 0.0f);
            this->right_vector = glm::normalize(glm::cross(world_up, this->direction_vector));
            this->up_vector = glm::normalize(glm::cross(this->direction_vector, this->right_vector));
            this->projection_matrix = glm::perspectiveLH(glm::radians(this->field_of_view), 4.0f/3.0f, this->z_near, this->z_far);
            this->view_matrix = glm::lookAtLH(this->position, this->position + this->direction_vector, this->up_vector);
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
    // COMMAND PATTERN
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
    // UTILITIES
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
    // JOB SYSTEM (Thread Safe)
    // ==========================================

    namespace Job
    {
        static const int PRIORITY_LOW    = 5;
        static const int PRIORITY_NORMAL = 15;
        static const int PRIORITY_HIGH   = 30;

        class AbstractJobSystem {
        public:
            virtual ~AbstractJobSystem(){};
            virtual void submit(std::pair<std::function<void()>, int> task) = 0;
            std::atomic<bool> is_running{true};
        };

        class ThreadedPriorityJobSystem : public shs::Job::AbstractJobSystem
        {
        public:
            ThreadedPriorityJobSystem(int concurrency_count) {
                this->concurrency_count = concurrency_count;
                for (int i = 0; i < this->concurrency_count; ++i) {
                    this->workers.emplace_back([this] {
                        while (this->is_running) {
                            auto task_pair = this->job_queue.pop();
                            if (task_pair.has_value()) {
                                task_pair.value().first();
                            } else {
                                std::this_thread::sleep_for(std::chrono::microseconds(100));
                            }
                        } 
                    });
                }
                std::cout << "STATUS : Job System started with " << concurrency_count << " threads." << std::endl;
            }
            
            ~ThreadedPriorityJobSystem() {
                this->is_running = false;
                for (auto &worker : this->workers) {
                    if (worker.joinable()) worker.join();
                }
            }
            
            void submit(std::pair<std::function<void()>, int> task) override {
                this->job_queue.push(task);
            }
            
            void submit(std::function<void()> task) {
                 this->job_queue.push({task, PRIORITY_NORMAL});
            }

        private:
            int concurrency_count;
            std::vector<std::thread> workers;
            shs::Util::ThreadSafePriorityQueue<std::pair<std::function<void()>, int>> job_queue;
        };

        using ThreadedLocklessPriorityJobSystem = ThreadedPriorityJobSystem;
        using ThreadedLocklessJobSystem         = ThreadedPriorityJobSystem;
        using ThreadedJobSystem                 = ThreadedPriorityJobSystem;
    }
}