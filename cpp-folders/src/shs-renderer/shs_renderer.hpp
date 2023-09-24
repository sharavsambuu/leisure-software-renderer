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
#include <optional>
#include <atomic>
#include <utility>
#include <functional>
#include <thread>
#include <mutex>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/constants.hpp>


namespace shs
{

    struct Color
    {
        std::uint8_t r;
        std::uint8_t g;
        std::uint8_t b;
        std::uint8_t a;
    };

    struct RawTriangle
    {
        glm::vec3 v1;
        glm::vec3 v2;
        glm::vec3 v3;
    };


    class Pixel
    {
    public:
        Pixel()
        {
        }
        Pixel(shs::Color color)
        {
            this->color = color;
        }
        Pixel(std::uint8_t r, std::uint8_t g, std::uint8_t b)
        {
            this->color = shs::Color{r, g, b, 255};
        }
        Pixel(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            this->color = shs::Color{r, g, b, a};
        }
        ~Pixel()
        {
        }

        void change_color(shs::Color color)
        {
            this->color.r = color.r;
            this->color.g = color.g;
            this->color.b = color.b;
            this->color.a = color.a;
        }
        void change_color(std::uint8_t r, std::uint8_t g, std::uint8_t b)
        {
            this->color.r = r;
            this->color.g = g;
            this->color.b = b;
        }
        void change_color(std::uint8_t r, std::uint8_t g, std::uint8_t b, std::uint8_t a)
        {
            this->color.r = r;
            this->color.g = g;
            this->color.b = b;
            this->color.a = a;
        }

        void change_red_channel(std::uint8_t r)
        {
            this->color.r = r;
        }
        void change_green_channel(std::uint8_t g)
        {
            this->color.g = g;
        }
        void change_blue_channel(std::uint8_t b)
        {
            this->color.b = b;
        }
        void change_alpha_channel(std::uint8_t a)
        {
            this->color.a = a;
        }

        std::uint8_t get_red_channel()
        {
            return this->color.r;
        }
        std::uint8_t get_green_channel()
        {
            return this->color.g;
        }
        std::uint8_t get_blue_channel()
        {
            return this->color.b;
        }
        std::uint8_t get_alpha_channel()
        {
            return this->color.a;
        }

        void set_color(shs::Color color)
        {
            this->color = color;
        }
        shs::Color get_color()
        {
            return this->color;
        }

        static shs::Pixel red_pixel()
        {
            return shs::Pixel{255, 0, 0, 255};
        }
        static shs::Pixel green_pixel()
        {
            return shs::Pixel{0, 255, 0, 255};
        }
        static shs::Pixel blue_pixel()
        {
            return shs::Pixel{0, 0, 255, 255};
        }
        static shs::Pixel black_pixel()
        {
            return shs::Pixel{0, 0, 0, 255};
        }
        static shs::Pixel white_pixel()
        {
            return shs::Pixel{255, 255, 255, 255};
        }
        static shs::Pixel random_pixel()
        {
            return shs::Pixel{
                uint8_t(rand() % 256),
                uint8_t(rand() % 256),
                uint8_t(rand() % 256),
                uint8_t(255)};
        }

    private:
        shs::Color color;
    };


    class ZBuffer
    {
    public:
        ZBuffer(int width, int height, float z_near, float z_far)
            : width(width), height(height), z_near(z_near), z_far(z_far)
        {
            this->depth_buffer.resize(width, std::vector<float>(height, std::numeric_limits<float>::max()));
        }
        ~ZBuffer()
        {
        }
        bool test_and_set_depth(int x, int y, float fragment_depth)
        {
            if (x >= 0 && x < this->width && y >= 0 && y < this->height)
            {
                float normalized_depth = (fragment_depth - this->z_near) / (this->z_far - z_near);
                normalized_depth = std::min(std::max(normalized_depth, 0.0f), 1.0f);
                if (normalized_depth < this->depth_buffer[x][y])
                {
                    this->depth_buffer[x][y] = normalized_depth;
                    return true; // Fragment is visible.
                }
            }
            return false; // Fragment is not visible.
        }
        void clear()
        {
            for (int x = 0; x < this->width; ++x)
            {
                for (int y = 0; y < this->height; ++y)
                {
                    this->depth_buffer[x][y] = std::numeric_limits<float>::max();
                }
            }
        }

    private:
        std::vector<std::vector<float>> depth_buffer;
        int   width;
        int   height;
        float z_near;
        float z_far;
    };


    class Canvas
    {
    public:
        Canvas()
        {
        }
        Canvas(int width, int height)
        {
            this->width = width;
            this->height = height;
            srand(static_cast<unsigned>(time(nullptr))); // seeding
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; x++)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; y++)
                {
                    this->canvas[x][y].r = rand() % 256;
                    this->canvas[x][y].g = rand() % 256;
                    this->canvas[x][y].b = rand() % 256;
                    this->canvas[x][y].a = 255;
                }
            }
        }
        Canvas(int width, int height, shs::Pixel pixel)
        {
            this->width = width;
            this->height = height;
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; x++)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; y++)
                {
                    this->canvas[x][y].r = pixel.get_red_channel();
                    this->canvas[x][y].g = pixel.get_green_channel();
                    this->canvas[x][y].b = pixel.get_blue_channel();
                    this->canvas[x][y].a = pixel.get_alpha_channel();
                }
            }
        }
        Canvas(int width, int height, shs::Color color)
        {
            this->width = width;
            this->height = height;
            this->canvas.resize(this->width);
            for (int x = 0; x < this->width; x++)
            {
                this->canvas[x].resize(this->height);
                for (int y = 0; y < this->height; y++)
                {
                    this->canvas[x][y].r = color.r;
                    this->canvas[x][y].g = color.g;
                    this->canvas[x][y].b = color.b;
                    this->canvas[x][y].a = color.a;
                }
            }
        }
        ~Canvas()
        {
        }

        int get_width()
        {
            return this->width;
        }
        int get_height()
        {
            return this->height;
        }

        shs::Color get_color_at(int x, int y)
        {
            return this->canvas[x][y];
        }
        static shs::Color get_color_at(shs::Canvas &canvas, int x, int y)
        {
            return canvas.get_color_at(x, y);
        }
        shs::Pixel get_pixel_at(int x, int y)
        {
            return shs::Pixel(this->canvas[x][y]);
        }
        static shs::Pixel get_pixel_at(shs::Canvas &canvas, int x, int y)
        {
            return canvas.get_pixel_at(x, y);
        }

        inline static glm::ivec2 vec2_screen_to_canvas(glm::ivec2 v_in_screen, int screen_height)
        {
            glm::ivec2 v_in_canvas = v_in_screen;
            v_in_canvas.y = screen_height - v_in_screen.y;
            return v_in_canvas;
        }

        void draw_pixel(int x, int y, shs::Pixel pixel)
        {
            glm::ivec2 loc_in_canvas = shs::Canvas::vec2_screen_to_canvas(glm::ivec2(x, y), this->height);
            loc_in_canvas.x = std::clamp(loc_in_canvas.x, 0, this->width );
            loc_in_canvas.y = std::clamp(loc_in_canvas.y, 0, this->height);
            this->canvas[loc_in_canvas.x][loc_in_canvas.y].r = pixel.get_red_channel();
            this->canvas[loc_in_canvas.x][loc_in_canvas.y].g = pixel.get_green_channel();
            this->canvas[loc_in_canvas.x][loc_in_canvas.y].b = pixel.get_blue_channel();
            this->canvas[loc_in_canvas.x][loc_in_canvas.y].a = pixel.get_alpha_channel();
        };

        static void draw_pixel(shs::Canvas &canvas, int x, int y, shs::Color color)
        {
            canvas.draw_pixel(x, y, shs::Pixel(color));
        };
        static void draw_pixel(shs::Canvas &canvas, int x, int y, shs::Pixel pixel)
        {
            canvas.draw_pixel(x, y, pixel);
        };

        static void fill_pixel(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Pixel pixel)
        {
            for (int x = x0; x < x1; x++)
            {
                for (int y = y0; y < y1; y++)
                {
                    shs::Canvas::draw_pixel(canvas, x, y, pixel);
                }
            }
        }
        static void fill_random_pixel(shs::Canvas &canvas, int x0, int y0, int x1, int y1)
        {
            for (int x = x0; x < x1; x++)
            {
                for (int y = y0; y < y1; y++)
                {
                    shs::Canvas::draw_pixel(canvas, x, y, shs::Pixel::random_pixel());
                }
            }
        }

        static void draw_line(shs::Canvas &canvas, int x0, int y0, int x1, int y1, shs::Pixel pixel)
        {
            x0 = std::clamp(x0, 0, canvas.get_width()-1);
            y0 = std::clamp(y0, 0, canvas.get_height()-1);
            x1 = std::clamp(x1, 0, canvas.get_width()-1);
            y1 = std::clamp(y1, 0, canvas.get_height()-1);

            bool steep = false;
            if (std::abs(x0 - x1) < std::abs(y0 - y1))
            {
                std::swap(x0, y0);
                std::swap(x1, y1);
                steep = true;
            }
            if (x0 > x1)
            {
                std::swap(x0, x1);
                std::swap(y0, y1);
            }
            int dx = x1 - x0;
            int dy = y1 - y0;
            int derror2 = std::abs(dy) * 2;
            int error2 = 0;
            int y = y0;
            for (int x = x0; x <= x1; x++)
            {
                if (steep)
                {
                    shs::Canvas::draw_pixel(canvas, y, x, pixel);
                }
                else
                {
                    shs::Canvas::draw_pixel(canvas, x, y, pixel);
                }
                error2 += derror2;
                if (error2 > dx)
                {
                    y += (y1 > y0 ? 1 : -1);
                    error2 -= dx * 2;
                }
            }
        }

/*
        inline static glm::vec3 barycentric_coordinate(const glm::vec2 &P, const std::vector<glm::vec2> &triangle_vertices)
        {
            if (triangle_vertices.size() != 3)
            {
                // Ensure there are exactly three vertices in the triangle
                throw std::invalid_argument("The triangle must have exactly three vertices.");
            }

            // Extract the vertices A, B, and C
            const glm::vec2 &A = triangle_vertices[0];
            const glm::vec2 &B = triangle_vertices[1];
            const glm::vec2 &C = triangle_vertices[2];

            // Calculate the area of the full triangle
            float areaABC = glm::cross(glm::vec3(B - A, 0), glm::vec3(C - A, 0)).z;

            // Calculate the barycentric coordinates
            glm::vec3 barycentric;

            barycentric.x = glm::cross(glm::vec3(B - P, 0), glm::vec3(C - P, 0)).z / areaABC;
            barycentric.y = glm::cross(glm::vec3(C - P, 0), glm::vec3(A - P, 0)).z / areaABC;
            barycentric.z = 1.0f - barycentric.x - barycentric.y;

            return barycentric;
        }
*/

        inline static glm::vec3 barycentric_coordinate(const glm::vec2 &P, const std::vector<glm::vec2> &triangle_vertices)
        {
            if (triangle_vertices.size() != 3)
            {
                // Ensure there are exactly three vertices in the triangle
                throw std::invalid_argument("The triangle must have exactly three vertices.");
            }

            // Extract the vertices A, B, and C
            const glm::vec2 &A = triangle_vertices[0];
            const glm::vec2 &B = triangle_vertices[1];
            const glm::vec2 &C = triangle_vertices[2];

            // Calculate the area of the full triangle
            float areaABC = glm::cross(glm::vec3(B - A, 0), glm::vec3(C - A, 0)).z;

            // Check for degenerate triangles (zero area)
            if (areaABC == 0.0f)
            {
                throw std::invalid_argument("The triangle is degenerate (zero area).");
            }

            // Calculate the barycentric coordinates
            glm::vec3 barycentric;

            barycentric.x = glm::cross(glm::vec3(B - P, 0), glm::vec3(C - P, 0)).z / areaABC;
            barycentric.y = glm::cross(glm::vec3(C - P, 0), glm::vec3(A - P, 0)).z / areaABC;
            barycentric.z = 1.0f - barycentric.x - barycentric.y;

            return barycentric;
        }

        static void draw_triangle(shs::Canvas &canvas, std::vector<glm::vec2> &vertices, shs::Pixel pixel)
        {
            int max_x = canvas.get_width();
            int max_y = canvas.get_height();

            glm::vec2 bboxmin(canvas.get_width() - 1, canvas.get_height() - 1);
            glm::vec2 bboxmax(0, 0);
            glm::vec2 clamp(canvas.get_width() - 1, canvas.get_height() - 1);

            for (int i = 0; i < 3; i++)
            {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, vertices[i]));
                bboxmax = glm::min(clamp, glm::max(bboxmax, vertices[i]));
            }

            glm::vec2 p;
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++)
            {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++)
                {
                    glm::vec3 bc_screen = shs::Canvas::barycentric_coordinate(p, vertices);

                    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
                        continue;

                    glm::ivec2 p_copy = p;
                    p_copy.x = std::clamp<int>(p_copy.x, 0, max_x);
                    p_copy.y = std::clamp<int>(p_copy.y, 0, max_y);
                    shs::Canvas::draw_pixel(canvas, p_copy.x, p_copy.y, pixel);
                }
            }
        }

        inline static glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec)
        {
            glm::vec4 clamped_value = glm::clamp(input_vec, 0.0f, 1.0f);
            glm::vec4 scaled_value = clamped_value * 255.0f;
            return scaled_value;
        }

        static void draw_triangle_color_approximation(shs::Canvas &canvas, std::vector<glm::vec2> &vertices, std::vector<glm::vec3> &colors)
        {
            int max_x = canvas.get_width();
            int max_y = canvas.get_height();

            glm::vec2 bboxmin(canvas.get_width() - 1, canvas.get_height() - 1);
            glm::vec2 bboxmax(0, 0);
            glm::vec2 clamp(canvas.get_width() - 1, canvas.get_height() - 1);

            for (int i = 0; i < 3; i++)
            {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, vertices[i]));
                bboxmax = glm::min(clamp, glm::max(bboxmax, vertices[i]));
            }

            glm::vec2 p;
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++)
            {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++)
                {
                    glm::vec2 pixel_position(p.x + 0.5f, p.y + 0.5f);
                    glm::vec3 bc_screen = shs::Canvas::barycentric_coordinate(pixel_position, vertices);

                    if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
                        continue;

                    glm::ivec2 p_copy = p;
                    p_copy.x = std::clamp<int>(p_copy.x, 0, max_x);
                    p_copy.y = std::clamp<int>(p_copy.y, 0, max_y);

                    glm::vec3 interpolated_color = bc_screen.x * colors[0] + bc_screen.y * colors[1] + bc_screen.z * colors[2];
                    glm::vec4 rescaled_color     = shs::Canvas::rescale_vec4_1_255(glm::vec4(interpolated_color, 1.0));

                    shs::Canvas::draw_pixel(canvas, p_copy.x, p_copy.y, shs::Color{std::uint8_t(rescaled_color.x), std::uint8_t(rescaled_color.y), std::uint8_t(rescaled_color.z), std::uint8_t(rescaled_color.w)});
                }
            }
        }

        static void draw_triangle_flat_shading(shs::Canvas &canvas, shs::ZBuffer &z_buffer, std::vector<glm::vec3> &vertices_screen, std::vector<glm::vec3> &view_space_normals, glm::vec3 &light_direction_in_view_space)
        {
            int max_x = canvas.get_width();
            int max_y = canvas.get_height();

            glm::vec2 bboxmin(canvas.get_width() - 1, canvas.get_height() - 1);
            glm::vec2 bboxmax(0, 0);
            glm::vec2 clamp(canvas.get_width() - 1, canvas.get_height() - 1);


            std::vector<glm::vec2> vertices_2d;
            std::transform(vertices_screen.begin(), vertices_screen.end(), std::back_inserter(vertices_2d), [](const glm::vec3 &v3){ return glm::vec2(v3.x, v3.y); });

            for (int i = 0; i < 3; i++)
            {
                bboxmin = glm::max(glm::vec2(0), glm::min(bboxmin, vertices_2d[i]));
                bboxmax = glm::min(clamp, glm::max(bboxmax, vertices_2d[i]));
            }

            glm::vec2 p;
            for (p.x = bboxmin.x; p.x <= bboxmax.x; p.x++)
            {
                for (p.y = bboxmin.y; p.y <= bboxmax.y; p.y++)
                {
                    try
                    {
                        glm::vec2 pixel_position(p.x + 0.5f, p.y + 0.5f);
                        glm::vec3 bc_screen = shs::Canvas::barycentric_coordinate(pixel_position, vertices_2d);
                        std::cout << bc_screen.x << " " << bc_screen.y << " " << bc_screen.z << std::endl;

                        if (bc_screen.x < 0 || bc_screen.y < 0 || bc_screen.z < 0)
                            continue;

                        // std::cout << "z value : " << bc_screen.z << std::endl;

                        glm::ivec2 p_copy = p;
                        p_copy.x = std::clamp<int>(p_copy.x, 0, max_x);
                        p_copy.y = std::clamp<int>(p_copy.y, 0, max_y);

                        glm::vec3 interpolated_fragment = bc_screen.x * vertices_screen[0] + bc_screen.y * vertices_screen[1] + bc_screen.z * vertices_screen[2];
                        // std::cout << bc_screen.x << " " << bc_screen.y << " " << bc_screen.z << std::endl;
                        // glm::vec3 interpolated_normal        = bc_screen.x * view_space_normals[0] + bc_screen.y * view_space_normals[1] + bc_screen.z * view_space_normals[2];
                        // std::cout << interpolated_fragment.x << " " << interpolated_fragment.y << " " << interpolated_fragment.z << std::endl;

                        // glm::vec3 normalized_normal          = glm::normalize(interpolated_normal);
                        glm::vec3 normalized_normal = glm::normalize(view_space_normals[0]);
                        glm::vec3 normalized_light_direction = glm::normalize(light_direction_in_view_space);
                        float light_intensity = glm::dot(normalized_light_direction, normalized_normal);
                        // light_intensity       = glm::clamp(light_intensity, 0.0f, 1.0f);
                        if (light_intensity > 0)
                        {
                            glm::vec3 default_color = glm::vec3(1.0f, 1.0f, 1.0f);
                            glm::vec3 shaded_color = default_color * light_intensity;
                            glm::vec4 rescaled_color = shs::Canvas::rescale_vec4_1_255(glm::vec4(shaded_color, 1.0));
                            shs::Canvas::draw_pixel(canvas, p_copy.x, p_copy.y, shs::Color{std::uint8_t(rescaled_color.x), std::uint8_t(rescaled_color.y), std::uint8_t(rescaled_color.z), std::uint8_t(rescaled_color.w)});
                        }
                    }
                    catch (const std::invalid_argument &e)
                    {
                        continue;
                    }
                }
            }
        }

        inline static glm::vec3 clip_to_screen(const glm::vec4 &clip_coord, int screen_width, int screen_height)
        {
            // Normalize the clip space coordinates
            glm::vec3 ndc_coord = glm::vec3(clip_coord.x, clip_coord.y, clip_coord.z) / clip_coord.w;

            // Map the normalized coordinates to screen space
            glm::vec3 screen_coord;
            screen_coord.x = (ndc_coord.x + 1.0f) * 0.5f * screen_width;
            screen_coord.y = (1.0f + ndc_coord.y) * 0.5f * screen_height;
            screen_coord.z = ndc_coord.z;

            return screen_coord;
        }

        static void set_rawcolor_at_SDLSurface(SDL_Surface *surface, int x, int y, Uint32 raw_color)
        {
            Uint32 *pixel = (Uint32 *)((Uint8 *)surface->pixels + y * surface->pitch + x * sizeof(Uint32));
            *pixel = raw_color;
        }
        static void set_color_at_SDLSurface(SDL_Surface *surface, int x, int y, shs::Color color)
        {
            Uint32 raw_color = SDL_MapRGBA(surface->format, color.r, color.g, color.b, color.a);
            shs::Canvas::set_rawcolor_at_SDLSurface(surface, x, y, raw_color);
        }
        static void set_pixel_at_SDLSurface(SDL_Surface *surface, int x, int y, shs::Pixel pixel)
        {
            shs::Canvas::set_color_at_SDLSurface(surface, x, y, pixel.get_color());
        }
        static void copy_to_SDLSurface(SDL_Surface *surface, shs::Canvas *canvas)
        {
            for (int x = 0; x < canvas->get_width(); x++)
            {
                for (int y = 0; y < canvas->get_height(); y++)
                {
                    shs::Color color = canvas->get_color_at(x, y);
                    shs::Canvas::set_color_at_SDLSurface(surface, x, y, color);
                }
            }
        }

        SDL_Surface *create_sdl_surface()
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, this->width, this->height, 32, 0, 0, 0, 0);
            return surface;
        }
        static SDL_Surface *create_sdl_surface(int width, int height)
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, 0, 0, 0, 0);
            return surface;
        }
        static SDL_Surface *create_sdl_surface(int width, int height, shs::Color color)
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, width, height, 32, color.r, color.g, color.b, color.a);
            return surface;
        }
        SDL_Surface *create_sdl_surface(shs::Color color)
        {
            SDL_Surface *surface = SDL_CreateRGBSurface(0, this->width, this->height, 32, color.r, color.g, color.b, color.a);
            return surface;
        }

        void save_png(const std::string &filename)
        {
        }

    private:
        std::vector<std::vector<shs::Color>> canvas;
        int width;
        int height;
    };

    class Camera3D
    {
    public:
        Camera3D()
        {
            // some random default values can be changed them later
            this->width            = 10.0;
            this->height           = 10.0;
            this->z_near           = 0.2;
            this->z_far            = 1000.0;
            this->field_of_view    = 35.0;
            this->horizontal_angle = 0.0;
            this->vertical_angle   = 10.0;
            this->position         = glm::vec3(0.0, 0.0, 0.0);
            this->direction_vector = glm::vec3(0.0, 0.0, 1.0);
        }
        ~Camera3D()
        {
        }

        void update()
        {
            this->direction_vector = glm::vec3(
                cos(glm::radians(this->vertical_angle)) * sin(glm::radians(this->horizontal_angle)),
                sin(glm::radians(this->vertical_angle)),
                cos(glm::radians(this->vertical_angle)) * cos(glm::radians(this->horizontal_angle)));
            this->right_vector = glm::vec3(
                sin(horizontal_angle - glm::pi<float>() / 2.0f),
                0.0,
                cos(horizontal_angle - glm::pi<float>() / 2.0f));
            this->up_vector = glm::cross(this->right_vector, direction_vector);

            this->projection_matrix = glm::perspective(glm::radians(this->field_of_view), this->width / this->height, this->z_near, z_far);
            this->view_matrix = glm::lookAt(this->position, this->position + this->direction_vector, this->up_vector);
        }

        glm::mat4 view_matrix;
        glm::mat4 projection_matrix;

        glm::vec3 position;
        glm::vec3 direction_vector;
        glm::vec3 right_vector;
        glm::vec3 up_vector;

        float horizontal_angle;
        float vertical_angle;

        float width;
        float height;

        float field_of_view;
        float z_near;
        float z_far;

    private:
    };


    class AbstractObject3D
    {
    public:
        virtual void update(float delta_time) = 0;
        virtual void render() = 0;
        virtual glm::mat4 get_world_matrix() = 0;
    };

    class AbstractSceneState
    {
    public:
        virtual void process() = 0;
    };

    class AbstractSystem
    {
    public:
        virtual void process(float delta_time) = 0;
    };

    class Command
    {
        // See https://gameprogrammingpatterns.com/command.html
    public:
        virtual ~Command() {}
        virtual void execute() = 0;
        // virtual void undo   () = 0;
    };

    class MoveForwardCommand : public shs::Command
    {
    public:
        MoveForwardCommand(glm::vec3 &position, glm::vec3 direction, float speed, float delta_time) : position(position), direction(direction), speed(speed), delta_time(delta_time) {}
        void execute() override
        {
            this->position += this->direction * this->speed * delta_time;
        }

    private:
        glm::vec3 &position;
        glm::vec3 direction;
        float speed;
        float delta_time;
    };

    class MoveBackwardCommand : public shs::Command
    {
    public:
        MoveBackwardCommand(glm::vec3 &position, glm::vec3 direction, float speed, float delta_time) : position(position), direction(direction), speed(speed), delta_time(delta_time) {}
        void execute() override
        {
            this->position -= this->direction * this->speed * delta_time;
        }

    private:
        glm::vec3 &position;
        glm::vec3 direction;
        float speed;
        float delta_time;
    };

    class MoveRightCommand : public shs::Command
    {
    public:
        MoveRightCommand(glm::vec3 &position, glm::vec3 right_vector, float speed, float delta_time) : position(position), right_vector(right_vector), speed(speed), delta_time(delta_time) {}
        void execute() override
        {
            this->position += this->right_vector * this->speed * this->delta_time;
        }

    private:
        glm::vec3 &position;
        glm::vec3 right_vector;
        float speed;
        float delta_time;
    };

    class MoveLeftCommand : public shs::Command
    {
    public:
        MoveLeftCommand(glm::vec3 &position, glm::vec3 right_vector, float speed, float delta_time) : position(position), right_vector(right_vector), speed(speed), delta_time(delta_time) {}
        void execute() override
        {
            this->position -= this->right_vector * this->speed * this->delta_time;
        }

    private:
        glm::vec3 &position;
        glm::vec3 right_vector;
        float speed;
        float delta_time;
    };

    class CommandProcessor
    {
    public:
        void add_command(shs::Command *new_command)
        {
            this->commands.push(new_command);
        }
        void process()
        {
            while (!this->commands.empty())
            {
                shs::Command *command = this->commands.front();
                command->execute();
                delete command;
                this->commands.pop();
            }
        }

    private:
        std::queue<shs::Command *> commands;
    };


    namespace Util
    {

        class Obj3DFile
        {
        public:
            Obj3DFile()
            {
            }
            ~Obj3DFile()
            {
            }
            static std::vector<shs::RawTriangle> read_triangles(const std::string& file_path)
            {

                std::cout << "reading triangle data from 3D wavefront obj file : " << file_path.c_str() << std::endl;
                std::ifstream objFile(file_path.c_str());
                if (!objFile.is_open())
                {
                    std::cerr << "Failed to open OBJ file." << std::endl;
                    return {};
                }

                std::vector<shs::RawTriangle> triangles;

                std::string line;
                while (std::getline(objFile, line))
                {
                    if (line.empty() || line[0] == '#')
                    {
                        continue;
                    }
                    std::istringstream iss(line);
                    std::string type;
                    iss >> type;
                    if (type == "f")
                    {
                        shs::RawTriangle triangle;
                        iss >> triangle.v1.x >> triangle.v2.x >> triangle.v3.x;
                        iss >> triangle.v1.y >> triangle.v2.y >> triangle.v3.y;
                        iss >> triangle.v1.z >> triangle.v2.z >> triangle.v3.z;
                        triangles.push_back(triangle);
                    }
                }
                objFile.close();

                return triangles;
            }
        };

        template <typename T>
        class LocklessQueue
        {
        public:
            LocklessQueue()
            {
                head_ = new Node;
                tail_ = head_.load(std::memory_order_relaxed);
            }

            ~LocklessQueue()
            {
                while (Node *old_head = head_.load())
                {
                    head_ = old_head->next;
                    delete old_head;
                }
            }

            void push(const T &value)
            {
                Node *new_node = new Node;
                new_node->data = value;
                new_node->next = nullptr;
                Node *prev_tail = tail_.exchange(new_node, std::memory_order_relaxed);
                prev_tail->next = new_node;
            }

            std::optional<T> pop()
            {
                Node *old_head = head_.load(std::memory_order_relaxed);
                Node *new_head = old_head->next;
                if (new_head)
                {
                    T value = new_head->data;
                    head_ = new_head;
                    delete old_head;
                    return value;
                }
                return std::nullopt; // Queue is empty
            }

        private:
            struct Node
            {
                T data;
                Node *next;
            };

            std::atomic<Node *> head_;
            std::atomic<Node *> tail_;
        };


        template <typename T>
        class LocklessPriorityQueue
        {
        public:
            LocklessPriorityQueue() : head_(nullptr) {}
            void push(const T &value)
            {
                Node *new_node = new Node(value);
                new_node->next = head_.load(std::memory_order_relaxed);
                while (!head_.compare_exchange_weak(new_node->next, new_node,
                                                    std::memory_order_release,
                                                    std::memory_order_relaxed))
                {
                }
            }
            std::optional<T> pop()
            {
                Node *old_head = head_.load(std::memory_order_acquire);
                while (old_head && !head_.compare_exchange_weak(old_head, old_head->next,
                                                                std::memory_order_relaxed,
                                                                std::memory_order_relaxed))
                {
                }
                if (old_head)
                {
                    T value = old_head->data;
                    delete old_head;
                    return value;
                }
                else
                {
                    return std::nullopt;
                }
            }
            long count()
            {
                Node *current = head_.load(std::memory_order_relaxed);
                long count = 0;
                while (current)
                {
                    count++;
                    current = current->next;
                }
                return count;
            }

        private:
            struct Node
            {
                T data;
                Node *next;

                Node(const T &val) : data(val), next(nullptr) {}
            };
            std::atomic<Node *> head_;
        };
    
    }

    namespace Job
    {
        static int PRIORITY_LOW    = 5;
        static int PRIORITY_NORMAL = 15;
        static int PRIORITY_HIGH   = 30;

        class AbstractJobSystem
        {
        public:
            virtual ~AbstractJobSystem(){};
            virtual void submit(std::pair<std::function<void()>, int> task) = 0; // second one is priority
            bool is_running = true;
        };

        class ThreadedJobSystem : public shs::Job::AbstractJobSystem
        {
        public:
            ThreadedJobSystem(int concurrency_count)
            {
                this->concurrency_count = concurrency_count;
                this->workers.reserve(this->concurrency_count);
                for (int i = 0; i < this->concurrency_count; ++i)
                {
                    this->workers[i] = std::thread([this, i] {
                        while (this->is_running)
                        {
                            std::function<void()> task;

                            {
                                std::unique_lock<std::mutex> lock(this->mutex);
                                if (!this->job_queue.empty())
                                {
                                    task = std::move(this->job_queue.front());
                                    this->job_queue.pop();
                                }
                            }

                            if (task)
                            {
                                task();
                            }

                        } 
                    });
                }
                std::cout << "STATUS : Threaded Job system is started." << std::endl;
            }
            ~ThreadedJobSystem()
            {
                std::cout << "STATUS : Threaded Job system is shutting down..." << std::endl;
                for (auto &worker : this->workers)
                {
                    worker.join();
                }
            }
            void submit(std::pair<std::function<void()>, int> task) override
            {
                {
                    std::unique_lock<std::mutex> lock(this->mutex);
                    job_queue.push(std::move(task.first));
                }
            }

        private:
            int concurrency_count;
            std::vector<std::thread> workers;
            std::queue<std::function<void()>> job_queue;
            std::mutex mutex;
        };

        class ThreadedLocklessJobSystem : public shs::Job::AbstractJobSystem
        {
        public:
            ThreadedLocklessJobSystem(int concurrency_count)
            {

                this->concurrency_count = concurrency_count;
                this->workers.reserve(this->concurrency_count);

                for (int i = 0; i < this->concurrency_count; ++i)
                {
                    this->workers[i] = std::thread([this, i] {
                        while (this->is_running)
                        {
                            auto task = this->job_queue.pop();
                            if (task.has_value())
                            {
                                task.value()();
                            }
                        } 
                    });
                }
                std::cout << "STATUS : Threaded lockless job system is started." << std::endl;
            }
            ~ThreadedLocklessJobSystem()
            {
                std::cout << " STATUS : Threaded lockless job system is shutting down..." << std::endl;
                for (auto &worker : this->workers)
                {
                    worker.join();
                }
            }
            void submit(std::pair<std::function<void()>, int> task) override
            {
                this->job_queue.push(task.first);
            }

        private:
            int concurrency_count;
            std::vector<std::thread> workers;
            shs::Util::LocklessQueue<std::function<void()>> job_queue;
        };

        class ThreadedLocklessPriorityJobSystem : public shs::Job::AbstractJobSystem
        {
        public:
            ThreadedLocklessPriorityJobSystem(int concurrency_count)
            {
                this->concurrency_count = concurrency_count;
                this->workers.reserve(this->concurrency_count);
                for (int i = 0; i < this->concurrency_count; ++i)
                {
                    this->workers[i] = std::thread([this, i] {
                        while (this->is_running)
                        {
                            auto task_priority = this->job_queue.pop();
                            if (task_priority.has_value())
                            {
                                auto [task, priority] = task_priority.value();
                                task();
                            }
                        } 
                    });
                }

                std::cout << "STATUS : Lockless priority job system is started." << std::endl;

                for (auto &worker : this->workers)
                {
                    worker.join();
                }
            }
            ~ThreadedLocklessPriorityJobSystem()
            {
                std::cout << "STATUS : Lockless priority job system is shutting down..." << std::endl;
                for (auto &worker : this->workers)
                {
                    worker.join();
                }
            }
            void submit(std::pair<std::function<void()>, int> task) override
            {
                this->job_queue.push(task);
            }

        private:
            int concurrency_count;
            std::vector<std::thread> workers;
            shs::Util::LocklessPriorityQueue<std::pair<std::function<void()>, int>> job_queue;
        };
    
    }

}