/*
    3D Software Renderer - Threaded Pipeline Implementation, Flat Shading
*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

// External Libraries
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

#define WINDOW_WIDTH      1240
#define WINDOW_HEIGHT     980
#define CANVAS_WIDTH      1240
#define CANVAS_HEIGHT     980
#define THREAD_COUNT      20
#define TILE_SIZE_X       80
#define TILE_SIZE_Y       80

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

// Shader-лүү дамжих өгөгдөл
struct Uniforms {
    glm::mat4  mvp;             // Model-View-Projection Matrix (Clip space руу хувиргана)
    glm::mat4  mv;              // Model-View Matrix (View space руу хувиргана - Нормальд хэрэгтэй)
    glm::vec3  light_dir_view;  // View space дээрх гэрлийн чиглэл
    shs::Color color;           // Объектын үндсэн өнгө
};

/*
    VERTEX SHADER
    Оройн цэгүүдийг дэлгэцийн координат руу (Clip Space) хувиргах.
    Мөн Нормаль векторыг View Space руу хувиргаж Fragment shader рүү дамжуулна.
*/
shs::Varyings flat_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;

    // Position: Clip Space руу хувиргана (Projection * View * Model * Pos)
    out.position = u.mvp * glm::vec4(aPos, 1.0f);

    // Normal: View Space руу хувиргана (Гэрлийн тооцоог View space дээр хийх нь хялбар)
    // Нормалийг зөв хувиргахын тулд (mv) матрицын зөвхөн эргэлтийг авч байна.
    out.normal = glm::mat3(u.mv) * aNormal; 

    // Одоогоор World Pos болон UV хэрэггүй, гэхдээ pipeline шаарддаг тул хоосон орхиж болно
    out.world_pos = glm::vec3(0.0f); 
    out.uv        = glm::vec2(0.0f);

    return out;
}

/*
    FRAGMENT SHADER
    Пиксел бүрийн өнгийг тооцоолох.
    Энд зөвхөн Ambient болон Diffuse гэрлийг тооцно (Specular байхгүй).
*/
shs::Color flat_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    // Дөхөлт хийгдсэн нормаль векторыг дахин normalize хийх шаардлагатай
    glm::vec3 n = glm::normalize(in.normal);
    
    // Гэрлийн чиглэл (View Space дээр ирсэн бэлэн вектор)
    glm::vec3 l = glm::normalize(u.light_dir_view);

    // Diffuse (Сарнисан гэрэл), Гэрэл болон гадаргуугийн өнцгөөс хамаарна
    // dot(n, l) нь хоёр векторын хоорондох өнцгийн косинус. 
    // Хэрэв утга нь хасах байвал гэрэл ард байна гэсэн үг тул 0-ээр хязгаарлана.
    float diffuse = glm::max(glm::dot(n, l), 0.0f);

    // Ambient (Орчны гэрэл), Сүүдэр хэт харанхуй байхаас сэргийлнэ
    float ambient = 0.2f;

    // Нийт гэрлийн хүч
    float intensity = ambient + diffuse;
    
    // Хэрэв 1.0-ээс их байвал цайралт үүсэх тул хязгаарлана
    if (intensity > 1.0f) intensity = 1.0f;

    // Эцсийн өнгө = Объектын өнгө * Гэрлийн хүч
    return shs::Color{
        (uint8_t)(u.color.r * intensity),
        (uint8_t)(u.color.g * intensity),
        (uint8_t)(u.color.b * intensity),
        255
    };
}

// ==========================================
// SCENE & OBJECT CLASSES
// ==========================================

// Using standardized shs::Viewer
using Viewer = shs::Viewer;

// Using standardized shs::ModelGeometry
using ModelGeometry = shs::ModelGeometry;

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 position, glm::vec3 scale, shs::Color color)
    {
        this->position       = position;
        this->scale          = scale;
        this->color          = color;
        this->geometry       = new ModelGeometry("./assets/obj/monkey/monkey.rawobj");
        this->rotation_angle = 0.0f;
    }
    ~MonkeyObject() { delete this->geometry; }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float delta_time) override
    {
        this->rotation_angle += 45.0f * delta_time; // Эргүүлэх
    }
    void render() override {}

    ModelGeometry *geometry;
    glm::vec3      scale;
    glm::vec3      position;
    shs::Color     color;
    float          rotation_angle;
};

class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::Canvas *canvas, Viewer *viewer) 
    {
        this->canvas = canvas;
        this->viewer = viewer;

        // Гэрлийн чиглэл (World Space) - Баруун, Дээд, Наанаас тусна
        this->light_direction = glm::normalize(glm::vec3(1.0f, 1.0f, -1.0f));

        // Сармагчин үүсгэх (Цэнхэрдүү өнгөтэй)
        this->scene_objects.push_back(new MonkeyObject(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(4.0f), shs::Color{100, 150, 255, 255}));
    }
    ~HelloScene() {
        for (auto *obj : this->scene_objects) delete obj;
    }

    void process() override {}

    std::vector<shs::AbstractObject3D *> scene_objects;
    shs::Canvas *canvas;
    Viewer      *viewer;
    glm::vec3    light_direction;
};

// ==========================================
// RENDERER SYSTEM (THREADED PIPELINE)
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys) 
        : scene(scene), job_system(job_sys)
    {
        this->z_buffer = new shs::ZBuffer(
            this->scene->canvas->get_width(),
            this->scene->canvas->get_height(),
            this->scene->viewer->camera->z_near,
            this->scene->viewer->camera->z_far
        );
    }
    ~RendererSystem() { delete this->z_buffer; }

    /*
        Thread-safe Pipeline Helper
        Дэлгэцийн зөвхөн (min_x, min_y) -> (max_x, max_y) хэсэг буюу Tile дотор зурна.
        Энэ нь өөр өөр thread-үүд санах ойн нэг байршил руу зэрэг хандахаас сэргийлнэ.
    */
    static void draw_triangle_tile(
        shs::Canvas &canvas, 
        shs::ZBuffer &z_buffer,
        const std::vector<glm::vec3> &vertices,    
        const std::vector<glm::vec3> &normals,     
        std::function<shs::Varyings(const glm::vec3&, const glm::vec3&)> vertex_shader,
        std::function<shs::Color(const shs::Varyings&)> fragment_shader,
        glm::ivec2 tile_min, glm::ivec2 tile_max)
    {
        // [VERTEX STAGE]
        shs::Varyings vout[3];
        glm::vec3 screen_coords[3];

        for (int i = 0; i < 3; i++) {
            vout[i] = vertex_shader(vertices[i], normals[i]);
            screen_coords[i] = shs::Canvas::clip_to_screen(vout[i].position, canvas.get_width(), canvas.get_height());
        }

        // [RASTER PREP] 
        // Bounding box-ийг TILE-ийн хэмжээгээр хязгаарлана (clamp)
        glm::vec2 bboxmin(tile_max.x, tile_max.y);
        glm::vec2 bboxmax(tile_min.x, tile_min.y);
        std::vector<glm::vec2> v2d = { glm::vec2(screen_coords[0]), glm::vec2(screen_coords[1]), glm::vec2(screen_coords[2]) };

        for (int i = 0; i < 3; i++) {
            bboxmin = glm::max(glm::vec2(tile_min), glm::min(bboxmin, v2d[i]));
            bboxmax = glm::min(glm::vec2(tile_max), glm::max(bboxmax, v2d[i]));
        }

        // Хэрэв гурвалжин тухайн Tile-аас гадна байвал зурахгүй
        if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

        float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) - (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
        if (area <= 0) return;

        // [FRAGMENT STAGE]
        for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
            for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {
                
                glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                // Z-Buffer test (Race condition үүсэхгүй, учир нь Thread бүр өөрийн бүсэд ажиллана)
                float z = bc.x * screen_coords[0].z + bc.y * screen_coords[1].z + bc.z * screen_coords[2].z;
                if (z_buffer.test_and_set_depth(px, py, z)) {
                    
                    shs::Varyings interpolated;
                    interpolated.normal = glm::normalize(bc.x * vout[0].normal + bc.y * vout[1].normal + bc.z * vout[2].normal);
                    
                    canvas.draw_pixel_screen_space(px, py, fragment_shader(interpolated));
                }
            }
        }
    }

    void process(float delta_time) override
    {
        this->z_buffer->clear();

        // Матриц болон гэрлийн тооцооллыг үндсэн thread дээр нэг удаа хийнэ
        glm::mat4 view = this->scene->viewer->camera->view_matrix;
        glm::mat4 proj = this->scene->viewer->camera->projection_matrix;
        glm::vec3 light_dir_view = glm::normalize(glm::vec3(view * glm::vec4(this->scene->light_direction, 0.0f)));

        int w = this->scene->canvas->get_width();
        int h = this->scene->canvas->get_height();
        
        // Дэлгэцийг Tile-уудад хуваах
        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wait_group.reset();

        for(int ty = 0; ty < rows; ty++) {
            for(int tx = 0; tx < cols; tx++) {
                
                wait_group.add(1);

                // Job submit хийх
                job_system->submit({[this, tx, ty, w, h, view, proj, light_dir_view]() {
                    
                    // Tile-ийн хязгааруудыг тооцох
                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, w) - 1, std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                    // Объектуудыг давтах
                    for (shs::AbstractObject3D *object : this->scene->scene_objects)
                    {
                        MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                        if (!monkey) continue;

                        Uniforms uniforms;
                        uniforms.mv  = view * monkey->get_world_matrix();
                        uniforms.mvp = proj * uniforms.mv;
                        uniforms.light_dir_view = light_dir_view;
                        uniforms.color = monkey->color;

                        const auto& verts = monkey->geometry->triangles;
                        const auto& norms = monkey->geometry->normals;

                        for (size_t i = 0; i < verts.size(); i += 3)
                        {
                            std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                            std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                            // Tile дээр зурах функц дуудах
                            draw_triangle_tile(
                                *this->scene->canvas,
                                *this->z_buffer,
                                tri_verts,
                                tri_norms,
                                [&uniforms](const glm::vec3& p, const glm::vec3& n) { return flat_vertex_shader(p, n, uniforms); },
                                [&uniforms](const shs::Varyings& v) { return flat_fragment_shader(v, uniforms); },
                                t_min, t_max
                            );
                        }
                    }
                    wait_group.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }
        
        // Бүх Tile зурагдаж дуустал хүлээнэ
        wait_group.wait();
    }

private:
    HelloScene                          *scene;
    shs::ZBuffer                        *z_buffer;
    shs::Job::ThreadedPriorityJobSystem *job_system;
    shs::Job::WaitGroup                  wait_group;
};

// ==========================================
// LOGIC & MAIN LOOP
// ==========================================

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {
        this->scene->viewer->update();
        for (auto *obj : this->scene->scene_objects)
            obj->update(delta_time);
    }
private:
    HelloScene *scene;
};

class SystemProcessor
{
public:
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys) 
    {
        this->command_processor = new shs::CommandProcessor();
        this->renderer_system   = new RendererSystem(scene, job_sys);
        this->logic_system      = new LogicSystem(scene);
    }
    ~SystemProcessor()
    {
        delete this->command_processor;
        delete this->renderer_system;
        delete this->logic_system;
    }
    void process(float delta_time) 
    {
        this->command_processor->process();
        this->logic_system->process(delta_time);
    }
    void render(float delta_time)
    {
        this->renderer_system->process(delta_time);
    }

    shs::CommandProcessor *command_processor;
    LogicSystem           *logic_system;
    RendererSystem        *renderer_system;  
};

int main(int argc, char* argv[])
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    // Job System үүсгэх
    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    
    shs::Canvas *main_canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    // Камерын байршлыг бага зэрэг хойшлуулж тохируулав
    Viewer *viewer = new Viewer(glm::vec3(0.0f, 5.0f, -20.0f), 100.0f, CANVAS_WIDTH, CANVAS_HEIGHT);
    HelloScene *hello_scene = new HelloScene(main_canvas, viewer);
    SystemProcessor *sys = new SystemProcessor(hello_scene, job_system);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();

    while (!exit)
    {
        Uint32 current_tick = SDL_GetTicks();
        float delta_time = (current_tick - last_tick) / 1000.0f;
        last_tick = current_tick;

        while (SDL_PollEvent(&event_data))
        {
            if (event_data.type == SDL_QUIT) exit = true;
            if (event_data.type == SDL_KEYDOWN) {
                if(event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;
                
                // Удирдлага
                if(event_data.key.keysym.sym == SDLK_w)
                    sys->command_processor->add_command(new shs::MoveForwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_s)
                    sys->command_processor->add_command(new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_a)
                    sys->command_processor->add_command(new shs::MoveLeftCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_d)
                    sys->command_processor->add_command(new shs::MoveRightCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time));
            }
        }

        // Logic
        sys->process(delta_time);

        // Clear Screen (Хар саарал дэвсгэр)
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color::black());
        
        // Render Scene using Job System (Threaded)
        sys->render(delta_time);

        // Update SDL
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    delete sys;
    delete hello_scene;
    delete viewer;
    delete main_canvas;
    delete job_system;
    
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}