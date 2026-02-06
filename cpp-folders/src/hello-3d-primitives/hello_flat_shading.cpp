/*
    3D Software Renderer - Flat Shading 
*/

#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <queue>
#include <iostream>
#include <vector>

// Гадаад сангууд (External Libraries)
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

// Тохиргооны тогтмолууд
#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480

/**
 * Камерын байршил, чиглэл, харах өнцөг зэргийг удирдана.
 */
// Using standardized shs::Viewer
using Viewer = shs::Viewer;

/**
 * 3D моделийн файлыг (.obj) уншиж, оройн цэг (vertex) болон нормалиудыг (normal) хадгална.
 */
// Using standardized shs::ModelGeometry
using ModelGeometry = shs::ModelGeometry;

/**
 * Тухайн 3D объектын байршил, эргэлт, хэмжээг удирдана.
 */
class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 position, glm::vec3 scale)
    {
        this->position        = position;
        this->scale           = scale;
        this->geometry        = new ModelGeometry("./assets/obj/monkey/monkey.rawobj");
        this->rotation_angle  = 0.0f;
    }
    ~MonkeyObject()
    {
        delete this->geometry;
    }

    // Объектын World Matrix (Model Matrix) үүсгэх
    glm::mat4 get_world_matrix() override
    {
        glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 rotation_matrix    = glm::rotate   (glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 scaling_matrix     = glm::scale    (glm::mat4(1.0f), scale);
        
        return translation_matrix * rotation_matrix * scaling_matrix;
    }

    // Хүрээ болгонд эргүүлэх логик
    void update(float delta_time) override
    {
        float rotation_speed = 30.0f;
        this->rotation_angle += rotation_speed * delta_time; 
        if (this->rotation_angle >= 360.0f) this->rotation_angle -= 360.0f;
    }
    void render() override {}

    ModelGeometry *geometry;
    glm::vec3      scale;
    glm::vec3      position;
    float          rotation_angle;
};

/**
 * Сцен доторх объектууд, гэрэл болон viewer-ийг агуулна.
 */
class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::Canvas *canvas, Viewer *viewer) 
    {
        this->canvas = canvas;
        this->viewer = viewer;

        float step = 15.0f;
        // 4 ширхэг сармагчин үүсгэх (2x2 матриц)
        for (int i=0; i<2; ++i)
        {
            for (int j=0; j<2; ++j)
            {
                this->scene_objects.push_back(new MonkeyObject(glm::vec3(i*step - 7.5f, 0.0f, j*step + 20.0f), glm::vec3(5.0f, 5.0f, 5.0f)));
            }
        }
    }
    ~HelloScene()
    {
        for (auto *obj : this->scene_objects)
        {
            delete obj;
        }
    }

    void process() override {}

    std::vector<shs::AbstractObject3D *> scene_objects;
    shs::Canvas  *canvas;
    Viewer       *viewer;

    // Гэрлийн чиглэл (World Space)
    // World Space чиглэлээс гэрлийн эх үүсвэр лүү
    glm::vec3 light_direction = glm::vec3(1.0f, 0.3f, 1.0f);
};

/**
 * 3D геометрийг боловсруулж, Z-Buffer ашиглан Flat Shading хийнэ.
 */
class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene) : scene(scene) 
    {
        // Z-Buffer үүсгэх
        this->z_buffer = new shs::ZBuffer(
            this->scene->canvas->get_width(),
            this->scene->canvas->get_height(),
            this->scene->viewer->camera->z_near,
            this->scene->viewer->camera->z_far
        );
    }
    ~RendererSystem()
    {
        delete this->z_buffer;
    }

    void process(float delta_time) override
    {
        // Z-Buffer цэвэрлэх (Frame бүрт)
        this->z_buffer->clear();

        glm::mat4 view_matrix       = this->scene->viewer->camera->view_matrix;
        glm::mat4 projection_matrix = this->scene->viewer->camera->projection_matrix;

        // --- ГЭРЛИЙН ТООЦООЛОЛ (World -> View Space) ---
        // Scene доторх гэрлийн чиглэлийг авч, View Matrix-аар үржүүлнэ.
        // Ингэснээр гэрэл камертай харьцуулахад зөв байршилд очно.
        // w=0.0 байна учир нь энэ бол чиглэл (vector), байршил биш (point).
        glm::vec3 light_dir_world = glm::normalize(this->scene->light_direction);
        glm::vec3 light_dir_view  = glm::vec3(view_matrix * glm::vec4(light_dir_world, 0.0f));
        light_dir_view = glm::normalize(light_dir_view);

        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            if (typeid(*object) == typeid(MonkeyObject))
            {
                MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                if (monkey)
                {
                    glm::mat4 model_matrix = monkey->get_world_matrix();
                    
                    // Model-View Matrix (World -> View Space шилжүүлэг)
                    glm::mat4 model_view = view_matrix * model_matrix;

                    for (size_t i = 0; i < monkey->geometry->triangles.size(); i += 3)
                    {
                        // 2. Vertex Transformation (Clip Space руу)
                        glm::vec4 vertex1_clip_space = projection_matrix * (model_view * glm::vec4(monkey->geometry->triangles[i    ], 1.0f));
                        glm::vec4 vertex2_clip_space = projection_matrix * (model_view * glm::vec4(monkey->geometry->triangles[i + 1], 1.0f));
                        glm::vec4 vertex3_clip_space = projection_matrix * (model_view * glm::vec4(monkey->geometry->triangles[i + 2], 1.0f));

                        // 3. CLIPPING
                        if (vertex1_clip_space.w <= 0.0f || vertex2_clip_space.w <= 0.0f || vertex3_clip_space.w <= 0.0f)
                            continue;

                        // 4. Normal Transformation (View Space руу)
                        glm::mat3 nmat = glm::transpose(glm::inverse(glm::mat3(model_view)));

                        glm::vec3 n1 = nmat * monkey->geometry->normals[i];
                        glm::vec3 n2 = nmat * monkey->geometry->normals[i + 1];
                        glm::vec3 n3 = nmat * monkey->geometry->normals[i + 2];
                        n1 = glm::normalize(n1);
                        n2 = glm::normalize(n2);
                        n3 = glm::normalize(n3);

                        std::vector<glm::vec3> view_space_normals = { n1, n2, n3 };

                        // 5. Screen Space Conversion (Дэлгэцийн координат руу)
                        std::vector<glm::vec3> vertices_screen(3);
                        vertices_screen[0] = shs::Canvas::clip_to_screen(vertex1_clip_space, CANVAS_WIDTH,  CANVAS_HEIGHT);
                        vertices_screen[1] = shs::Canvas::clip_to_screen(vertex2_clip_space, CANVAS_WIDTH,  CANVAS_HEIGHT);
                        vertices_screen[2] = shs::Canvas::clip_to_screen(vertex3_clip_space, CANVAS_WIDTH,  CANVAS_HEIGHT);

                        // 6. Draw Call (Flat Shading)
                        // Тооцоолсон 'light_dir_view' гэрлийн чиглэлийг дамжуулна
                        shs::Canvas::draw_triangle_flat_shading(
                            *this->scene->canvas, 
                            *this->z_buffer, 
                            vertices_screen, 
                            view_space_normals, 
                            light_dir_view 
                        );
                    }
                }
            }
        }
    }
private:
    HelloScene   *scene;
    shs::ZBuffer *z_buffer;
};

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {
        this->scene->viewer->update();
        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            if (typeid(*object) == typeid(MonkeyObject))
            {
                MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                if (monkey)
                {
                    monkey->update(delta_time);
                }
            }
        }
    }
private:
    HelloScene *scene;
};

class SystemProcessor
{
public:
    SystemProcessor(HelloScene *scene) 
    {
        this->command_processor = new shs::CommandProcessor();
        this->renderer_system   = new RendererSystem(scene);
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

// ==========================================
// MAIN FUNCTION
// ==========================================
int main(int argc, char* argv[])
{
    // SDL эхлүүлэх
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    // Canvas болон Texture үүсгэх
    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    // Viewer үүсгэх (Байршил: Z=-50, Хурд: 150)
    // Left-Handed системд +Z нь урагшаа. Камер 0 руу харахын тулд -Z дээр байна.
    Viewer *viewer = new Viewer(glm::vec3(0.0f, 10.0f, -50.0f), 150.0f, CANVAS_WIDTH, CANVAS_HEIGHT);

    HelloScene      *hello_scene      = new HelloScene(main_canvas, viewer);
    SystemProcessor *system_processor = new SystemProcessor(hello_scene);

    bool exit = false;
    SDL_Event event_data;

    int   frame_delay            = 1000 / FRAMES_PER_SECOND; 
    float frame_time_accumulator = 0.0f;
    int   frame_counter          = 0;
    Uint32 delta_frame_time      = 0;

    // Үндсэн давталт (Game Loop)
    while (!exit)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();
        float delta_time_float = delta_frame_time / 1000.0f;

        // Input Handling
        while (SDL_PollEvent(&event_data))
        {
            switch (event_data.type)
            {
            case SDL_QUIT:
                exit = true;
                break;
            case SDL_KEYDOWN:
                switch(event_data.key.keysym.sym) {
                    case SDLK_ESCAPE: 
                        exit = true;
                        break;
                    case SDLK_w:
                        system_processor->command_processor->add_command(new shs::MoveForwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_s:
                        system_processor->command_processor->add_command(new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_a:
                        system_processor->command_processor->add_command(new shs::MoveLeftCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_d:
                        system_processor->command_processor->add_command(new shs::MoveRightCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time_float));
                        break;
                }
                break;
            }
        }

        // Logic Process
        system_processor->process(delta_time_float);

        // Render Preparation (Clear Screen)
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Canvas-ийг хараар будах
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color::black());

        // 3D Rendering хийх
        system_processor->render(delta_time_float);

        // Дэлгэц рүү хуулах
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        // FPS Control
        frame_counter++;
        delta_frame_time  = SDL_GetTicks() - frame_start_ticks;
        frame_time_accumulator += delta_frame_time / 1000.0f;

        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
            delta_frame_time = frame_delay;
        }

        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "Flat Shading Demo - FPS: " + std::to_string(frame_counter);
            SDL_SetWindowTitle(window, window_title.c_str());
            frame_time_accumulator   = 0.0f;
            frame_counter            = 0;
        }
    }

    // Cleanup
    delete system_processor;
    delete hello_scene;
    delete viewer;
    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}