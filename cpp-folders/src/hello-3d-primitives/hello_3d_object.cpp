#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <queue>
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480


class Viewer
{
public:
    Viewer(glm::vec3 position, float speed)
    {
        this->position                 = position;
        this->speed                    = speed;
        this->camera                   = new shs::Camera3D();
        this->camera->position         = this->position;
        this->camera->width            = float(CANVAS_WIDTH);
        this->camera->height           = float(CANVAS_HEIGHT);
        this->camera->horizontal_angle = 0.0;
        this->camera->vertical_angle   = 0.0;
        this->camera->z_near           = 1.0;
        this->camera->z_far            = 1000.0f;
    }
    ~Viewer() {}

    void update()
    {
        this->camera->position = this->position;
        this->camera->update();
    }

    glm::vec3 get_direction_vector()
    {
        return this->camera->direction_vector;
    }
    glm::vec3 get_right_vector()
    {
        return this->camera->right_vector;
    }

    shs::Camera3D *camera;
    glm::vec3 position;
    glm::vec3 direction;
    float speed;

private:
};

class ModelTriangles3D
{
public:
    ModelTriangles3D(std::string model_path)
    {
        Assimp::Importer importer;
        const aiScene *scene = importer.ReadFile(model_path.c_str(), aiProcess_Triangulate);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cerr << "Error loading OBJ file: " << importer.GetErrorString() << std::endl;
        }
        for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
        {
            aiMesh *mesh = scene->mMeshes[i];

            for (unsigned int j = 0; j < mesh->mNumFaces; ++j)
            {
                aiFace face = mesh->mFaces[j];

                // Check if the face is a triangle
                if (face.mNumIndices == 3)
                {
                    aiVector3D vertex1 = mesh->mVertices[face.mIndices[0]];
                    aiVector3D vertex2 = mesh->mVertices[face.mIndices[1]];
                    aiVector3D vertex3 = mesh->mVertices[face.mIndices[2]];

                    this->triangles.push_back(glm::vec3(vertex1.x, vertex1.y, vertex1.z));
                    this->triangles.push_back(glm::vec3(vertex2.x, vertex2.y, vertex2.z));
                    this->triangles.push_back(glm::vec3(vertex3.x, vertex3.y, vertex3.z));
                }
            }
        }
        std::cout << model_path.c_str() << " is loaded." << std::endl;
    }
    ~ModelTriangles3D()
    {
    }
    std::vector<glm::vec3> triangles;

private:
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 position, glm::vec3 scale)
    {
        this->position     = position;
        this->scale        = scale;
        this->geometry     = new ModelTriangles3D("./obj/monkey/monkey.rawobj");
        this->model_matrix = glm::mat4(1.0);
    }
    ~MonkeyObject()
    {
        delete this->geometry;
    }
    glm::mat4 get_world_matrix() override
    {
        glm::mat4 updated_matrix = this->model_matrix;
        updated_matrix = glm::scale(updated_matrix, this->scale);
        updated_matrix = glm::translate(updated_matrix, this->position);
        return updated_matrix;
    }
    void update(float delta_time) override 
    {
    }
    void render() override
    {
    }

    ModelTriangles3D *geometry;
    glm::mat4         model_matrix;
    glm::vec3         scale;
    glm::vec3         position;
};


class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::Canvas *canvas, Viewer *viewer) 
    {
        this->canvas = canvas;
        this->viewer = viewer;

        MonkeyObject *monkey_object = new MonkeyObject(glm::vec3(500.2, 20.2, 15.0), glm::vec3(1.0, 1.0, 1.0));
        this->scene_objects.push_back(monkey_object);

    }
    ~HelloScene()
    {
        for (auto *obj : this->scene_objects)
        {
            delete obj;
        }
        delete this->canvas;
        delete this->viewer;
    }

    void process() override
    {
    }

    std::vector<shs::AbstractObject3D *> scene_objects;
    shs::Canvas  *canvas;
    Viewer       *viewer;

};


class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {
        //std::cout << "render systen " << delta_time << std::endl;
        glm::mat4 view_matrix       = this->scene->viewer->camera->view_matrix;
        glm::mat4 projection_matrix = this->scene->viewer->camera->projection_matrix;

        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            if (typeid(*object) == typeid(MonkeyObject))
            {
                MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                if (monkey)
                {
                    glm::mat4 world_matrix = monkey->get_world_matrix();
                    for (size_t i = 0; i < monkey->geometry->triangles.size(); i += 3)
                    {
                        glm::vec4 vertex1 = glm::vec4(monkey->geometry->triangles[i    ], 1.0);
                        glm::vec4 vertex2 = glm::vec4(monkey->geometry->triangles[i + 1], 1.0);
                        glm::vec4 vertex3 = glm::vec4(monkey->geometry->triangles[i + 2], 1.0);

                        glm::vec3 transformed_vertex1 = glm::vec3(world_matrix * view_matrix * projection_matrix * vertex1);
                        glm::vec3 transformed_vertex2 = glm::vec3(world_matrix * view_matrix * projection_matrix * vertex2);
                        glm::vec3 transformed_vertex3 = glm::vec3(world_matrix * view_matrix * projection_matrix * vertex3);

                        std::vector<glm::vec2> vertices_2d(3);
                        vertices_2d[0] = glm::vec2(transformed_vertex1.x, transformed_vertex1.y);
                        vertices_2d[1] = glm::vec2(transformed_vertex2.x, transformed_vertex2.y);
                        vertices_2d[2] = glm::vec2(transformed_vertex3.x, transformed_vertex3.y);

                        shs::Canvas::draw_triangle(*this->scene->canvas, vertices_2d, shs::Pixel::random_pixel());
                    }
                }
            }
        }
    }
private:
    HelloScene *scene;
};

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {
        this->scene->viewer->update();
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



int main()
{

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);


    Viewer *viewer = new Viewer(glm::vec3(0.0, 0.0, -23.0), 150.0f);

    HelloScene      *hello_scene      = new HelloScene(main_canvas, viewer);
    SystemProcessor *system_processor = new SystemProcessor(hello_scene);


    bool exit = false;
    SDL_Event event_data;

    int   frame_delay            = 1000 / FRAMES_PER_SECOND; // Delay for 60 FPS
    float frame_time_accumulator = 0;
    int   frame_counter          = 0;
    int   fps                    = 0;
    Uint32 delta_frame_time      = 0;

    while (!exit)
    {

        Uint32 frame_start_ticks = SDL_GetTicks();

        float delta_time_float = delta_frame_time/1000.0f;

        // catching up input events happened on hardware and feeding commands 
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

        system_processor->process(delta_time_float);

        // preparing to render on SDL2
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // software rendering or drawing stuffs goes around here
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Pixel::black_pixel());

        system_processor->render(delta_time_float);

        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);


        // actually prensenting canvas data on hardware surface
        shs::Canvas::flip_vertically(*main_canvas); // origin at the left bottom corner of the canvas
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);


    
        frame_counter++;
        delta_frame_time  = SDL_GetTicks() - frame_start_ticks;

        frame_time_accumulator  += delta_frame_time/1000.0;
        if (delta_frame_time < frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0) {
            std::string window_title = "FPS : "+std::to_string(frame_counter);
            frame_time_accumulator   = 0.0;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }


    delete system_processor;
    delete hello_scene;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}