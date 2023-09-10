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
        this->position = position;
        this->speed    = speed;
        this->camera   = new shs::Camera3D();
        this->camera->position = this->position;
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

                    this->triangles.push_back(vertex1);
                    this->triangles.push_back(vertex2);
                    this->triangles.push_back(vertex3);
                }
            }
        }
        std::cout << model_path.c_str() << " is loaded." << std::endl;
    }
    ~ModelTriangles3D()
    {
    }
    std::vector<aiVector3D> triangles;

private:
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject()
    {
        this->geometry = new ModelTriangles3D("./obj/monkey/monkey.rawobj");
    }
    ~MonkeyObject()
    {
        delete this->geometry;
    }
    glm::mat4 get_model_matrix() override
    {
        return this->model_matrix;
    }
    void update(float delta_time) override 
    {

    }
    void render() override
    {

    }

    ModelTriangles3D *geometry;
    glm::mat4         model_matrix;
};


class HelloScene;

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(std::function<HelloScene *()> getHelloSceneCallback)
        : getHelloSceneCallback(getHelloSceneCallback) {}

    void process() override
    {
        HelloScene *hello_scene = getHelloSceneCallback();
        if (hello_scene)
        {
            //float delta_time = hello_scene->delta_time;
            std::cout << "RendererSystem is processing..." << std::endl;
        }
    }

private:
    std::function<HelloScene *()> getHelloSceneCallback;
};

class LogicSystem : public shs::AbstractSystem
{
public:
    LogicSystem(std::function<HelloScene *()> getHelloSceneCallback)
        : getHelloSceneCallback(getHelloSceneCallback) {}
    void process() override
    {
        HelloScene *hello_scene = getHelloSceneCallback();
        if (hello_scene)
        {
            //float delta_time = hello_scene->delta_time;
            std::cout << "Logic system is processing..." << std::endl;
        }
    }

private:
    std::function<HelloScene *()> getHelloSceneCallback;
};


class HelloScene : public shs::AbstractScene
{
public:
    HelloScene(shs::Canvas *canvas) 
    {
        this->canvas = canvas;

        MonkeyObject *monkey_object = new MonkeyObject();
        this->scene_objects.push_back(monkey_object);


        this->systems.push_back(new LogicSystem   ([this]() { return this; }));
        this->systems.push_back(new RendererSystem([this]() { return this; }));

    }
    ~HelloScene()
    {
    }

    void process(float delta_time) override
    {
        this->delta_time = delta_time;
        for (auto &system : this->systems)
        {
            system->process();
        }
    }

    std::vector<shs::AbstractObject3D *> scene_objects;
    std::vector<shs::AbstractSystem   *> systems;
    shs::Canvas  *canvas;
    float         delta_time;

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


    Viewer *viewer = new Viewer(glm::vec3(0.0, 0.0, -3.0), 25.0f);
    shs::CommandProcessor *command_processor = new shs::CommandProcessor();

    HelloScene *hello_scene = new HelloScene(main_canvas);


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
                        command_processor->add_command(new shs::MoveForwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_s:
                        command_processor->add_command(new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_a:
                        command_processor->add_command(new shs::MoveLeftCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_d:
                        command_processor->add_command(new shs::MoveRightCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time_float));
                        break;
                }
                break;
            }
        }

        viewer->update();
        command_processor->process();


        // preparing to render on SDL2
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // software rendering or drawing stuffs goes around here
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Pixel::blue_pixel());

        hello_scene->process(delta_time_float);

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


    delete hello_scene;
    delete main_canvas;
    delete viewer;
    delete command_processor;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}