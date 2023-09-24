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
        this->camera->width            = float(CANVAS_WIDTH );
        this->camera->height           = float(CANVAS_HEIGHT);
        this->camera->field_of_view    = 45.0;
        this->camera->horizontal_angle = 0.0;
        this->camera->vertical_angle   = 0.0;
        this->camera->z_near           = 0.01f;
        this->camera->z_far            = 1000.0f;
    }
    ~Viewer() {}

    void update()
    {
        this->camera->position         = this->position;
        this->camera->horizontal_angle = this->horizontal_angle;
        this->camera->vertical_angle   = this->vertical_angle;
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
    float horizontal_angle;
    float vertical_angle;
    float speed;

private:
};

class ModelGeometry
{
public:
    ModelGeometry(std::string model_path)
    {
        unsigned int flags = aiProcess_Triangulate | aiProcess_GenNormals;
        const aiScene *scene = this->importer.ReadFile(model_path.c_str(), flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cerr << "Error loading OBJ file: " << this->importer.GetErrorString() << std::endl;
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

                    aiVector3D normal1 = mesh->mNormals[face.mIndices[0]];
                    aiVector3D normal2 = mesh->mNormals[face.mIndices[1]];
                    aiVector3D normal3 = mesh->mNormals[face.mIndices[2]];
                    this->normals.push_back(glm::vec3(normal1.x, normal1.y, normal1.z));
                    this->normals.push_back(glm::vec3(normal2.x, normal2.y, normal2.z));
                    this->normals.push_back(glm::vec3(normal3.x, normal3.y, normal3.z));
                }
            }
        }
        std::cout << model_path.c_str() << " is loaded." << std::endl;
    }
    ~ModelGeometry()
    {
        this->importer.FreeScene();
    }
    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;
    Assimp::Importer       importer;

private:
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 position, glm::vec3 scale)
    {
        this->position        = position;
        this->scale           = scale;
        this->geometry        = new ModelGeometry("./obj/monkey/monkey.rawobj");
        this->rotation_angle  = 0.0;
    }
    ~MonkeyObject()
    {
        delete this->geometry;
    }
    glm::mat4 get_world_matrix() override
    {
        glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0), this->position);
        glm::mat4 rotation_matrix    = glm::rotate   (glm::mat4(1.0), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 scaling_matrix     = glm::scale    (glm::mat4(1.0), scale);
        return translation_matrix * rotation_matrix * scaling_matrix;
    }
    void update(float delta_time) override
    {
        float rotation_speed = 30.0;
        this->rotation_angle -= rotation_speed * delta_time; // rotation by clock-wise 
        if (this->rotation_angle > 360.0f)
        {
            this->rotation_angle = 0.0f;
        }
    }
    void render() override
    {
    }

    ModelGeometry *geometry;
    glm::vec3      scale;
    glm::vec3      position;
    float          rotation_angle;
};


class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::Canvas *canvas, Viewer *viewer) 
    {
        this->canvas = canvas;
        this->viewer = viewer;

        float step = 15.0;
        for (int i=0; i<2; ++i)
        {
            for (int j=0; j<2; ++j)
            {
                this->scene_objects.push_back(new MonkeyObject(glm::vec3(i*step, 0.0, j*step+30.0f), glm::vec3(5.0, 5.0, 5.0)));
            }
        }
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

    glm::vec3    light_direction = glm::vec3(0.1, -0.1, 0.3);

};


class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override
    {

        glm::mat4 view_matrix       = this->scene->viewer->camera->view_matrix;
        glm::mat4 projection_matrix = this->scene->viewer->camera->projection_matrix;

        glm::vec4 light_direction_in_world_space = glm::vec4(this->scene->light_direction, 1.0);
        glm::vec4 light_direction_in_view_space_ = view_matrix * light_direction_in_world_space;
        glm::vec3 light_direction_in_view_space  = glm::vec3(light_direction_in_view_space_.x, light_direction_in_view_space_.y, light_direction_in_view_space_.z);

        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            if (typeid(*object) == typeid(MonkeyObject))
            {
                MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                if (monkey)
                {
                    glm::mat4 model_matrix = monkey->get_world_matrix();

                    for (size_t i = 0; i < monkey->geometry->triangles.size(); i += 3)
                    {

                        glm::vec4 normal1 = view_matrix * (model_matrix * glm::vec4(monkey->geometry->normals[i    ], 0.0));
                        glm::vec4 normal2 = view_matrix * (model_matrix * glm::vec4(monkey->geometry->normals[i + 1], 0.0));
                        glm::vec4 normal3 = view_matrix * (model_matrix * glm::vec4(monkey->geometry->normals[i + 2], 0.0));
                        std::vector<glm::vec3> view_space_normals(3);
                        view_space_normals[0] = glm::vec3(normal1.x, normal1.y, normal1.z);
                        view_space_normals[1] = glm::vec3(normal2.x, normal2.y, normal2.z);
                        view_space_normals[2] = glm::vec3(normal3.x, normal3.y, normal3.z);

                        glm::vec4 vertex1_clip_space = projection_matrix * (view_matrix * (model_matrix * glm::vec4(monkey->geometry->triangles[i    ], 1.0)));
                        glm::vec4 vertex2_clip_space = projection_matrix * (view_matrix * (model_matrix * glm::vec4(monkey->geometry->triangles[i + 1], 1.0)));
                        glm::vec4 vertex3_clip_space = projection_matrix * (view_matrix * (model_matrix * glm::vec4(monkey->geometry->triangles[i + 2], 1.0)));

                        std::vector<glm::vec2> vertices_2d(3);
                        vertices_2d[0] = shs::Canvas::clip_to_screen(vertex1_clip_space, CANVAS_WIDTH,  CANVAS_HEIGHT);
                        vertices_2d[1] = shs::Canvas::clip_to_screen(vertex2_clip_space, CANVAS_WIDTH,  CANVAS_HEIGHT);
                        vertices_2d[2] = shs::Canvas::clip_to_screen(vertex2_clip_space, CANVAS_WIDTH,  CANVAS_HEIGHT);


                        //shs::Canvas::draw_line(*this->scene->canvas, vertices_2d[0].x, vertices_2d[0].y, vertices_2d[1].x, vertices_2d[1].y, shs::Pixel::green_pixel());
                        //shs::Canvas::draw_line(*this->scene->canvas, vertices_2d[0].x, vertices_2d[0].y, vertices_2d[2].x, vertices_2d[2].y, shs::Pixel::blue_pixel());
                        //shs::Canvas::draw_line(*this->scene->canvas, vertices_2d[1].x, vertices_2d[1].y, vertices_2d[2].x, vertices_2d[2].y, shs::Pixel::red_pixel());

                        /*
                        std::vector<glm::vec3> colors = {
                            {1.0f, 0.0f, 0.0f},
                            {0.0f, 1.0f, 0.0f},
                            {0.0f, 0.0f, 1.0f},
                        };
                        shs::Canvas::draw_triangle_color_approximation(*this->scene->canvas, vertices_2d, colors);
                        */

                        shs::Canvas::draw_triangle_flat_shading(*this->scene->canvas, vertices_2d, view_space_normals, light_direction_in_view_space);
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


    Viewer *viewer = new Viewer(glm::vec3(0.0, 10.0, -50.0), 150.0f);

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
                    case SDLK_UP:
                        break;
                    case SDLK_DOWN:
                        break;
                    case SDLK_LEFT:
                        break;
                    case SDLK_RIGHT:
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