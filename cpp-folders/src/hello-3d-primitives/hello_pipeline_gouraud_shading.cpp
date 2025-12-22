/*
    3D Software Renderer - Gouraud Shading Pipeline
    Lighting calculated per-vertex (Vertex Shader)
    + FPS Camera Controls (Drag to Look)
*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

// External Libraries
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "shs_renderer.hpp"

// Тохиргооны тогтмолууд
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480
#define MOUSE_SENSITIVITY 0.2f

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

struct Uniforms {
    glm::mat4 mvp;          
    glm::mat4 model;        
    glm::vec3 light_dir;    
    glm::vec3 camera_pos;   
    shs::Color color;       
};

// VERTEX SHADER (Gouraud)
// Гэрлийн бүх тооцооллыг энд хийнэ
shs::Varyings gouraud_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;

    // Position: Clip Space руу (Зурахын тулд зайлшгүй шаардлагатай)
    out.position = u.mvp * glm::vec4(aPos, 1.0f);

    // Тооцоололд хэрэгтэй векторууд (World Space)
    glm::vec3 worldPos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    glm::vec3 normal   = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - worldPos);

    // AMBIENT
    float ambientStrength = 0.15f;
    glm::vec3 ambient = ambientStrength * glm::vec3(1.0f, 1.0f, 1.0f);

    // DIFFUSE
    float diff = glm::max(glm::dot(normal, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f, 1.0f, 1.0f);

    // SPECULAR (Blinn-Phong style but per vertex)
    // Gouraud дээр Specular хийхэд highlight нь оройн цэгтэй давхцахгүй бол алга болох магадлалтай
    glm::vec3 halfwayDir = glm::normalize(lightDir + viewDir);
    float specularStrength = 0.5f;
    float shininess = 32.0f; 

    float spec = glm::pow(glm::max(glm::dot(normal, halfwayDir), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f, 1.0f, 1.0f);

    // НИЙЛБЭР ӨНГӨ
    glm::vec3 objectColorVec = glm::vec3(u.color.r, u.color.g, u.color.b) / 255.0f;
    glm::vec3 finalColor = (ambient + diffuse + specular) * objectColorVec;
    finalColor = glm::clamp(finalColor, 0.0f, 1.0f);

    // HACK: Varyings дотор 'color' байхгүй тул 'world_pos' талбарыг ашиглан өнгөө дамжуулна.
    // Rasterizer нь үүнийг автоматаар interpolate хийнэ.
    out.world_pos = finalColor; 

    return out;
}

// FRAGMENT SHADER (Gouraud)
// Энд зүгээр л Vertex Shader-аас ирсэн өнгийг гаргана
shs::Color gouraud_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    // Vertex Shader-аас interpolate хийгдэж ирсэн өнгө (world_pos дотор хадгалсан)
    glm::vec3 color = in.world_pos;

    // 0-1 range-ийг 0-255 руу хөрвүүлнэ
    return shs::Color{
        (uint8_t)(color.r * 255),
        (uint8_t)(color.g * 255),
        (uint8_t)(color.b * 255),
        255
    };
}

// ==========================================
// SCENE & OBJECT CLASSES
// ==========================================

class Viewer
{
public:
    Viewer(glm::vec3 position, float speed)
    {
        this->position         = position;
        this->speed            = speed;
        this->camera           = new shs::Camera3D();
        this->camera->position = this->position;
        this->camera->width    = float(CANVAS_WIDTH);
        this->camera->height   = float(CANVAS_HEIGHT);
        this->camera->field_of_view = 60.0f;
        this->camera->z_near   = 0.1f;
        this->camera->z_far    = 1000.0f;
        
        this->horizontal_angle = 0.0f;
        this->vertical_angle   = 0.0f;
    }
    ~Viewer() { delete camera; }

    void update()
    {
        this->camera->position         = this->position;
        this->camera->horizontal_angle = this->horizontal_angle;
        this->camera->vertical_angle   = this->vertical_angle;
        this->camera->update(); 
    }

    glm::vec3 get_direction_vector() { return this->camera->direction_vector; }
    glm::vec3 get_right_vector() { return this->camera->right_vector; }

    shs::Camera3D *camera;
    glm::vec3 position;
    float horizontal_angle;
    float vertical_angle;
    float speed;
};

class ModelGeometry
{
public:
    ModelGeometry(std::string model_path)
    {
        unsigned int flags = aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices;
        const aiScene *scene = this->importer.ReadFile(model_path.c_str(), flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Model load error: " << this->importer.GetErrorString() << std::endl;
            return;
        }

        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                if (mesh->mFaces[j].mNumIndices == 3) {
                    for(int k=0; k<3; k++) {
                        aiVector3D v = mesh->mVertices[mesh->mFaces[j].mIndices[k]];
                        this->triangles.push_back(glm::vec3(v.x, v.y, v.z));
                        if (mesh->HasNormals()) {
                            aiVector3D n = mesh->mNormals[mesh->mFaces[j].mIndices[k]];
                            this->normals.push_back(glm::vec3(n.x, n.y, n.z));
                        } else {
                            this->normals.push_back(glm::vec3(0, 0, 1));
                        }
                    }
                }
            }
        }
    }
    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;
    Assimp::Importer importer;
};

class MonkeyObject : public shs::AbstractObject3D
{
public:
    MonkeyObject(glm::vec3 position, glm::vec3 scale, shs::Color color)
    {
        this->position       = position;
        this->scale          = scale;
        this->color          = color;
        this->geometry       = new ModelGeometry("./obj/monkey/monkey.rawobj");
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
        //this->rotation_angle += 30.0f * delta_time; 
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
        // Гэрэл баруун дээд урдаас (45 градус орчим)
        this->light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));
        this->scene_objects.push_back(new MonkeyObject(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(4.0f), shs::Color{60, 100, 200, 255}));
    }
    ~HelloScene() {
        for (auto *obj : this->scene_objects) delete obj;
    }
    void process() override {}

    std::vector<shs::AbstractObject3D *> scene_objects;
    shs::Canvas *canvas;
    Viewer *viewer;
    glm::vec3 light_direction;
};

// ==========================================
// RENDERER SYSTEM
// ==========================================

class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene) : scene(scene) 
    {
        this->z_buffer = new shs::ZBuffer(
            this->scene->canvas->get_width(),
            this->scene->canvas->get_height(),
            this->scene->viewer->camera->z_near,
            this->scene->viewer->camera->z_far
        );
    }
    ~RendererSystem() { delete this->z_buffer; }

    void process(float delta_time) override
    {
        this->z_buffer->clear();

        glm::mat4 view = this->scene->viewer->camera->view_matrix;
        glm::mat4 proj = this->scene->viewer->camera->projection_matrix;

        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
            if (monkey)
            {
                Uniforms uniforms;
                uniforms.model      = monkey->get_world_matrix();
                uniforms.mvp        = proj * view * uniforms.model;
                uniforms.light_dir  = this->scene->light_direction; 
                uniforms.camera_pos = this->scene->viewer->position; 
                uniforms.color      = monkey->color;

                const auto& verts = monkey->geometry->triangles;
                const auto& norms = monkey->geometry->normals;

                for (size_t i = 0; i < verts.size(); i += 3)
                {
                    std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                    std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                    shs::Canvas::draw_triangle_pipeline(
                        *this->scene->canvas,
                        *this->z_buffer,
                        tri_verts,
                        tri_norms,
                        // Gouraud Vertex Shader
                        [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                            return gouraud_vertex_shader(p, n, uniforms);
                        },
                        // Gouraud Fragment Shader
                        [&uniforms](const shs::Varyings& v) {
                            return gouraud_fragment_shader(v, uniforms);
                        }
                    );
                }
            }
        }
    }
private:
    HelloScene   *scene;
    shs::ZBuffer *z_buffer;
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

int main(int argc, char* argv[])
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    
    shs::Canvas *main_canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    Viewer *viewer = new Viewer(glm::vec3(0.0f, 5.0f, -20.0f), 50.0f);
    HelloScene *hello_scene = new HelloScene(main_canvas, viewer);
    SystemProcessor *sys = new SystemProcessor(hello_scene);

    bool exit = false;
    SDL_Event event_data;
    Uint32 last_tick = SDL_GetTicks();
    bool is_dragging = false;

    while (!exit)
    {
        Uint32 current_tick = SDL_GetTicks();
        float delta_time = (current_tick - last_tick) / 1000.0f;
        last_tick = current_tick;

        while (SDL_PollEvent(&event_data))
        {
            if (event_data.type == SDL_QUIT) exit = true;
            
            if (event_data.type == SDL_MOUSEBUTTONDOWN) {
                if (event_data.button.button == SDL_BUTTON_LEFT) {
                    is_dragging = true;
                }
            }
            if (event_data.type == SDL_MOUSEBUTTONUP) {
                if (event_data.button.button == SDL_BUTTON_LEFT) {
                    is_dragging = false;
                }
            }

            if (event_data.type == SDL_MOUSEMOTION)
            {
                if (is_dragging) {
                    viewer->horizontal_angle += event_data.motion.xrel * MOUSE_SENSITIVITY;
                    viewer->vertical_angle   -= event_data.motion.yrel * MOUSE_SENSITIVITY;
                    if (viewer->vertical_angle > 89.0f) viewer->vertical_angle = 89.0f;
                    if (viewer->vertical_angle < -89.0f) viewer->vertical_angle = -89.0f;
                }
            }

            if (event_data.type == SDL_KEYDOWN) {
                if(event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;
                if(event_data.key.keysym.sym == SDLK_w) sys->command_processor->add_command(new shs::MoveForwardCommand (viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_s) sys->command_processor->add_command(new shs::MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_a) sys->command_processor->add_command(new shs::MoveLeftCommand    (viewer->position, viewer->get_right_vector()    , viewer->speed, delta_time));
                if(event_data.key.keysym.sym == SDLK_d) sys->command_processor->add_command(new shs::MoveRightCommand   (viewer->position, viewer->get_right_vector()    , viewer->speed, delta_time));
            }
        }

        sys->process(delta_time);
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{20, 20, 25, 255}); 
        sys->render(delta_time);

        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }

    delete sys;
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