/*
    3D Software Renderer - Pipeline Basic Implementation, Flat Shading
*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>

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

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

// Shader-үүд рүү дамжих өгөгдөл
struct Uniforms {
    glm::mat4 mvp;             // Model-View-Projection Matrix (Clip space руу хувиргана)
    glm::mat4 mv;              // Model-View Matrix (View space руу хувиргана - Нормальд хэрэгтэй)
    glm::vec3 light_dir_view;  // View space дээрх гэрлийн чиглэл
    shs::Color color;          // Объектын үндсэн өнгө
};

/**
 * VERTEX SHADER
 * Оройн цэгүүдийг дэлгэцийн координат руу (Clip Space) хувиргах.
 * Мөн Нормаль векторыг View Space руу хувиргаж Fragment shader рүү дамжуулна.
 */
shs::Varyings flat_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;

    // Position: Clip Space руу хувиргана (Projection * View * Model * Pos)
    out.position = u.mvp * glm::vec4(aPos, 1.0f);

    // Normal: View Space руу хувиргана (Гэрлийн тооцоог View space дээр хийх нь хялбар)
    // Нормалийг зөв хувиргахын тулд (mv) матрицын зөвхөн эргэлтийг авч байна.
    out.normal = glm::mat3(u.mv) * aNormal; 

    // Одоогоор бидэнд World Pos болон UV хэрэггүй, гэхдээ pipeline шаарддаг тул хоосон орхиж болно
    out.world_pos = glm::vec3(0.0f); 
    out.uv = glm::vec2(0.0f);

    return out;
}

/**
 * FRAGMENT SHADER
 * Пиксел бүрийн өнгийг тооцоолох.
 * Энд зөвхөн Ambient болон Diffuse гэрлийг тооцно (Specular байхгүй).
 */
shs::Color flat_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    // Дөхөлт хийгдсэн нормаль векторыг дахин normalize хийх шаардлагатай
    glm::vec3 n = glm::normalize(in.normal);
    
    // Гэрлийн чиглэл (View Space дээр ирсэн бэлэн вектор)
    glm::vec3 l = glm::normalize(u.light_dir_view);

    // 1. Diffuse (Сарнисан гэрэл): Гэрэл болон гадаргуугийн өнцгөөс хамаарна
    // dot(n, l) нь хоёр векторын хоорондох өнцгийн косинус. 
    // Хэрэв утга нь хасах байвал гэрэл ард байна гэсэн үг тул 0-ээр хязгаарлана.
    float diffuse = glm::max(glm::dot(n, l), 0.0f);

    // 2. Ambient (Орчны гэрэл): Сүүдэр хэт харанхуй байхаас сэргийлнэ
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

class Viewer
{
public:
    Viewer(glm::vec3 position, float speed)
    {
        this->position = position;
        this->speed = speed;
        this->camera = new shs::Camera3D();
        this->camera->position = this->position;
        this->camera->width = float(CANVAS_WIDTH);
        this->camera->height = float(CANVAS_HEIGHT);
        this->camera->field_of_view = 60.0f;
        this->camera->z_near = 0.1f;
        this->camera->z_far = 1000.0f;
    }
    ~Viewer() { delete camera; }

    void update()
    {
        this->camera->position = this->position;
        this->camera->update(); // View болон Projection матрицыг шинэчилнэ
    }

    glm::vec3 get_direction_vector() { return this->camera->direction_vector; }
    glm::vec3 get_right_vector() { return this->camera->right_vector; }

    shs::Camera3D *camera;
    glm::vec3 position;
    float speed;
};

class ModelGeometry
{
public:
    ModelGeometry(std::string model_path)
    {
        // aiProcess_Triangulate: Бүх дүрсийг гурвалжин болгоно
        // aiProcess_GenNormals: Нормаль байхгүй бол үүсгэнэ
        // Анхаар: aiProcess_JoinIdenticalVertices ашиглаагүй тул оройнууд давхардсан хэвээр үлдэнэ. 
        // Энэ нь Flat shading харагдуулахад тус болдог.
        unsigned int flags = aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipUVs;
        
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
        this->position = position;
        this->scale = scale;
        this->color = color;
        this->geometry = new ModelGeometry("./obj/monkey/monkey.rawobj");
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
    glm::vec3 scale;
    glm::vec3 position;
    shs::Color color;
    float rotation_angle;
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
    Viewer *viewer;
    glm::vec3 light_direction;
};

// ==========================================
// 3. RENDERER SYSTEM (PIPELINE VERSION)
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
        // Z-Buffer цэвэрлэх
        this->z_buffer->clear();

        // View болон Projection матрицуудыг авах
        glm::mat4 view = this->scene->viewer->camera->view_matrix;
        glm::mat4 proj = this->scene->viewer->camera->projection_matrix;

        // Гэрлийн чиглэлийг View Space руу хөрвүүлэх
        // Pipeline дээр бид бүх тооцооллыг аль болох камерын систем (View Space) дээр хийвэл хялбар байдаг.
        glm::vec3 light_dir_view = glm::vec3(view * glm::vec4(this->scene->light_direction, 0.0f));
        light_dir_view = glm::normalize(light_dir_view);

        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
            if (monkey)
            {
                // Uniforms бэлтгэх (Shader-лүү илгээх өгөгдөл)
                Uniforms uniforms;
                glm::mat4 model = monkey->get_world_matrix();
                
                uniforms.mv  = view * model;            // Model-View Matrix
                uniforms.mvp = proj * uniforms.mv;      // Model-View-Projection Matrix
                uniforms.light_dir_view = light_dir_view;
                uniforms.color = monkey->color;

                // Vertex болон Normal-уудын reference-ийг авна
                const auto& verts = monkey->geometry->triangles;
                const auto& norms = monkey->geometry->normals;

                // Гурвалжин бүрээр давтаж Pipeline руу илгээнэ
                for (size_t i = 0; i < verts.size(); i += 3)
                {
                    // Гурвалжны 3 орой, 3 нормаль
                    std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                    std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };

                    // PIPELINE ДУУДАХ
                    // Lambda function ашиглан гаднаас бичсэн shader-лүүгээ холбож өгнө.
                    shs::Canvas::draw_triangle_pipeline(
                        *this->scene->canvas,
                        *this->z_buffer,
                        tri_verts,
                        tri_norms,
                        // Vertex Shader Wrapper
                        [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                            return flat_vertex_shader(p, n, uniforms);
                        },
                        // Fragment Shader Wrapper
                        [&uniforms](const shs::Varyings& v) {
                            return flat_fragment_shader(v, uniforms);
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

    // Камерын байршлыг бага зэрэг хойшлуулж тохируулав
    Viewer *viewer = new Viewer(glm::vec3(0.0f, 5.0f, -20.0f), 100.0f);
    HelloScene *hello_scene = new HelloScene(main_canvas, viewer);
    SystemProcessor *sys = new SystemProcessor(hello_scene);

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
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color{30, 30, 30, 255});
        
        // Render Scene using Pipeline
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
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}