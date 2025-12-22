/*
    3D Software Renderer Demo
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

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480

/**
 * Камерын байршил, чиглэл, харах өнцөг зэргийг удирдана.
 */
class Viewer
{
public:
    Viewer(glm::vec3 position, float speed)
    {
        this->position                 = position;
        this->speed                    = speed;
        this->camera                   = new shs::Camera3D();
        this->camera->position         = this->position;
        // Камерын viewport хэмжээ
        this->camera->width            = float(CANVAS_WIDTH);
        this->camera->height           = float(CANVAS_HEIGHT);
        // Харах өнцөг (FOV)
        this->camera->field_of_view    = 45.0f;
        this->camera->horizontal_angle = 0.0f;
        this->camera->vertical_angle   = 0.0f;
        // Clipping planes (наана болон цаана харагдах хязгаар)
        this->camera->z_near           = 0.1f;
        this->camera->z_far            = 1000.0f;
    }
    ~Viewer() { delete camera; }

    // Камерын матрицыг шинэчлэх функц
    void update()
    {
        this->camera->position         = this->position;
        this->camera->horizontal_angle = this->horizontal_angle;
        this->camera->vertical_angle   = this->vertical_angle;
        this->camera->update(); // View болон Projection матрицыг дахин бодно
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
    float horizontal_angle;
    float vertical_angle;
    float speed;
};

/**
 * 3D моделийн файлыг (жишээ нь .obj) уншиж, оройн цэгүүдийг (vertices) хадгална.
 */
class ModelGeometry
{
public:
    ModelGeometry(std::string model_path)
    {
        Assimp::Importer importer;
        // Assimp ашиглан файлыг унших
        const aiScene *scene = importer.ReadFile(model_path.c_str(), aiProcessPreset_TargetRealtime_Quality);
        
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
        {
            std::cerr << "Алдаа: Модель уншихад алдаа гарлаа: " << importer.GetErrorString() << std::endl;
            return;
        }

        // Бүх mesh-үүдээр давтаж vertex-үүдийг авна
        for (unsigned int i = 0; i < scene->mNumMeshes; ++i)
        {
            aiMesh *mesh = scene->mMeshes[i];

            for (unsigned int j = 0; j < mesh->mNumFaces; ++j)
            {
                aiFace face = mesh->mFaces[j];

                // Зөвхөн гурвалжин (triangle) дүрсүүдийг авна
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
        std::cout << model_path << " амжилттай уншигдлаа. Нийт гурвалжин: " << this->triangles.size() / 3 << std::endl;
    }
    ~ModelGeometry() {}

    std::vector<glm::vec3> triangles;
};

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
        this->geometry        = new ModelGeometry("./obj/monkey/monkey.rawobj");
        this->rotation_angle  = 0.0f;
    }
    ~MonkeyObject()
    {
        delete this->geometry;
    }

    // Объект бүр өөрийн гэсэн World Matrix-тай байна (Model Matrix)
    // Энэ нь объектыг дэлхийн координат руу шилжүүлнэ.
    glm::mat4 get_world_matrix() override
    {
        glm::mat4 translation_matrix = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 rotation_matrix    = glm::rotate   (glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 scaling_matrix     = glm::scale    (glm::mat4(1.0f), scale);
        
        // Дараалал чухал: Scale -> Rotate -> Translate
        return translation_matrix * rotation_matrix * scaling_matrix;
    }

    // Хүрээ (frame) болгонд хийгдэх шинэчлэлт
    void update(float delta_time) override
    {
        float rotation_speed = 30.0f;
        this->rotation_angle -= rotation_speed * delta_time; // Цагийн зүүний дагуу эргүүлнэ
        if (this->rotation_angle <= -360.0f)
        {
            this->rotation_angle = 0.0f;
        }
    }

    void render() override {}

    ModelGeometry *geometry;
    glm::vec3      scale;
    glm::vec3      position;
    float          rotation_angle;
};

/**
 * бүх объектууд болон View-ийг агуулна.
 */
class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::Canvas *canvas, Viewer *viewer) 
    {
        this->canvas = canvas;
        this->viewer = viewer;

        float step = 15.0f;
        // 2x2 матрицаар 4 ширхэг сармагчин үүсгэх
        for (int i=0; i<2; ++i)
        {
            for (int j=0; j<2; ++j)
            {
                // Байршлыг тохируулж байна
                this->scene_objects.push_back(new MonkeyObject(glm::vec3(i*step - 7.5f, 0.0f, j*step + 10.0f), glm::vec3(5.0f, 5.0f, 5.0f)));
            }
        }
    }
    ~HelloScene()
    {
        for (auto *obj : this->scene_objects)
        {
            delete obj;
        }
        // Canvas болон Viewer-ийг main-д үүсгэсэн тул энд устгах эсэхээ шийдэх хэрэгтэй.
        // Одоогийн бүтцээр энд заагчийг ашиглаж байгаа тул double-free хийхгүйн тулд болгоомжтой байх.
        // Энэ жишээнд main функц устгалыг хариуцна.
    }

    void process() override {}

    std::vector<shs::AbstractObject3D *> scene_objects;
    shs::Canvas  *canvas;
    Viewer       *viewer;
};

/**
 * 3D геометрийг 2D дэлгэц рүү буулгаж зурах системийг хариуцна.
 * Pipeline: Model -> World -> View -> Projection -> Screen -> Canvas
 */
class RendererSystem : public shs::AbstractSystem
{
public:
    RendererSystem(HelloScene *scene) : scene(scene) {}

    void process(float delta_time) override
    {
        // Камерын View болон Projection матрицыг авна
        glm::mat4 view_matrix       = this->scene->viewer->camera->view_matrix;
        glm::mat4 projection_matrix = this->scene->viewer->camera->projection_matrix;

        for (shs::AbstractObject3D *object : this->scene->scene_objects)
        {
            // Зөвхөн MonkeyObject төрлийн объектуудыг зурна
            if (typeid(*object) == typeid(MonkeyObject))
            {
                MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                if (monkey)
                {
                    glm::mat4 model_matrix = monkey->get_world_matrix();

                    // Гурвалжин бүрээр давталт хийнэ
                    for (size_t i = 0; i < monkey->geometry->triangles.size(); i += 3)
                    {
                        // CLIP SPACE Тооцоолол
                        // P * V * M * Vertex гэсэн дарааллаар үржүүлнэ.
                        glm::vec4 vertex1_clip_space = projection_matrix * (view_matrix * (model_matrix * glm::vec4(monkey->geometry->triangles[i    ], 1.0f)));
                        glm::vec4 vertex2_clip_space = projection_matrix * (view_matrix * (model_matrix * glm::vec4(monkey->geometry->triangles[i + 1], 1.0f)));
                        glm::vec4 vertex3_clip_space = projection_matrix * (view_matrix * (model_matrix * glm::vec4(monkey->geometry->triangles[i + 2], 1.0f))); // ЗАСВАР: vertex3 ашигласан

                        // CLIPPING (Энгийн шалгалт)
                        // W < 0.1 байвал камерын ард байна гэсэн үг тул зурахгүй
                        if (vertex1_clip_space.w < 0.1f || vertex2_clip_space.w < 0.1f || vertex3_clip_space.w < 0.1f) continue;

                        // SCREEN SPACE & CANVAS SPACE Хөрвүүлэлт
                        // clip_to_screen нь (0,0)-ийг Зүүн Дээд (Top-Left) буланд өгдөг.
                        glm::vec3 v1_screen = shs::Canvas::clip_to_screen(vertex1_clip_space, CANVAS_WIDTH, CANVAS_HEIGHT);
                        glm::vec3 v2_screen = shs::Canvas::clip_to_screen(vertex2_clip_space, CANVAS_WIDTH, CANVAS_HEIGHT);
                        glm::vec3 v3_screen = shs::Canvas::clip_to_screen(vertex3_clip_space, CANVAS_WIDTH, CANVAS_HEIGHT);

                        // Canvas дээр зурахад (0,0) нь Зүүн Доод (Bottom-Left) буланд байх ёстой.
                        // Тиймээс Y координатыг (Height - Y) гэж хөрвүүлнэ.
                        auto to_canvas_coords = [](glm::vec3 screen_pt, float height) -> glm::ivec2 {
                            return glm::ivec2((int)screen_pt.x, (int)(height - screen_pt.y));
                        };

                        glm::ivec2 p1 = to_canvas_coords(v1_screen, CANVAS_HEIGHT);
                        glm::ivec2 p2 = to_canvas_coords(v2_screen, CANVAS_HEIGHT);
                        glm::ivec2 p3 = to_canvas_coords(v3_screen, CANVAS_HEIGHT);

                        // WIREFRAME ЗУРАХ
                        // Canvas дээр шулуун зурна
                        shs::Canvas::draw_line(*this->scene->canvas, p1.x, p1.y, p2.x, p2.y, shs::Pixel::green_pixel());
                        shs::Canvas::draw_line(*this->scene->canvas, p1.x, p1.y, p3.x, p3.y, shs::Pixel::green_pixel());
                        shs::Canvas::draw_line(*this->scene->canvas, p2.x, p2.y, p3.x, p3.y, shs::Pixel::green_pixel());
                    }
                }
            }
        }
    }
private:
    HelloScene *scene;
};

/**
 * Тоглоомын логик, хөдөлгөөн, update хийх үйлдлүүдийг хариуцна.
 */
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

/**
 * SystemProcessor класс
 * Command, Logic, Renderer системүүдийг нэгтгэж удирдах класс.
 */
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
    // Логик боловсруулах (command execute + logic update)
    void process(float delta_time) 
    {
        this->command_processor->process();
        this->logic_system->process(delta_time);
    }
    // Рендер хийх
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
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    // Цонх болон SDL Renderer үүсгэх
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    // Өөрсдийн Canvas болон Texture үүсгэх
    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    // Viewer үүсгэх (Байршил: Z=-50, Хурд: 150)
    // Left-Handed системд +Z нь урагшаа гэж тооцвол камераа -Z дээр тавьж 0 руу харуулна.
    Viewer *viewer = new Viewer(glm::vec3(0.0f, 10.0f, -50.0f), 150.0f);

    HelloScene      *hello_scene      = new HelloScene(main_canvas, viewer);
    SystemProcessor *system_processor = new SystemProcessor(hello_scene);

    bool exit = false;
    SDL_Event event_data;

    // FPS тохиргоо
    int   frame_delay            = 1000 / FRAMES_PER_SECOND; 
    float frame_time_accumulator = 0.0f;
    int   frame_counter          = 0;
    Uint32 delta_frame_time      = 0;

    // Тоглоомын үндсэн давталт (Game Loop)
    while (!exit)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();
        float delta_time_float = delta_frame_time / 1000.0f;

        // INPUT HANDLING - Гараас орж ирэх командуудыг шалгах
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
                    // Хөдөлгөөний командууд
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

        // LOGIC PROCESS - Логик тооцоолол
        system_processor->process(delta_time_float);

        // RENDER PREPARATION - Зурах бэлтгэл
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer); // SDL дэлгэц цэвэрлэх

        // Canvas-аа хараар будаж цэвэрлэх
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Pixel::black_pixel());

        // SOFTWARE RENDER - Програмаар зурах (Wireframe render)
        system_processor->render(delta_time_float);

        // Тест: Санамсаргүй цэгүүд нэмж зурах (optional)
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);

        // PRESENTATION - Дэлгэцэнд гаргах
        // Canvas өгөгдлийг SDL Surface руу хуулах (Coordinate flip энд бас хийгддэг, shs_renderer дотор)
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        // FPS CONTROL - Хурд тохируулах
        frame_counter++;
        delta_frame_time  = SDL_GetTicks() - frame_start_ticks;
        frame_time_accumulator  += delta_frame_time / 1000.0f;

        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
            delta_frame_time = frame_delay; // Delay хийсэн хугацаагаа нэмж тооцно
        }

        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "FPS : " + std::to_string(frame_counter);
            SDL_SetWindowTitle(window, window_title.c_str());
            frame_time_accumulator   = 0.0f;
            frame_counter            = 0;
        }
    }

    // Санах ой цэвэрлэгээ
    delete system_processor;
    delete hello_scene;
    // viewer, canvas нь hello_scene дотор устгагдаж байгаа эсэхийг шалгах, эсвэл энд устгах
    // Энэ тохиолдолд hello_scene нь canvas, viewer-ийг эзэмшдэггүй (pointer авсан) гэж үзвэл:
    delete viewer;
    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}