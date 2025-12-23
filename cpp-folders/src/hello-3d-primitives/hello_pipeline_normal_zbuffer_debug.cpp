/*
    3D Software Renderer - Threaded Split Screen Demo
    Left: Normal Viz | Center: Blinn-Phong | Right: Depth Viz
    Job System Integration
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

#define WINDOW_WIDTH      940
#define WINDOW_HEIGHT     280
#define CANVAS_WIDTH      940
#define CANVAS_HEIGHT     280
#define MOUSE_SENSITIVITY 0.2f
#define THREAD_COUNT      12
#define TILE_SIZE_X       100
#define TILE_SIZE_Y       100

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

struct Uniforms {
    glm::mat4  mvp;          
    glm::mat4  model;        
    glm::vec3  light_dir;    
    glm::vec3  camera_pos;   
    shs::Color color;       
};

// VERTEX SHADER 
shs::Varyings common_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;
    out.position  = u.mvp * glm::vec4(aPos, 1.0f);
    out.world_pos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal    = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv        = glm::vec2(0.0f); 
    return out;
}

// FRAGMENT SHADER 1: NORMAL VISUALIZER
shs::Color normal_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    glm::vec3 norm = glm::normalize(in.normal);
    // Нормаль вектор [-1, 1] хооронд байдаг тул [0, 1] рүү шилжүүлж өнгө болгоно
    glm::vec3 color = (norm + 1.0f) * 0.5f;
    
    return shs::Color{
        (uint8_t)(color.r * 255),
        (uint8_t)(color.g * 255),
        (uint8_t)(color.b * 255),
        255
    };
}

// FRAGMENT SHADER 2: BLINN-PHONG
shs::Color blinn_phong_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    glm::vec3 norm     = glm::normalize(in.normal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir); 
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - in.world_pos); 

    // Ambient
    float ambientStrength = 0.15f;
    glm::vec3 ambient     = ambientStrength * glm::vec3(1.0f, 1.0f, 1.0f);

    // Diffuse
    float diff        = glm::max(glm::dot(norm, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f, 1.0f, 1.0f);

    // Specular (Blinn-Phong)
    glm::vec3 halfwayDir = glm::normalize(lightDir + viewDir);
    float spec = glm::pow(glm::max(glm::dot(norm, halfwayDir), 0.0f), 64.0f);
    glm::vec3 specular = 0.5f * spec * glm::vec3(1.0f, 1.0f, 1.0f);

    glm::vec3 objectColorVec = glm::vec3(u.color.r, u.color.g, u.color.b) / 255.0f;
    glm::vec3 result = (ambient + diffuse + specular) * objectColorVec;

    result = glm::clamp(result, 0.0f, 1.0f);

    return shs::Color{
        (uint8_t)(result.r * 255),
        (uint8_t)(result.g * 255),
        (uint8_t)(result.b * 255),
        255
    };
}

// FRAGMENT SHADER 3: DEPTH VISUALIZER
shs::Color depth_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    float z = in.position.z;
    
    // Гүнийг харагдахүйц болгохын тулд хуваана
    float depthVal = z / 40.0f; 
    depthVal = glm::clamp(depthVal, 0.0f, 1.0f);
    float visibility = 1.0f - depthVal;

    return shs::Color{
        (uint8_t)(visibility * 255),
        (uint8_t)(visibility * 255),
        (uint8_t)(visibility * 255),
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
        this->position              = position;
        this->speed                 = speed;
        this->camera                = new shs::Camera3D();
        this->camera->position      = this->position;
        this->camera->width         = float(CANVAS_WIDTH);
        this->camera->height        = float(CANVAS_HEIGHT);
        this->camera->field_of_view = 60.0f;
        this->camera->z_near        = 0.1f;
        this->camera->z_far         = 1000.0f;
        
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
    glm::vec3      position;
    float          horizontal_angle;
    float          vertical_angle;
    float          speed;
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
        this->rotation_angle = -30.0f; 
    }
    ~MonkeyObject() { delete this->geometry; }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }
    void update(float delta_time) override { (void)delta_time; }
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
        this->light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));
        
        this->scene_objects.push_back(new MonkeyObject(
            glm::vec3(0.0f, 0.0f, 10.0f), 
            glm::vec3(4.0f), 
            shs::Color{60, 100, 200, 255} 
        ));
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
// RENDERER SYSTEM (THREADED)
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

        // [RASTER PREP] - Tile дотор хязгаарлах
        glm::vec2 bboxmin(tile_max.x, tile_max.y);
        glm::vec2 bboxmax(tile_min.x, tile_min.y);
        std::vector<glm::vec2> v2d = { glm::vec2(screen_coords[0]), glm::vec2(screen_coords[1]), glm::vec2(screen_coords[2]) };

        for (int i = 0; i < 3; i++) {
            bboxmin = glm::max(glm::vec2(tile_min), glm::min(bboxmin, v2d[i]));
            bboxmax = glm::min(glm::vec2(tile_max), glm::max(bboxmax, v2d[i]));
        }

        if (bboxmin.x > bboxmax.x || bboxmin.y > bboxmax.y) return;

        float area = (v2d[1].x - v2d[0].x) * (v2d[2].y - v2d[0].y) - (v2d[1].y - v2d[0].y) * (v2d[2].x - v2d[0].x);
        if (area <= 0) return;

        // [FRAGMENT STAGE]
        for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
            for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {
                
                glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                float z = bc.x * screen_coords[0].z + bc.y * screen_coords[1].z + bc.z * screen_coords[2].z;
                if (z_buffer.test_and_set_depth(px, py, z)) {
                    
                    shs::Varyings interpolated;
                    interpolated.normal    = glm::normalize(bc.x * vout[0].normal    + bc.y * vout[1].normal    + bc.z * vout[2].normal);
                    interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                    interpolated.position  = bc.x * vout[0].position  + bc.y * vout[1].position  + bc.z * vout[2].position; // Depth viz-д хэрэгтэй

                    canvas.draw_pixel_screen_space(px, py, fragment_shader(interpolated));
                }
            }
        }
    }

    void process(float delta_time) override
    {
        this->z_buffer->clear();

        glm::mat4 view = this->scene->viewer->camera->view_matrix;
        glm::mat4 proj = this->scene->viewer->camera->projection_matrix;

        // Дэлгэцийг 3 хуваахын тулд Viewport матрицуудыг бэлдэнэ
        glm::mat4 viewLeft = glm::translate(glm::mat4(1.0f), glm::vec3(-0.666f, 0.0f, 0.0f)) * 
                             glm::scale(glm::mat4(1.0f), glm::vec3(0.333f, 1.0f, 1.0f));

        glm::mat4 viewCenter = glm::scale(glm::mat4(1.0f), glm::vec3(0.333f, 1.0f, 1.0f));

        glm::mat4 viewRight = glm::translate(glm::mat4(1.0f), glm::vec3(0.666f, 0.0f, 0.0f)) * 
                              glm::scale(glm::mat4(1.0f), glm::vec3(0.333f, 1.0f, 1.0f));

        int w = this->scene->canvas->get_width();
        int h = this->scene->canvas->get_height();
        
        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wait_group.reset();

        for(int ty = 0; ty < rows; ty++) {
            for(int tx = 0; tx < cols; tx++) {
                
                wait_group.add(1);

                // Job System-д ажлыг илгээнэ
                job_system->submit({[this, tx, ty, w, h, view, proj, viewLeft, viewCenter, viewRight]() {
                    
                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, w) - 1, std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                    for (shs::AbstractObject3D *object : this->scene->scene_objects)
                    {
                        MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                        if (!monkey) continue;

                        const auto& verts = monkey->geometry->triangles;
                        const auto& norms = monkey->geometry->normals;
                        glm::mat4 model = monkey->get_world_matrix();

                        // PASS 1: LEFT (Normal Visualizer)
                        Uniforms uLeft;
                        uLeft.model = model;
                        uLeft.mvp   = viewLeft * proj * view * model;
                        uLeft.light_dir = this->scene->light_direction;
                        uLeft.camera_pos = this->scene->viewer->position;
                        uLeft.color = monkey->color;

                        // PASS 2: CENTER (Blinn-Phong)
                        Uniforms uCenter = uLeft;
                        uCenter.mvp = viewCenter * proj * view * model;

                        // PASS 3: RIGHT (Depth Visualizer)
                        Uniforms uRight = uLeft;
                        uRight.mvp = viewRight * proj * view * model;

                        for (size_t i = 0; i < verts.size(); i += 3) {
                            std::vector<glm::vec3> t_verts = {verts[i], verts[i+1], verts[i+2]};
                            std::vector<glm::vec3> t_norms = {norms[i], norms[i+1], norms[i+2]};

                            // Tile бүр дээр 3 өөр Shader-ыг дуудна.
                            // Rasterizer нь тухайн Tile-аас гарсан пикселийг автоматаар зурахгүй (clipping)
                            
                            // Draw Left
                            draw_triangle_tile(*this->scene->canvas, *this->z_buffer, t_verts, t_norms,
                                [&uLeft](const glm::vec3& p, const glm::vec3& n) { return common_vertex_shader(p, n, uLeft); },
                                [&uLeft](const shs::Varyings& v) { return normal_fragment_shader(v, uLeft); },
                                t_min, t_max);

                            // Draw Center
                            draw_triangle_tile(*this->scene->canvas, *this->z_buffer, t_verts, t_norms,
                                [&uCenter](const glm::vec3& p, const glm::vec3& n) { return common_vertex_shader(p, n, uCenter); },
                                [&uCenter](const shs::Varyings& v) { return blinn_phong_fragment_shader(v, uCenter); },
                                t_min, t_max);

                            // Draw Right
                            draw_triangle_tile(*this->scene->canvas, *this->z_buffer, t_verts, t_norms,
                                [&uRight](const glm::vec3& p, const glm::vec3& n) { return common_vertex_shader(p, n, uRight); },
                                [&uRight](const shs::Varyings& v) { return depth_fragment_shader(v, uRight); },
                                t_min, t_max);
                        }
                    }
                    wait_group.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }

        wait_group.wait();

        // Дэлгэц хуваах шугамууд зурах (Үндсэн thread)
        shs::Pixel white = shs::Pixel::white_pixel();
        shs::Canvas::draw_line(*this->scene->canvas, w/3, 0, w/3, h, white);
        shs::Canvas::draw_line(*this->scene->canvas, (w/3)*2, 0, (w/3)*2, h, white);
    }
private:
    HelloScene   *scene;
    shs::ZBuffer *z_buffer;
    shs::Job::ThreadedPriorityJobSystem *job_system;
    shs::Job::WaitGroup wait_group;
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

    // Job System эхлүүлэх
    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    
    shs::Canvas *main_canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    Viewer *viewer = new Viewer(glm::vec3(0.0f, 5.0f, -20.0f), 50.0f);
    HelloScene *hello_scene = new HelloScene(main_canvas, viewer);
    SystemProcessor *sys = new SystemProcessor(hello_scene, job_system);

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
    delete job_system;
    
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}