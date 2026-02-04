/*
    3D Software Renderer - Threaded Gouraud Shading Pipeline
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

#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480
#define MOUSE_SENSITIVITY 0.2f
#define THREAD_COUNT      12
#define TILE_SIZE_X       80
#define TILE_SIZE_Y       80

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
shs::Varyings gouraud_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const Uniforms& u)
{
    shs::Varyings out;
    out.position = u.mvp * glm::vec4(aPos, 1.0f);

    glm::vec3 worldPos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    glm::vec3 normal   = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - worldPos);

    float ambientStrength = 0.15f;
    glm::vec3 ambient     = ambientStrength * glm::vec3(1.0f, 1.0f, 1.0f);

    float diff        = glm::max(glm::dot(normal, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f, 1.0f, 1.0f);

    glm::vec3 halfwayDir   = glm::normalize(lightDir + viewDir);
    float specularStrength = 0.5f;
    float shininess        = 32.0f; 

    float spec         = glm::pow(glm::max(glm::dot(normal, halfwayDir), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f, 1.0f, 1.0f);

    glm::vec3 objectColorVec = glm::vec3(u.color.r, u.color.g, u.color.b) / 255.0f;
    glm::vec3 finalColor     = (ambient + diffuse + specular) * objectColorVec;
    finalColor               = glm::clamp(finalColor, 0.0f, 1.0f);

    // HACK: Varyings дотор 'color' байхгүй тул 'world_pos' талбарыг ашиглан өнгөө дамжуулна.
    out.world_pos = finalColor; 

    return out;
}

// FRAGMENT SHADER (Gouraud)
shs::Color gouraud_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    glm::vec3 color = in.world_pos;
    return shs::Color{
        (uint8_t)(color.r * 255),
        (uint8_t)(color.g * 255),
        (uint8_t)(color.b * 255),
        255
    };
}

// ==========================================
// SCENE & OBJECT CLASSES (Same logic)
// ==========================================
// [Viewer, ModelGeometry, MonkeyObject, HelloScene classes are identical to previous examples]

class Viewer {
public:
    Viewer(glm::vec3 pos, float spd) : position(pos), speed(spd) {
        camera = new shs::Camera3D(); camera->position = pos;
        camera->width = CANVAS_WIDTH; camera->height = CANVAS_HEIGHT;
        camera->field_of_view = 60.0f; camera->z_near = 0.1f; camera->z_far = 1000.0f;
        horizontal_angle = 0.0f; vertical_angle = 0.0f;
    }
    Viewer(glm::vec3 pos, float spd, int w, int h) : position(pos), speed(spd) {
        camera = new shs::Camera3D();
        camera->position = pos;
        camera->width = w;
        camera->height = h;
        camera->field_of_view = 60.0f;
        camera->z_near = 0.1f;
        camera->z_far = 1000.0f;
        horizontal_angle = 0.0f;
        vertical_angle = 0.0f;
    }
    ~Viewer() { delete camera; }
    void update() {
        camera->position         = position;
        camera->horizontal_angle = horizontal_angle;
        camera->vertical_angle   = vertical_angle;
        camera->update();
    }
    glm::vec3 get_direction_vector() { return camera->direction_vector; }
    glm::vec3 get_right_vector() { return camera->right_vector; }
    shs::Camera3D *camera; glm::vec3 position; float horizontal_angle, vertical_angle, speed;
};

class ModelGeometry {
public:
    ModelGeometry(std::string path) {
        const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs | aiProcess_JoinIdenticalVertices);
        if(!scene || !scene->mRootNode) return;
        aiMesh *mesh = scene->mMeshes[0];
        for(unsigned int i=0; i<mesh->mNumFaces; i++) {
            for(int k=0; k<3; k++) {
                aiVector3D v = mesh->mVertices[mesh->mFaces[i].mIndices[k]];
                triangles.push_back(glm::vec3(v.x, v.y, v.z));
                if(mesh->HasNormals()) { aiVector3D n = mesh->mNormals[mesh->mFaces[i].mIndices[k]]; normals.push_back(glm::vec3(n.x, n.y, n.z)); }
            }
        }
    }
    std::vector<glm::vec3> triangles, normals; Assimp::Importer importer;
};

class MonkeyObject : public shs::AbstractObject3D {
public:
    MonkeyObject(glm::vec3 pos, glm::vec3 scl, shs::Color col) : position(pos), scale(scl), color(col), rotation_angle(0.0f) { geometry = new ModelGeometry("./obj/monkey/monkey.rawobj"); }
    ~MonkeyObject() { delete geometry; }
    glm::mat4 get_world_matrix() override {
        return glm::translate(glm::mat4(1.0f), position) * glm::rotate(glm::mat4(1.0f), glm::radians(rotation_angle), glm::vec3(0,1,0)) * glm::scale(glm::mat4(1.0f), scale);
    }
    void update(float dt) override {} 
    void render() override {}
    ModelGeometry *geometry; glm::vec3 scale, position; shs::Color color; float rotation_angle;
};

class HelloScene : public shs::AbstractSceneState {
public:
    HelloScene(shs::Canvas *cvs, Viewer *vwr) : canvas(cvs), viewer(vwr) {
        light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));
        scene_objects.push_back(new MonkeyObject(glm::vec3(0,0,10), glm::vec3(4), shs::Color{60,100,200,255}));
    }
    ~HelloScene() { for(auto *o : scene_objects) delete o; }
    void process() override {}
    std::vector<shs::AbstractObject3D*> scene_objects; shs::Canvas *canvas; Viewer *viewer; glm::vec3 light_direction;
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
        shs::Canvas &canvas, shs::ZBuffer &z_buffer,
        const std::vector<glm::vec3> &vertices, const std::vector<glm::vec3> &normals,     
        std::function<shs::Varyings(const glm::vec3&, const glm::vec3&)> vertex_shader,
        std::function<shs::Color(const shs::Varyings&)> fragment_shader,
        glm::ivec2 tile_min, glm::ivec2 tile_max)
    {
        shs::Varyings vout[3];
        glm::vec3 screen_coords[3];

        for (int i = 0; i < 3; i++) {
            vout[i] = vertex_shader(vertices[i], normals[i]);
            screen_coords[i] = shs::Canvas::clip_to_screen(vout[i].position, canvas.get_width(), canvas.get_height());
        }

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

        for (int px = (int)bboxmin.x; px <= (int)bboxmax.x; px++) {
            for (int py = (int)bboxmin.y; py <= (int)bboxmax.y; py++) {
                
                glm::vec3 bc = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                if (bc.x < 0 || bc.y < 0 || bc.z < 0) continue;

                float z = bc.x * screen_coords[0].z + bc.y * screen_coords[1].z + bc.z * screen_coords[2].z;
                if (z_buffer.test_and_set_depth(px, py, z)) {
                    
                    shs::Varyings interpolated;
                    // Gouraud дээр 'world_pos' нь өнгө хадгалж байгаа тул interpolate хийхэд өнгө гарна
                    interpolated.world_pos = bc.x * vout[0].world_pos + bc.y * vout[1].world_pos + bc.z * vout[2].world_pos;
                    
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

        int w = this->scene->canvas->get_width();
        int h = this->scene->canvas->get_height();
        
        int cols = (w + TILE_SIZE_X - 1) / TILE_SIZE_X;
        int rows = (h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

        wait_group.reset();

        for(int ty = 0; ty < rows; ty++) {
            for(int tx = 0; tx < cols; tx++) {
                
                wait_group.add(1);

                job_system->submit({[this, tx, ty, w, h, view, proj]() {
                    
                    glm::ivec2 t_min(tx * TILE_SIZE_X, ty * TILE_SIZE_Y);
                    glm::ivec2 t_max(std::min((tx + 1) * TILE_SIZE_X, w) - 1, std::min((ty + 1) * TILE_SIZE_Y, h) - 1);

                    for (shs::AbstractObject3D *object : this->scene->scene_objects)
                    {
                        MonkeyObject *monkey = dynamic_cast<MonkeyObject *>(object);
                        if (!monkey) continue;

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

                            draw_triangle_tile(
                                *this->scene->canvas,
                                *this->z_buffer,
                                tri_verts,
                                tri_norms,
                                [&uniforms](const glm::vec3& p, const glm::vec3& n) {
                                    return gouraud_vertex_shader(p, n, uniforms);
                                },
                                [&uniforms](const shs::Varyings& v) {
                                    return gouraud_fragment_shader(v, uniforms);
                                },
                                t_min, t_max
                            );
                        }
                    }
                    wait_group.done();
                }, shs::Job::PRIORITY_HIGH});
            }
        }
        
        wait_group.wait();
    }
private:
    HelloScene   *scene;
    shs::ZBuffer *z_buffer;
    shs::Job::ThreadedPriorityJobSystem *job_system;
    shs::Job::WaitGroup wait_group;
};

class LogicSystem : public shs::AbstractSystem {
public:
    LogicSystem(HelloScene *scene) : scene(scene) {}
    void process(float delta_time) override { scene->viewer->update(); for(auto *o : scene->scene_objects) o->update(delta_time); }
private: HelloScene *scene;
};

class SystemProcessor {
public:
    SystemProcessor(HelloScene *scene, shs::Job::ThreadedPriorityJobSystem *job_sys) {
        command_processor = new shs::CommandProcessor();
        renderer_system = new RendererSystem(scene, job_sys);
        logic_system = new LogicSystem(scene);
    }
    ~SystemProcessor() { delete command_processor; delete renderer_system; delete logic_system; }
    void process(float dt) { command_processor->process(); logic_system->process(dt); }
    void render(float dt) { renderer_system->process(dt); }
    shs::CommandProcessor *command_processor; LogicSystem *logic_system; RendererSystem *renderer_system;
};

int main(int argc, char* argv[]) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window *window; SDL_Renderer *renderer;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    shs::Canvas *main_canvas = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    Viewer *viewer = new Viewer(glm::vec3(0, 5, -20), 50.0f, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    HelloScene *hello_scene = new HelloScene(main_canvas, viewer);
    SystemProcessor *sys = new SystemProcessor(hello_scene, job_system);

    bool exit = false; SDL_Event event_data; Uint32 last_tick = SDL_GetTicks(); bool is_dragging = false;
    while (!exit) {
        Uint32 current_tick = SDL_GetTicks(); float delta_time = (current_tick - last_tick) / 1000.0f; last_tick = current_tick;
        while (SDL_PollEvent(&event_data)) {
            if (event_data.type == SDL_QUIT) exit = true;
            if (event_data.type == SDL_MOUSEBUTTONDOWN && event_data.button.button == SDL_BUTTON_LEFT) is_dragging = true;
            if (event_data.type == SDL_MOUSEBUTTONUP && event_data.button.button == SDL_BUTTON_LEFT) is_dragging = false;
            if (event_data.type == SDL_MOUSEMOTION && is_dragging) {
                viewer->horizontal_angle += event_data.motion.xrel * MOUSE_SENSITIVITY;
                viewer->vertical_angle -= event_data.motion.yrel * MOUSE_SENSITIVITY;
                if (viewer->vertical_angle > 89.0f) viewer->vertical_angle = 89.0f;
                if (viewer->vertical_angle < -89.0f) viewer->vertical_angle = -89.0f;
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
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color::black()); 
        sys->render(delta_time);
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);
    }
    delete sys; delete hello_scene; delete viewer; delete main_canvas; delete job_system;
    SDL_DestroyTexture(screen_texture); SDL_FreeSurface(main_sdlsurface); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit();
    return 0;
}