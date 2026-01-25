/*
    3D Software Renderer - Threaded Blinn-Phong + Texture Mapping + OBJ UV

    - my header returns (v, w, u) barycentric but raster assumes (u, v, w).
    - Depth should use NDC z, not clip.w.
    - UV and world_pos interpolation should be perspective-correct using 1/w.

    3D Model credit :
        Subaru by mednios
        https://free3d.com/3d-model/my-subaru-43836.html
*/

#include <string>
#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>

// External Libraries
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
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
#define MOUSE_SENSITIVITY 0.2f
#define THREAD_COUNT      20
#define TILE_SIZE_X       80
#define TILE_SIZE_Y       80

// Хэрвээ texture урвуу харагдвал 1
#define UV_FLIP_V 0

// ==========================================
// UNIFORMS & SHADERS
// ==========================================

struct Uniforms {
    glm::mat4  mvp;
    glm::mat4  model;
    glm::vec3  light_dir;
    glm::vec3  camera_pos;

    shs::Color color;

    const shs::Texture2D *albedo = nullptr;
    bool use_texture = false;
};

static inline float clamp01(float v) { return (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v); }

static inline shs::Color sample_nearest(const shs::Texture2D &tex, glm::vec2 uv)
{
    float u = uv.x;
    float v = uv.y;

#if UV_FLIP_V
    v = 1.0f - v;
#endif

    u = clamp01(u);
    v = clamp01(v);

    int x = (int)std::lround(u * (float)(tex.w - 1));
    int y = (int)std::lround(v * (float)(tex.h - 1));

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= tex.w) x = tex.w - 1;
    if (y >= tex.h) y = tex.h - 1;

    return tex.texels.at(x, y);
}

/*
    VERTEX SHADER (Blinn-Phong + UV pass-through)
*/
shs::Varyings blinn_phong_tex_vertex_shader(const glm::vec3& aPos, const glm::vec3& aNormal, const glm::vec2& aUV, const Uniforms& u)
{
    shs::Varyings out;
    out.position  = u.mvp * glm::vec4(aPos, 1.0f);
    out.world_pos = glm::vec3(u.model * glm::vec4(aPos, 1.0f));
    out.normal    = glm::normalize(glm::mat3(glm::transpose(glm::inverse(u.model))) * aNormal);
    out.uv        = aUV;
    return out;
}

/*
    FRAGMENT SHADER (Blinn-Phong, texture albedo)
*/
shs::Color blinn_phong_tex_fragment_shader(const shs::Varyings& in, const Uniforms& u)
{
    glm::vec3 norm     = glm::normalize(in.normal);
    glm::vec3 lightDir = glm::normalize(-u.light_dir);
    glm::vec3 viewDir  = glm::normalize(u.camera_pos - in.world_pos);

    float ambientStrength = 0.15f;
    glm::vec3 ambient     = ambientStrength * glm::vec3(1.0f);

    float diff        = glm::max(glm::dot(norm, lightDir), 0.0f);
    glm::vec3 diffuse = diff * glm::vec3(1.0f);

    glm::vec3 halfwayDir   = glm::normalize(lightDir + viewDir);
    float specularStrength = 0.5f;
    float shininess        = 64.0f;

    float spec         = glm::pow(glm::max(glm::dot(norm, halfwayDir), 0.0f), shininess);
    glm::vec3 specular = specularStrength * spec * glm::vec3(1.0f);

    glm::vec3 baseColor;

    if (u.use_texture && u.albedo && u.albedo->valid()) {
        shs::Color tc = sample_nearest(*u.albedo, in.uv);
        baseColor = glm::vec3(tc.r, tc.g, tc.b) / 255.0f;
    } else {
        baseColor = glm::vec3(u.color.r, u.color.g, u.color.b) / 255.0f;
    }

    glm::vec3 result = (ambient + diffuse + specular) * baseColor;
    result = glm::clamp(result, 0.0f, 1.0f);

    return shs::Color{
        (uint8_t)(result.r * 255),
        (uint8_t)(result.g * 255),
        (uint8_t)(result.b * 255),
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
        this->horizontal_angle      = 0.0f;
        this->vertical_angle        = 0.0f;
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
        unsigned int flags =
            aiProcess_Triangulate |
            aiProcess_GenSmoothNormals |
            aiProcess_JoinIdenticalVertices;

        const aiScene *scene = this->importer.ReadFile(model_path.c_str(), flags);
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Model load error: " << this->importer.GetErrorString() << std::endl;
            return;
        }

        for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
            aiMesh *mesh = scene->mMeshes[i];

            bool has_uv = mesh->HasTextureCoords(0);

            for (unsigned int j = 0; j < mesh->mNumFaces; ++j) {
                if (mesh->mFaces[j].mNumIndices != 3) continue;

                for(int k = 0; k < 3; k++) {
                    unsigned int idx = mesh->mFaces[j].mIndices[k];

                    aiVector3D v = mesh->mVertices[idx];
                    this->triangles.push_back(glm::vec3(v.x, v.y, v.z));

                    if (mesh->HasNormals()) {
                        aiVector3D n = mesh->mNormals[idx];
                        this->normals.push_back(glm::vec3(n.x, n.y, n.z));
                    } else {
                        this->normals.push_back(glm::vec3(0, 0, 1));
                    }

                    if (has_uv) {
                        aiVector3D t = mesh->mTextureCoords[0][idx];
                        this->uvs.push_back(glm::vec2(t.x, t.y));
                    } else {
                        this->uvs.push_back(glm::vec2(0.0f));
                    }
                }
            }
        }
    }

    std::vector<glm::vec3> triangles;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;

    Assimp::Importer importer;
};

class SubaruObject : public shs::AbstractObject3D
{
public:
    SubaruObject(glm::vec3 position, glm::vec3 scale, shs::Color color, const shs::Texture2D *albedo)
    {
        this->position       = position;
        this->scale          = scale;
        this->color          = color;
        this->geometry       = new ModelGeometry("./obj/subaru/SUBARU_1.obj");
        this->rotation_angle = 0.0f;
        this->albedo         = albedo;
    }
    ~SubaruObject() { delete this->geometry; }

    glm::mat4 get_world_matrix() override
    {
        glm::mat4 t = glm::translate(glm::mat4(1.0f), this->position);
        glm::mat4 r = glm::rotate(glm::mat4(1.0f), glm::radians(this->rotation_angle), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 s = glm::scale(glm::mat4(1.0f), scale);
        return t * r * s;
    }

    void update(float delta_time) override
    {
        this->rotation_angle += 15.0f * delta_time;
    }
    void render() override {}

    ModelGeometry        *geometry;
    const shs::Texture2D *albedo;

    glm::vec3      scale;
    glm::vec3      position;
    shs::Color     color;
    float          rotation_angle;
};

class HelloScene : public shs::AbstractSceneState
{
public:
    HelloScene(shs::Canvas *canvas, Viewer *viewer, const shs::Texture2D *albedo)
    {
        this->canvas = canvas;
        this->viewer = viewer;

        this->light_direction = glm::normalize(glm::vec3(-1.0f, -0.4f, 1.0f));

        this->scene_objects.push_back(
            new SubaruObject(glm::vec3(0.0f, 0.0f, 25.0f), glm::vec3(0.08f), shs::Color{200, 200, 200, 255}, albedo)
        );
    }
    ~HelloScene() {
        for (auto *obj : this->scene_objects) delete obj;
    }
    void process() override {}

    std::vector<shs::AbstractObject3D *>  scene_objects;
    shs::Canvas                          *canvas;
    Viewer                               *viewer;
    glm::vec3                             light_direction;
};


// ==========================================
// RENDERER SYSTEM (THREADED RENDERING)
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
        const std::vector<glm::vec2> &uvs,
        std::function<shs::Varyings(const glm::vec3&, const glm::vec3&, const glm::vec2&)> vertex_shader,
        std::function<shs::Color(const shs::Varyings&)> fragment_shader,
        glm::ivec2 tile_min, glm::ivec2 tile_max)
    {
        // [VERTEX STAGE]
        shs::Varyings vout[3];
        glm::vec3 screen_coords[3];
        glm::vec3 ndc_coords[3];
        float invw[3];

        for (int i = 0; i < 3; i++) {
            vout[i] = vertex_shader(vertices[i], normals[i], uvs[i]);

            // screen x/y from helper
            screen_coords[i] = shs::Canvas::clip_to_screen(vout[i].position, canvas.get_width(), canvas.get_height());

            // correct NDC + invw
            float w = vout[i].position.w;
            invw[i] = 1.0f / w;
            ndc_coords[i] = glm::vec3(vout[i].position) * invw[i]; // (x/w, y/w, z/w)
        }

        // [RASTER PREP] - Bounding Box Clamped to Tile
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

                // header barycentric returns (v,w,u) but raster assumes (u,v,w).
                glm::vec3 bc_raw = shs::Canvas::barycentric_coordinate(glm::vec2(px + 0.5f, py + 0.5f), v2d);
                if (bc_raw.x < 0 || bc_raw.y < 0 || bc_raw.z < 0) continue;

                // Remap to (u,v,w) for vertex0/1/2
                glm::vec3 bc(bc_raw.z, bc_raw.x, bc_raw.y);

                // Perspective-correct denominator
                float invw_sum = bc.x * invw[0] + bc.y * invw[1] + bc.z * invw[2];
                if (invw_sum <= 0.0f) continue;

                // Depth (NDC z), perspective-correct
                float z_over_w = bc.x * (ndc_coords[0].z * invw[0]) +
                                 bc.y * (ndc_coords[1].z * invw[1]) +
                                 bc.z * (ndc_coords[2].z * invw[2]);
                float z_ndc = z_over_w / invw_sum; // expected ~[0,1] for LH

                if (z_buffer.test_and_set_depth(px, py, z_ndc)) {

                    shs::Varyings interpolated;

                    // Normal
                    // affine then normalize
                    interpolated.normal =
                        glm::normalize(bc.x * vout[0].normal + bc.y * vout[1].normal + bc.z * vout[2].normal);

                    // World pos
                    // perspective-correct (stabilizes lighting)
                    glm::vec3 wp_over_w =
                        bc.x * (vout[0].world_pos * invw[0]) +
                        bc.y * (vout[1].world_pos * invw[1]) +
                        bc.z * (vout[2].world_pos * invw[2]);
                    interpolated.world_pos = wp_over_w / invw_sum;

                    // UV
                    // perspective-correct 
                    glm::vec2 uv_over_w =
                        bc.x * (vout[0].uv * invw[0]) +
                        bc.y * (vout[1].uv * invw[1]) +
                        bc.z * (vout[2].uv * invw[2]);
                    interpolated.uv = uv_over_w / invw_sum;

                    canvas.draw_pixel_screen_space(px, py, fragment_shader(interpolated));
                }
            }
        }
    }

    void process(float delta_time) override
    {
        (void)delta_time;

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
                        SubaruObject *car = dynamic_cast<SubaruObject *>(object);
                        if (!car) continue;

                        Uniforms uniforms;
                        uniforms.model       = car->get_world_matrix();
                        uniforms.mvp         = proj * view * uniforms.model;
                        uniforms.light_dir   = this->scene->light_direction;
                        uniforms.camera_pos  = this->scene->viewer->position;
                        uniforms.color       = car->color;

                        uniforms.albedo      = car->albedo;
                        uniforms.use_texture = (car->albedo && car->albedo->valid());

                        const auto& verts = car->geometry->triangles;
                        const auto& norms = car->geometry->normals;
                        const auto& uvs   = car->geometry->uvs;

                        for (size_t i = 0; i < verts.size(); i += 3)
                        {
                            std::vector<glm::vec3> tri_verts = { verts[i], verts[i+1], verts[i+2] };
                            std::vector<glm::vec3> tri_norms = { norms[i], norms[i+1], norms[i+2] };
                            std::vector<glm::vec2> tri_uvs   = { uvs[i],   uvs[i+1],   uvs[i+2]   };

                            draw_triangle_tile(
                                *this->scene->canvas,
                                *this->z_buffer,
                                tri_verts,
                                tri_norms,
                                tri_uvs,
                                [&uniforms](const glm::vec3& p, const glm::vec3& n, const glm::vec2& uv) {
                                    return blinn_phong_tex_vertex_shader(p, n, uv, uniforms);
                                },
                                [&uniforms](const shs::Varyings& v) {
                                    return blinn_phong_tex_fragment_shader(v, uniforms);
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
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    IMG_Init(IMG_INIT_PNG | IMG_INIT_JPG);

    auto *job_system = new shs::Job::ThreadedPriorityJobSystem(THREAD_COUNT);

    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    shs::Texture2D car_tex = shs::load_texture_sdl_image("./obj/subaru/SUBARU1_M.bmp", true);

    Viewer          *viewer      = new Viewer(glm::vec3(0.0f, 5.0f, -35.0f), 50.0f);
    HelloScene      *hello_scene = new HelloScene(main_canvas, viewer, &car_tex);
    SystemProcessor *sys         = new SystemProcessor(hello_scene, job_system);

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
                if (event_data.button.button == SDL_BUTTON_LEFT) is_dragging = true;
            }
            if (event_data.type == SDL_MOUSEBUTTONUP) {
                if (event_data.button.button == SDL_BUTTON_LEFT) is_dragging = false;
            }

            if (event_data.type == SDL_MOUSEMOTION)
            {
                if (is_dragging) {
                    viewer->horizontal_angle += event_data.motion.xrel * MOUSE_SENSITIVITY;
                    viewer->vertical_angle   -= event_data.motion.yrel * MOUSE_SENSITIVITY;

                    if (viewer->vertical_angle >  89.0f) viewer->vertical_angle =  89.0f;
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

    IMG_Quit();
    SDL_Quit();

    return 0;
}
