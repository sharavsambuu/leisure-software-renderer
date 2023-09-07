
#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480

#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480


class TriangleObject
{
public:
    TriangleObject()
    {
    }
    TriangleObject(glm::vec2 position, float angle_radian, float speed)
    {
        this->position       = position;
        this->angle_radian   = angle_radian;
        this->speed          = speed;
        this->velocity       = this->speed*glm::vec2(glm::cos(this->angle_radian), glm::sin(this->angle_radian));
    }
    ~TriangleObject()
    {
    }
    void update(float delta_time)
    {
        this->update_angle(delta_time);
        this->update_position(delta_time);

        //std::cout << this->position.x << " " << this->position.y << std::endl;
    }
    void update_position(float delta_time)
    {
        this->velocity  = this->speed*glm::vec2(cos(angle_radian), sin(angle_radian));
        this->position += this->velocity*delta_time;
        this->position  = glm::clamp(this->position, glm::vec2(5.0f, 5.0f), glm::vec2(CANVAS_WIDTH-5.0f, CANVAS_HEIGHT-5.0f));

    }
    void update_angle(float delta_time)
    {
        float min_angle_radian = 0.0f;
        float max_angle_radian = 2*M_PI; // 360 degree

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> angle_dist(-13.0f, 13.0f);
        float rotation_speed = 0.3f;
        this->angle_radian  += angle_dist(gen)*delta_time*rotation_speed;
        this->angle_radian   = glm::clamp(this->angle_radian, min_angle_radian, max_angle_radian);

    }
    void render(shs::Canvas &canvas)
    {

        glm::mat4 translation_matrix    = glm::translate(glm::mat4(1.0f), glm::vec3(this->position, 0.0f));
        glm::mat4 rotation_matrix       = glm::rotate(glm::mat4(1.0f), this->angle_radian, glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 scaling_matrix        = glm::scale(glm::mat4(1.0f), glm::vec3(scale, 1.0f));
        glm::mat4 transformation_matrix = translation_matrix * rotation_matrix * scaling_matrix;


        std::vector<glm::vec2> new_vertices(3);
        for (int i = 0; i < 3; ++i) {
            glm::vec4 vertex_4d(this->vertices[i].x, this->vertices[i].y, 0.0f, 1.0f);
            glm::vec4 transformed_vertex = transformation_matrix * vertex_4d;
            new_vertices[i] = glm::vec2(transformed_vertex.x, transformed_vertex.y);
        }

        shs::Canvas::draw_line(canvas, new_vertices[0].x, new_vertices[0].y, new_vertices[1].x, new_vertices[1].y, this->color);
        shs::Canvas::draw_line(canvas, new_vertices[0].x, new_vertices[0].y, new_vertices[2].x, new_vertices[2].y, this->color);
        shs::Canvas::draw_line(canvas, new_vertices[1].x, new_vertices[1].y, new_vertices[2].x, new_vertices[2].y, this->color);

        shs::Canvas::draw_triangle(canvas, new_vertices, shs::Pixel::random_pixel());
    }

    shs::Color color{0, 255, 0, 255};
    float      speed          = 0.5f;
    float      angle_radian   = 0.0f;
    glm::vec2  velocity       = glm::vec2(0.0f);
    glm::vec2  position       = glm::vec2(0.0f);
    glm::vec2  scale          = glm::vec2(3.0f);
    std::vector<glm::vec2> vertices = {
        glm::vec2(-5.5f, -12.0f),
        glm::vec2( 13.3f, -12.0f),
        glm::vec2( 2.4f,  13.2f)
    };

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

    std::vector<TriangleObject> scene = {
        TriangleObject(glm::vec2(200.0f, 390.0f),  45.0f, 6.5f),
        };

    bool exit = false;
    SDL_Event event_data;

    int   frame_delay            = 1000 / FRAMES_PER_SECOND; // Delay for 60 FPS
    float frame_time_accumulator = 0;
    int   frame_counter          = 0;
    int   fps                    = 0;

    while (!exit)
    {

        Uint32 frame_start_ticks = SDL_GetTicks();

        // catching up input events happened on hardware
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
                }
                break;
            }
        }


        // preparing to render on SDL2
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // software rendering or drawing stuffs goes around here
        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Pixel::black_pixel());
        shs::Canvas::fill_pixel(*main_canvas, 10, 10, 20, 30, shs::Pixel::white_pixel());

        for (TriangleObject triangle_object : scene) 
        {
            triangle_object.render(*main_canvas);
        }

        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);

        // actually prensenting canvas data on hardware surface
        shs::Canvas::flip_vertically(*main_canvas); // origin at the left bottom corner of the canvas
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

    
        frame_counter++;
        Uint32 delta_frame_time  = SDL_GetTicks() - frame_start_ticks;

        for (TriangleObject& triangle_object : scene) 
        {
            triangle_object.update(delta_frame_time/1000.0f);
        }

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


    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}