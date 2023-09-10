#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <queue>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480
#define CANVAS_WIDTH      640
#define CANVAS_HEIGHT     480

class MoveForwardCommand : public shs::Command
{
public:
    MoveForwardCommand(glm::vec3 &position, glm::vec3 direction, float speed, float delta_time) : position(position), direction(direction), speed(speed), delta_time(delta_time) {}
    void execute() override
    {
        this->position += this->direction * this->speed * delta_time;
    }

private:
    glm::vec3 &position;
    glm::vec3 direction;
    float speed;
    float delta_time;
};

class MoveBackwardCommand : public shs::Command
{
public:
    MoveBackwardCommand(glm::vec3 &position, glm::vec3 direction, float speed, float delta_time) : position(position), direction(direction), speed(speed), delta_time(delta_time) {}
    void execute() override
    {
        this->position -= this->direction * this->speed * delta_time;
    }

private:
    glm::vec3 &position;
    glm::vec3 direction;
    float speed;
    float delta_time;
};

class MoveRightCommand : public shs::Command
{
public:
    MoveRightCommand(glm::vec3 &position, glm::vec3 right_vector, float speed, float delta_time) : position(position), right_vector(right_vector), speed(speed), delta_time(delta_time) {}
    void execute() override
    {
        this->position += this->right_vector * this->speed * this->delta_time;
    }

private:
    glm::vec3 &position;
    glm::vec3 right_vector;
    float speed;
    float delta_time;
};

class MoveLeftCommand : public shs::Command
{
public:
    MoveLeftCommand(glm::vec3 &position, glm::vec3 right_vector, float speed, float delta_time) : position(position), right_vector(right_vector), speed(speed), delta_time(delta_time) {}
    void execute() override
    {
        this->position -= this->right_vector * this->speed * this->delta_time;
    }

private:
    glm::vec3 &position;
    glm::vec3 right_vector;
    float speed;
    float delta_time;
};

class Viewer
{
public:
    Viewer(glm::vec3 position, float speed)
    {
        this->position = position;
        this->speed = speed;
        this->camera = new shs::Camera3D();
        this->camera->position = this->position;
    }
    ~Viewer() {}

    void update()
    {
        this->camera->position = this->position;
        this->camera->update();
        std::cout << this->position.x << " " << this->position.y << " " << this->position.z << std::endl;
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

class CommandProcessor
{
public:
    void add_command(shs::Command *new_command)
    {
        this->commands.push(new_command);
    }
    void process()
    {
        while (!this->commands.empty())
        {
            shs::Command *command = this->commands.front();
            command->execute();
            delete command;
            this->commands.pop();
        }
    }
private:
    std::queue<shs::Command *> commands;
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
    CommandProcessor *command_processor = new CommandProcessor();


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
                        command_processor->add_command(new MoveForwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_s:
                        command_processor->add_command(new MoveBackwardCommand(viewer->position, viewer->get_direction_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_a:
                        command_processor->add_command(new MoveLeftCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time_float));
                        break;
                    case SDLK_d:
                        command_processor->add_command(new MoveRightCommand(viewer->position, viewer->get_right_vector(), viewer->speed, delta_time_float));
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