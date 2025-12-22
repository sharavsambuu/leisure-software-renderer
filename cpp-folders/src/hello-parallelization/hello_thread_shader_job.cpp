/*
    PIXEL-BY-PIXEL JOB SUBMISSION EXAMPLE
*/

#include <SDL2/SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/noise.hpp> 
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <array>
#include <cstdlib>
#include <cmath>
#include <tuple>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "shs_renderer.hpp"

// Тохиргоонууд
#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       640
#define WINDOW_HEIGHT      520
#define CANVAS_WIDTH       360
#define CANVAS_HEIGHT      240
#define CONCURRENCY_COUNT  8
#define NUM_OCTAVES        5


// Job system синхрончлол
std::atomic<int> atomic_counter(0);
std::mutex mtx;
std::condition_variable cv;


// Векторын утгыг [0, 1]-ээс [0, 255] руу хөрвүүлэх
glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec) {
    glm::vec4 clamped_value = glm::clamp(input_vec, 0.0f, 1.0f);
    glm::vec4 scaled_value  = clamped_value * 255.0f;
    return scaled_value;
}

// Fractal Brownian Motion Noise Functions
float fbm(const glm::vec2& st) {
    glm::vec2 _st = st;
    float v = 0.0f;
    float a = 0.5f;
    glm::vec2 shift(100.0f);
    
    glm::mat2 rot(cos(0.5f), sin(0.5f),
                  -sin(0.5f), cos(0.5f));

    for (int i = 0; i < NUM_OCTAVES; ++i) {
        // glm::simplex нь glm/gtc/noise.hpp дотор байдаг
        v += a * glm::simplex(_st); 
        _st = rot * _st * 2.0f + shift;
        a *= 0.5f;
    }

    return v;
}

// Fragment Shader Logic
glm::vec4 fragment_shader(glm::vec2 uniform_uv, float uniform_time)
{
    glm::vec2 st = (uniform_uv/glm::vec2(CANVAS_WIDTH, CANVAS_HEIGHT))*3.0f;
    st += float(glm::abs(glm::sin(uniform_time*0.1f)*3.0f))*st;
    glm::vec3 color(0.0);

    glm::vec2 q(0.0);
    q.x = fbm(st + 0.00f * uniform_time);
    q.y = fbm(st + glm::vec2(1.0f));

    glm::vec2 r(0.0f);
    r.x = fbm(st + 1.0f * q + glm::vec2(1.7f, 9.2f) + 0.15f * uniform_time);
    r.y = fbm(st + 1.0f * q + glm::vec2(8.3f, 2.8f) + 0.126f * uniform_time);

    float f = fbm(st + r);

    color = glm::mix(glm::vec3(0.101961f, 0.619608f, 0.666667f),
                     glm::vec3(0.666667f, 0.666667f, 0.498039f),
                     glm::clamp((f * f) * 4.0f, 0.0f, 1.0f));

    color = glm::mix(color,
                     glm::vec3(0.0f, 0.0f, 0.164706f),
                     glm::clamp(glm::length(q), 0.0f, 1.0f));

    color = glm::mix(color,
                     glm::vec3(0.666667f, 1.0f, 1.0f),
                     glm::clamp(glm::length(r.x), 0.0f, 1.0f));
    
    glm::vec4 output_arr = glm::vec4(color*float(f*f*f+0.6f*f*f+0.5*f),1.0f);
    return rescale_vec4_1_255(output_arr);
};


int main(int argc, char* argv[])
{
    // Job System эхлүүлэх
    shs::Job::AbstractJobSystem *job_system = new shs::Job::ThreadedPriorityJobSystem(CONCURRENCY_COUNT);

    // SDL Init
    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }
    
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    // 3. Canvas setup
    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);


    bool exit_loop = false;
    SDL_Event event_data;

    int    frame_delay            = 1000 / FRAMES_PER_SECOND;
    float  frame_time_accumulator = 0;
    int    frame_counter          = 0;
    float  time_accumulator       = 0.0;


    while (!exit_loop)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();

        // -----------------------------
        // INPUT HANDLING 
        // -----------------------------
        while (SDL_PollEvent(&event_data))
        {
            switch (event_data.type)
            {
            case SDL_QUIT:
                exit_loop = true;
                // Энд job_system->is_running = false гэж бичиж болохгүй.
                // Thread-үүдийг зогсоовол доорх wait гацна.
                break;
            case SDL_KEYDOWN:
                if (event_data.key.keysym.sym == SDLK_ESCAPE) {
                    exit_loop = true;
                }
                break;
            }
        }

        // Хэрэв гарах комманд ирсэн бол render хийхгүйгээр шууд давталтаас гарна.
        if (exit_loop) break; 


        // -----------------------------
        // RENDER PREPARATION
        // -----------------------------
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        // Safety reset (хэдийгээр 0 байх ёстой ч баталгаажуулж байна)
        atomic_counter.store(0);


        // -----------------------------
        // JOB SUBMISSION
        // -----------------------------
        // Пиксел бүрээр гүйж fragment shader ажиллуулах
        for (int x = 0; x < CANVAS_WIDTH; x++)
        {
            for (int y = 0; y < CANVAS_HEIGHT; y++)
            {
                // Race Condition сэргийлэх
                // Job илгээхээс өмнө тоолуурыг нэмнэ.
                atomic_counter.fetch_add(1, std::memory_order_relaxed);

                job_system->submit({[x, y, time_accumulator, main_canvas]() {
                    
                    // --- WORKER THREAD CODE ---
                    glm::vec2 uv = {float(x), float(y)};
                    glm::vec4 shader_output = fragment_shader(uv, time_accumulator);
                    
                    // Pixel зурах
                    shs::Canvas::draw_pixel(*main_canvas, x, y, shs::Color{
                        (uint8_t)shader_output[0], 
                        (uint8_t)shader_output[1], 
                        (uint8_t)shader_output[2], 
                        (uint8_t)shader_output[3]
                    });

                    // Ажил дууссан тул counter-ийг хасна.
                    // Хэрэв үр дүн 1 гэж буцаж ирвэл энэ нь хасахаас өмнө 1 байсан,
                    // одоо 0 болсон гэсэн үг. Тэгвэл Main Thread-д мэдэгдэнэ.
                    if (atomic_counter.fetch_sub(1, std::memory_order_release) == 1) {
                         std::lock_guard<std::mutex> lock(mtx);
                         cv.notify_one(); 
                    }
                    // --------------------------

                }, shs::Job::PRIORITY_NORMAL});
            }
        }

        // -----------------------------
        // WAIT FOR GPU (THREADS)
        // -----------------------------
        {
            std::unique_lock<std::mutex> lock(mtx);
            // Counter 0 болтол хүлээнэ (Бүх thread ажлаа дуустал)
            cv.wait(lock, []{ return atomic_counter.load(std::memory_order_acquire) == 0; });
        }

        // Debug pixel (Render хийгдэж байгааг шалгах)
        shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);


        // -----------------------------
        // PRESENT TO SCREEN
        // -----------------------------
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

    
        // -----------------------------
        // FPS CALCULATION
        // -----------------------------
        frame_counter++;
        Uint32 delta_frame_time  = SDL_GetTicks() - frame_start_ticks;
        frame_time_accumulator  += delta_frame_time / 1000.0f;
        time_accumulator        += delta_frame_time / 1000.0f;

        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "FPS : " + std::to_string(frame_counter);
            frame_time_accumulator   = 0.0f;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }


    // -----------------------------
    // CLEANUP
    // -----------------------------
    
    // Destructor нь ажиллах үед Thread-үүдээ зөв зогсоож (join хийж) цэвэрлэнэ.
    delete job_system; 
    delete main_canvas;
    
    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    std::cout << "Application exited successfully." << std::endl;
    return 0;
}