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
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND  60
#define WINDOW_WIDTH       640
#define WINDOW_HEIGHT      520
#define CANVAS_WIDTH       640
#define CANVAS_HEIGHT      520
#define CONCURRENCY_COUNT  8

// Matrix эффектийн нэг үсэгний хэмжээ (пикселээр)
// Энэ тоог багасгавал үсэг жижиг, ихэсгэвэл том болно.
#define FONT_SCALE         16.0f 

glm::vec4 rescale_vec4_1_255(const glm::vec4 &input_vec) {
    glm::vec4 clamped_value = glm::clamp(input_vec, 0.0f, 1.0f);
    glm::vec4 scaled_value  = clamped_value * 255.0f;
    return scaled_value;
}

glm::vec4 fragment_shader(glm::vec2 u_uv, float u_time)
{
    glm::vec2 i = u_uv;
    
    // Grid тооцоолол
    glm::vec2 j = glm::fract(i);
    glm::vec2 floor_i = glm::floor(i); // Бүхэл хэсэг нь баганын индекс болно

    // Унах хурд болон баганын random шилжилт
    float speed = 10.0f; 
    float offset = 18.0f * glm::sin(floor_i.x); 
    
    // Борооны унах координатын тооцоолол
    glm::vec2 p = glm::vec2(0.0f, floor_i.y + static_cast<int>(u_time * (speed + offset)));

    glm::vec4 o(0.0f); 

    // Random brightness (үсэг болгоны тод бүдэг байдал)
    // p векторыг hash хийж random тоо гаргаж байна
    float noise = glm::fract(435.34f * glm::sin(glm::dot(p, glm::vec2(12.9898f, 78.233f))));
    
    o.g = noise; // Ногоон сувагт онооно

    // Grid line (үсэг хоорондын зайг гаргах)
    // j.x болон j.y нь 0-1 хооронд байгаа. 0.75-аас их бол харлуулна (border)
    // Мөн noise нь 0.25-аас бага бол хэт бүдэг тул зурахгүй (matrix trail effect)
    float mask = (noise > 0.1f && j.x < 0.75f && j.y < 0.85f) ? 1.0f : 0.0f;
    
    o *= mask;

    // Өнгийг бага зэрэг цайвар ногоон болгох (Matrix look)
    // o.r = o.g * 0.2f; // Optional: бага зэрэг цайвар болгох

    return rescale_vec4_1_255(o);
};


int main()
{
    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;

    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);


    bool exit = false;
    SDL_Event event_data;

    int    frame_delay            = 1000 / FRAMES_PER_SECOND; 
    float  frame_time_accumulator = 0;
    int    frame_counter          = 0;
    float  time_accumulator       = 0.0;

    while (!exit)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();

        while (SDL_PollEvent(&event_data))
        {
            if (event_data.type == SDL_QUIT) exit = true;
            if (event_data.type == SDL_KEYDOWN && event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;
        }

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        std::vector<std::thread> thread_pool;

        int region_width  = CANVAS_WIDTH  / CONCURRENCY_COUNT;
        int region_height = CANVAS_HEIGHT / CONCURRENCY_COUNT;

        for (int i = 0; i < CONCURRENCY_COUNT; i++) {
            int start_x = i       * region_width;
            int end_x   = (i + 1) * region_width;

            for (int j = 0; j < CONCURRENCY_COUNT; j++) {
                int start_y = j       * region_height;
                int end_y   = (j + 1) * region_height;

                // Lambda дотор variable capture хийхдээ анхаарах
                std::thread task([start_x, end_x, start_y, end_y, time_accumulator, &main_canvas]() {
                    for (int x = start_x; x < end_x; x++) {
                        for (int y = start_y; y < end_y; y++) {
                            
                            // Coordinate Scaling
                            // Пикселийн координатыг тодорхой тоонд хувааж (FONT_SCALE) 
                            // shader-луу илгээнэ. Ингэснээр Matrix-ийн "нүднүүд" томорно.
                            glm::vec2 uv = { float(x) / FONT_SCALE, float(y) / FONT_SCALE };
                            
                            glm::vec4 shader_output = fragment_shader(uv, time_accumulator);
                            
                            // Multithreading ашиглаж байгаа ч pixel coordinate бүр давхцахгүй тул
                            // lock хийх шаардлагагүй, шууд бичих нь хурдан.
                            shs::Canvas::draw_pixel(*main_canvas, x, y, shs::Color{
                                (uint8_t)shader_output[0], 
                                (uint8_t)shader_output[1], 
                                (uint8_t)shader_output[2], 
                                (uint8_t)shader_output[3]
                            });
                        }
                    }
                });
                thread_pool.emplace_back(std::move(task));
            }
        }

        for (auto &thread : thread_pool)
        {
            thread.join();
        }

        // debug draw - одоо харагдах ёстой тул үүнийг авч хаяж болно
        // shs::Canvas::fill_random_pixel(*main_canvas, 40, 30, 60, 80);

        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);
        SDL_RenderCopy(renderer, screen_texture, NULL, NULL);
        SDL_RenderPresent(renderer);

        frame_counter++;
        Uint32 delta_frame_time  = SDL_GetTicks() - frame_start_ticks;
        frame_time_accumulator  += delta_frame_time/1000.0;
        time_accumulator        += delta_frame_time/1000.0;

        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay(frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0) {
            std::string window_title = "FPS : "+std::to_string(frame_counter);
            frame_time_accumulator   = 0.0;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }

    // ЗАСВАР: Санах ой чөлөөлөх дарааллыг засав
    delete main_canvas;      // Эхлээд устгана
    main_canvas = nullptr;   // Дараа нь null болгоно

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}