#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <string>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <vector>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480

#define CANVAS_WIDTH      320
#define CANVAS_HEIGHT     240

struct Sprite
{
    float   x , y;
    float   vx, vy;
    int     w , h;
    uint8_t opacity;
};

static float randf(std::mt19937 &rng, float a, float b)
{
    std::uniform_real_distribution<float> dist(a, b);
    return dist(rng);
}

static int randi(std::mt19937 &rng, int a, int b)
{
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

int main()
{
    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    SDL_Init(SDL_INIT_VIDEO);
    IMG_Init(IMG_INIT_PNG | IMG_INIT_JPG);

    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    std::string image_path = "./images/box_texture.jpg";
    shs::Texture2D tex = shs::load_texture_sdl_image(image_path, true);
    if (!tex.valid()) {
        std::cout << "Texture load failed..\n";
    }

    std::mt19937 rng((uint32_t)std::time(nullptr));

    // өөр өөр scale-тэй sprite-үүд
    int sprite_count = randi(rng, 3, 4);
    std::vector<Sprite> sprites;
    sprites.reserve((size_t)sprite_count);

    for (int i = 0; i < sprite_count; i++)
    {
        // scale-уудыг тус тусад нь random утгаар цэнэглэх
        float sx = randf(rng, 0.12f, 0.55f);
        float sy = randf(rng, 0.12f, 0.55f);

        Sprite s;

        s.w = (int)std::lround((float)tex.w * sx);
        s.h = (int)std::lround((float)tex.h * sy);

        // min хэмжээ
        if (s.w < 12) s.w = 12;
        if (s.h < 12) s.h = 12;

        // canvas-ын тодорхой хувиар max хэмжээ тогтоох
        int max_w = (int)(CANVAS_WIDTH  * 0.45f);
        int max_h = (int)(CANVAS_HEIGHT * 0.45f);
        if (s.w > max_w) s.w = max_w;
        if (s.h > max_h) s.h = max_h;

        s.x = randf(rng, 0.0f, (float)(CANVAS_WIDTH  - s.w));
        s.y = randf(rng, 0.0f, (float)(CANVAS_HEIGHT - s.h));

        // px/sec
        s.vx = randf(rng, -140.0f, 140.0f);
        s.vy = randf(rng, -140.0f, 140.0f);
        if (std::abs(s.vx) < 30.0f) s.vx = (s.vx < 0 ? -60.0f : 60.0f);
        if (std::abs(s.vy) < 30.0f) s.vy = (s.vy < 0 ? -60.0f : 60.0f);

        s.opacity = (uint8_t)randi(rng, 180, 255);

        sprites.push_back(s);
    }

    bool exit = false;
    SDL_Event event_data;

    int   frame_delay            = 1000 / FRAMES_PER_SECOND;
    float frame_time_accumulator = 0.0f;
    int   frame_counter          = 0;

    Uint32 last_ticks = SDL_GetTicks();

    while (!exit)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();

        while (SDL_PollEvent(&event_data))
        {
            switch (event_data.type)
            {
            case SDL_QUIT:
                exit = true;
                break;
            case SDL_KEYDOWN:
                if (event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;
                break;
            }
        }

        // dt
        Uint32 now_ticks = SDL_GetTicks();
        float dt = (now_ticks - last_ticks) / 1000.0f;
        last_ticks = now_ticks;
        if (dt > 0.05f) dt = 0.05f;

        // move + bounce (canvas bounds)
        for (auto &s : sprites)
        {
            s.x += s.vx * dt;
            s.y += s.vy * dt;

            float minx = 0.0f;
            float miny = 0.0f;
            float maxx = (float)(CANVAS_WIDTH  - s.w);
            float maxy = (float)(CANVAS_HEIGHT - s.h);

            if (s.x < minx) { s.x = minx; s.vx = -s.vx; }
            if (s.x > maxx) { s.x = maxx; s.vx = -s.vx; }
            if (s.y < miny) { s.y = miny; s.vy = -s.vy; }
            if (s.y > maxy) { s.y = maxy; s.vy = -s.vy; }
        }

        // render
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Pixel::black_pixel());

        // blit 
        for (const auto &s : sprites)
        {
            shs::image_blit(
                *main_canvas, tex,
                (int)std::lround(s.x), (int)std::lround(s.y),
                0, 0, -1, -1,               // src rect = full
                s.w, s.h,                   // dst size = scaled
                s.opacity,                  // opacity
                shs::Tex::BLEND_ALPHA,      // blend
                shs::Tex::FILTER_NEAREST    // filter (MVP)
            );
        }

        // present
        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);

        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

        // fps cap + title
        frame_counter++;
        Uint32 delta_frame_time = SDL_GetTicks() - frame_start_ticks;

        frame_time_accumulator += delta_frame_time / 1000.0f;
        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay((Uint32)frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "Blit Bounce Demo | FPS : " + std::to_string(frame_counter);
            frame_time_accumulator   = 0.0f;
            frame_counter            = 0;
            SDL_SetWindowTitle(window, window_title.c_str());
        }
    }

    delete main_canvas;

    SDL_DestroyTexture(screen_texture);
    SDL_FreeSurface(main_sdlsurface);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);

    IMG_Quit();
    SDL_Quit();

    return 0;
}
