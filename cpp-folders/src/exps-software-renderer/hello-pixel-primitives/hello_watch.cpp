#include <SDL2/SDL.h>
#include <string>
#include <cmath>
#include <ctime>
#include <chrono>
#include "shs_renderer.hpp"

#define FRAMES_PER_SECOND 60
#define WINDOW_WIDTH      640
#define WINDOW_HEIGHT     480

#define CANVAS_WIDTH      320
#define CANVAS_HEIGHT     240

static inline double deg2rad(double deg) { return deg * 3.14159265358979323846 / 180.0; }

// Canvas space дээр (0° = 12 цаг, цагийн зүүний чиглэлээр эргэнэ)
static inline void angle_to_dir(double angle_deg, double &dx, double &dy)
{
    // 0° = +Y (дээш), CW
    double a = deg2rad(90.0 - angle_deg);
    dx = std::cos(a);
    dy = std::sin(a);
}

static void draw_hand(shs::Canvas &canvas, int cx, int cy, double angle_deg, int len, shs::Color p)
{
    double dx, dy;
    angle_to_dir(angle_deg, dx, dy);
    int x1 = cx + (int)std::lround(dx * (double)len);
    int y1 = cy + (int)std::lround(dy * (double)len);
    shs::Canvas::draw_line(canvas, cx, cy, x1, y1, p);
}

static void draw_tick(shs::Canvas &canvas, int cx, int cy, double angle_deg, int r0, int r1, shs::Color p)
{
    double dx, dy;
    angle_to_dir(angle_deg, dx, dy);
    int x0 = cx + (int)std::lround(dx * (double)r0);
    int y0 = cy + (int)std::lround(dy * (double)r0);
    int x1 = cx + (int)std::lround(dx * (double)r1);
    int y1 = cy + (int)std::lround(dy * (double)r1);
    shs::Canvas::draw_line(canvas, x0, y0, x1, y1, p);
}

int main(int argc, char* argv[])
{
    SDL_Window   *window   = nullptr;
    SDL_Renderer *renderer = nullptr;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(WINDOW_WIDTH, WINDOW_HEIGHT, 0, &window, &renderer);
    SDL_RenderSetScale(renderer, 1, 1);

    shs::Canvas *main_canvas     = new shs::Canvas(CANVAS_WIDTH, CANVAS_HEIGHT);
    SDL_Surface *main_sdlsurface = main_canvas->create_sdl_surface();
    SDL_Texture *screen_texture  = SDL_CreateTextureFromSurface(renderer, main_sdlsurface);

    bool exit = false;
    SDL_Event event_data;

    int   frame_delay            = 1000 / FRAMES_PER_SECOND;
    float frame_time_accumulator = 0.0f;
    int   frame_counter          = 0;

    const int cx = CANVAS_WIDTH / 2;
    const int cy = CANVAS_HEIGHT / 2;
    const int R  = (CANVAS_HEIGHT < CANVAS_WIDTH ? CANVAS_HEIGHT : CANVAS_WIDTH) / 2 - 10;

    shs::Color yellow(255, 220, 40, 255);
    shs::Color red   = shs::Color::red();
    shs::Color green = shs::Color::green();
    shs::Color blue  = shs::Color::blue();
    shs::Color white = shs::Color::white();

    while (!exit)
    {
        Uint32 frame_start_ticks = SDL_GetTicks();

        while (SDL_PollEvent(&event_data))
        {
            switch (event_data.type)
            {
            case SDL_QUIT: exit = true; break;
            case SDL_KEYDOWN:
                if (event_data.key.keysym.sym == SDLK_ESCAPE) exit = true;
                break;
            }
        }

        auto now = std::chrono::system_clock::now();
        auto ms_since_epoch = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

        std::time_t t = (std::time_t)(ms_since_epoch / 1000);
        std::tm local_tm = *std::localtime(&t);

        double ms   = (double)(ms_since_epoch % 1000);
        double sec  = (double)local_tm.tm_sec + ms / 1000.0;
        double min  = (double)local_tm.tm_min + sec / 60.0;
        double hour = (double)(local_tm.tm_hour % 12) + min / 60.0;

        double sec_deg  = sec  * 6.0;    // 360/60
        double min_deg  = min  * 6.0;
        double hour_deg = hour * 30.0;   // 360/12

        shs::Canvas::fill_pixel(*main_canvas, 0, 0, CANVAS_WIDTH, CANVAS_HEIGHT, shs::Color::black());

        shs::Canvas::draw_circle_poly(*main_canvas, cx, cy, R, 180, yellow);

        for (int i = 0; i < 60; i++) {
            double a = (double)i * 6.0;
            bool hour_tick = (i % 5 == 0);

            int r1 = R - 2;
            int r0 = hour_tick ? (R - 16) : (R - 9);

            draw_tick(*main_canvas, cx, cy, a, r0, r1, yellow);
        }

        draw_hand(*main_canvas, cx, cy, hour_deg, (int)(R * 0.55), red);
        draw_hand(*main_canvas, cx, cy, min_deg,  (int)(R * 0.78), green);
        draw_hand(*main_canvas, cx, cy, sec_deg,  (int)(R * 0.90), blue);

        shs::Canvas::draw_circle_poly(*main_canvas, cx, cy, 3, 24, white);

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);

        shs::Canvas::copy_to_SDLSurface(main_sdlsurface, main_canvas);
        SDL_UpdateTexture(screen_texture, NULL, main_sdlsurface->pixels, main_sdlsurface->pitch);

        SDL_Rect destination_rect{0, 0, WINDOW_WIDTH, WINDOW_HEIGHT};
        SDL_RenderCopy(renderer, screen_texture, NULL, &destination_rect);
        SDL_RenderPresent(renderer);

        frame_counter++;
        Uint32 delta_frame_time = SDL_GetTicks() - frame_start_ticks;

        frame_time_accumulator += delta_frame_time / 1000.0f;
        if (delta_frame_time < (Uint32)frame_delay) {
            SDL_Delay((Uint32)frame_delay - delta_frame_time);
        }
        if (frame_time_accumulator >= 1.0f) {
            std::string window_title = "Analog Clock | FPS : " + std::to_string(frame_counter);
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
    SDL_Quit();

    return 0;
}
