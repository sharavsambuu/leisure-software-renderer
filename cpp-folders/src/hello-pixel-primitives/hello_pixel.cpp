#include "shs_renderer.hpp"

int main()
{

    std::cout << "Hello Pixel" << std::endl;

    int canvas_width = 100;
    int canvas_height = 100;

    shs::Canvas *random_canvas = new shs::Canvas(canvas_width, canvas_height);
    shs::Canvas *red_canvas = new shs::Canvas(canvas_width, canvas_height, shs::Pixel::red_pixel());
    shs::Canvas *black_canvas = new shs::Canvas(canvas_width, canvas_height, shs::Pixel::black_pixel());
    shs::Canvas white_canvas = shs::Canvas(canvas_width, canvas_height, shs::Pixel::white_pixel());

    black_canvas->draw_pixel(10, 10, shs::Pixel::red_pixel());
    shs::Canvas::draw_pixel(*black_canvas, 20, 20, shs::Pixel::green_pixel());
    shs::Canvas::draw_pixel(*black_canvas, 30, 30, shs::Pixel::white_pixel());
    black_canvas->draw_pixel(5, 60, shs::Pixel::blue_pixel());
    shs::Canvas::flip_vertically(*black_canvas); // origin at the left bottom corner of the canvas

    for (int i=0; i<50; i+=2) {
        if (i%3==1) {
            shs::Canvas::draw_pixel(white_canvas, i, i, shs::Pixel::blue_pixel());
        } 
        if (i%3==2)
        {
            shs::Canvas::draw_pixel(white_canvas, i, i, shs::Pixel::red_pixel());
        } else
        {
            white_canvas.draw_pixel(i, i, shs::Pixel::green_pixel());
        }
    }
    white_canvas.flip_vertically();

    random_canvas->save_png("hello_pixel_random_canvas.png");
    red_canvas->save_png("hello_pixel_red_canvas.png");
    black_canvas->save_png("hello_pixel_canvas_canvas.png");
    white_canvas.save_png("hello_pixel_white_canvas.png");

    return 0;
}