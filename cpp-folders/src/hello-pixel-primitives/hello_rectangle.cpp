#include "shs_renderer.hpp"

int main()
{

    std::cout << "Hello Pixel" << std::endl;

    int canvas_width  = 100;
    int canvas_height = 100;

    shs::Canvas *random_canvas = new shs::Canvas(canvas_width, canvas_height);
    shs::Canvas *white_canvas  = new shs::Canvas(canvas_width, canvas_height, shs::Color::white());
    shs::Canvas *red_canvas    = new shs::Canvas(canvas_width, canvas_height, shs::Color::red());
    shs::Canvas *black_canvas  = new shs::Canvas(canvas_width, canvas_height, shs::Color::black());

    black_canvas->draw_pixel(10, 10, shs::Color::red());
    shs::Canvas::draw_pixel(*black_canvas, 20, 20, shs::Color::green());
    shs::Canvas::draw_pixel(*black_canvas, 30, 30, shs::Color::white());
    black_canvas->draw_pixel(5, 60, shs::Color::blue());

    shs::Canvas::flip_horizontally(*black_canvas); // origin at the left bottom corner of the canvas

    random_canvas->save_png("random_canvas.png");
    white_canvas ->save_png("white_canvas.png" );
    red_canvas   ->save_png("red_canvas.png"   );
    black_canvas ->save_png("canvas_canvas.png");

    return 0;
}