#include "shs_renderer.hpp"

/*
*
* Following line drawing tutorial from https://github.com/ssloy/tinyrenderer/wiki/Lesson-1:-Bresenham%E2%80%99s-Line-Drawing-Algorithm
*
*/


int main()
{

    std::cout << "Hello Line" << std::endl;

    int canvas_width  = 100;
    int canvas_height = 100;

    shs::Canvas final_line_canvas = shs::Canvas(canvas_width, canvas_height, shs::Pixel::white_pixel());

    shs::Canvas::draw_line(final_line_canvas, 13, 20, 80, 40, shs::Pixel::green_pixel());
    shs::Canvas::draw_line(final_line_canvas, 20, 13, 40, 80, shs::Pixel::red_pixel());
    shs::Canvas::flip_vertically(final_line_canvas); // origin at the left bottom corner of the canvas

    final_line_canvas.save_png("hello_line_final_line_canvas.png" );

    return 0;
}