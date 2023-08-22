#include "shs_renderer.hpp"

/*
*
*
*/


int main()
{

    std::cout << "Hello Wireframe" << std::endl;

    int canvas_width  = 100;
    int canvas_height = 100;
    shs::Canvas canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::black());

    shs::Canvas::flip_horizontally(canvas); // origin at the left bottom corner of the canvas


    return 0;
}