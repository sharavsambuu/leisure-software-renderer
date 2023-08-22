#include "shs_renderer.hpp"

/*
*
*
*/


int main()
{

    std::cout << "Hello Line" << std::endl;

    int canvas_width  = 100;
    int canvas_height = 100;
    shs::Canvas first_line_canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::white());
    shs::Canvas second_line_canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::white());
    shs::Canvas third_line_canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::white());
    shs::Canvas fourth_line_canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::white());
    shs::Canvas final_line_canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::white());

    // Drawing first line
    shs::Canvas::draw_line_first(first_line_canvas, 13, 20, 80, 40, shs::Color::red());
    shs::Canvas::flip_horizontally(first_line_canvas); // origin at the left bottom corner of the canvas

    // Drawing second line
    shs::Canvas::draw_line_second(second_line_canvas, 20, 13, 40, 80, shs::Color::red());
    shs::Canvas::flip_horizontally(second_line_canvas); // origin at the left bottom corner of the canvas

    // Drawing third line
    shs::Canvas::draw_line_third(third_line_canvas, 13, 20, 80, 40, shs::Color::green());
    shs::Canvas::draw_line_third(third_line_canvas, 20, 13, 40, 80, shs::Color::red());
    shs::Canvas::flip_horizontally(third_line_canvas); // origin at the left bottom corner of the canvas

    // Drawing fourth line
    shs::Canvas::draw_line_fourth(fourth_line_canvas, 13, 20, 80, 40, shs::Color::green());
    shs::Canvas::draw_line_fourth(fourth_line_canvas, 20, 13, 40, 80, shs::Color::red());
    shs::Canvas::flip_horizontally(fourth_line_canvas); // origin at the left bottom corner of the canvas

    // Drawing final line
    shs::Canvas::draw_line(final_line_canvas, 13, 20, 80, 40, shs::Color::green());
    shs::Canvas::draw_line(final_line_canvas, 20, 13, 40, 80, shs::Color::red());
    shs::Canvas::flip_horizontally(final_line_canvas); // origin at the left bottom corner of the canvas


    return 0;
}