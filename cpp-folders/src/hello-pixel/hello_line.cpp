#include <iostream>
#include <vector>
#include <tuple>
#include "tgaimage.h"

using namespace std;

const TGAColor white = {255, 255, 255, 255};
const TGAColor red   = {  0,   0, 255, 255};

void draw_line(int x0, int y0, int x1, int y1, TGAImage &image, TGAColor color) {
    for (float t=0.; t<1.; t+=.01) {
        int x = x0 + (x1-x0)*t;
        int y = y0 + (y1-y0)*t;
        image.set(x, y, color);
    }
}

int main() {

    cout<<"Hello Lines"<<endl;

    TGAImage image(100, 100, TGAImage::RGB);
    image.flip_vertically(); // left bottom origin

    std::vector<std::tuple<int, int, int, int>> lines = {
        {50, 10, 20, 90},
        {20, 90, 80, 90},
        {80, 90, 50, 10}
    };

    for (const auto& line : lines) {
        int x0, y0, x1, y1;
        std::tie(x0, y0, x1, y1) = line;
        std::cout << "Line: (" << x0 << ", " << y0 << ", " << x1 << ", " << y1 << ")" << std::endl;
        draw_line(x0, y0, x1, y1, image, red);

    }

    image.write_tga_file("output.tga");

    return 0;
}