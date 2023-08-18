#include <iostream>
#include "tgaimage.h"

using namespace std;

const TGAColor white = {255, 255, 255, 255};
const TGAColor red   = {  0,   0, 255, 255};

int main() {

    cout<<"Hello Pixel"<<endl;

    TGAImage image(100, 100, TGAImage::RGB);
    image.set(52, 41, red);
    image.flip_vertically(); // left bottom origin
    image.write_tga_file("output.tga");

    return 0;
}