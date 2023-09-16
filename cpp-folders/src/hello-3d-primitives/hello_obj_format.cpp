#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "shs_renderer.hpp"


/*
 * Reading triangles from 3D wavefront obj file
 */


int main()
{

    std::cout << "Hello 3D obj file format." << std::endl;

    std::vector<shs::RawTriangle> triangles;
    triangles = shs::Util::Obj3DFile::read_triangles("./obj/teapot/stanford-teapot.rawobj");

    for (auto &triangle : triangles)
    {
        std::cout << "v1=["
                  << triangle.v1.x << ", " << triangle.v1.y << ", " << triangle.v1.z << "], "
                  << "v2=["
                  << triangle.v2.x << ", " << triangle.v2.y << ", " << triangle.v2.z << "], "
                  << "v3=["
                  << triangle.v3.x << ", " << triangle.v3.y << ", " << triangle.v3.z << "]"
                  << std::endl;
    }


    std::cout << "done." << std::endl;

    return EXIT_SUCCESS;
}