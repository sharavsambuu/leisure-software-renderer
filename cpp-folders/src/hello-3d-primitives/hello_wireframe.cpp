#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "shs_renderer.hpp"

/*
 * Rendering 3D object on the 2D surface by line by line.
 */

int main()
{

    std::cout << "Hello Wireframe" << std::endl;

    int canvas_width  = 600;
    int canvas_height = 600;
    shs::Canvas canvas = shs::Canvas(canvas_width, canvas_height, shs::Color::black());

    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile("./obj/monkey/monkey.rawobj", aiProcess_Triangulate);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cerr << "Error loading OBJ file: " << importer.GetErrorString() << std::endl;
        return 1;
    }

    aiVector3D prev_vertex;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[i];
        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++)
            {
                unsigned int vertex_index = face.mIndices[k];
                aiVector3D vertex = mesh->mVertices[vertex_index];
                if (k > 0)
                {
                    int x0 = (prev_vertex.x + 1.0) * canvas_width / 2.0;
                    int y0 = (prev_vertex.y + 1.0) * canvas_height / 2.0;
                    int x1 = (vertex.x + 1.0) * canvas_width / 2.0;
                    int y1 = (vertex.y + 1.0) * canvas_height / 2.0;
                    if (
                        (x0 > 0 && x0 < canvas_width) &&
                        (y0 > 0 && y0 < canvas_height) &&
                        (x1 > 0 && x1 < canvas_width) &&
                        (y1 > 0 && y1 < canvas_height))
                    {
                        //std::cout << x0 << " " << y0 << " " << x1 << " " << y1 << std::endl;
                        shs::Canvas::draw_line(canvas, x0, y0, x1, y1, shs::Color::green());
                    }
                }
                prev_vertex = vertex;
            }
        }
    }
    canvas.save_png("hello_wireframe_monkeyobj_canvas.png" );

    std::cout << "done." << std::endl;

    return EXIT_SUCCESS;
}