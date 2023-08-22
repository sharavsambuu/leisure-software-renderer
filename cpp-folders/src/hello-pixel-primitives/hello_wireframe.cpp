#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include "shs_renderer.hpp"

/*
 *
 *
 */

int main()
{

    std::cout << "Hello Wireframe" << std::endl;

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
                    float x0 = prev_vertex.x;
                    float y0 = prev_vertex.y;
                    float x1 = vertex.x;
                    float y1 = vertex.y;

                    std::cout << "Line: (" << x0 << ", " << y0 << "), (" << x1 << ", " << y1 << ")" << std::endl;
                }

                prev_vertex = vertex;
            }
        }
    }

    std::cout << "end." << std::endl;

    return EXIT_SUCCESS;
}