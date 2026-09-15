#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: mesh_loader_assimp.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>
#include <vector>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "shs/resources/mesh.hpp"

namespace shs
{
    struct MeshLoadOptions
    {
        bool triangulate = true;
        bool generate_normals = true;
        bool join_identical_vertices = true;
        bool flip_uvs = false;
    };

    inline unsigned int to_assimp_flags(const MeshLoadOptions& opt)
    {
        unsigned int flags = 0;
        if (opt.triangulate) flags |= aiProcess_Triangulate;
        if (opt.generate_normals) flags |= aiProcess_GenSmoothNormals;
        if (opt.join_identical_vertices) flags |= aiProcess_JoinIdenticalVertices;
        if (opt.flip_uvs) flags |= aiProcess_FlipUVs;
        return flags;
    }

    inline std::vector<MeshData> load_meshes_assimp(const std::string& path, const MeshLoadOptions& opt = {})
    {
        std::vector<MeshData> out{};

        Assimp::Importer importer;
        const aiScene* scene = importer.ReadFile(path.c_str(), to_assimp_flags(opt));
        if (!scene || (scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE) || !scene->mRootNode) return out;

        out.reserve(scene->mNumMeshes);
        for (unsigned int mi = 0; mi < scene->mNumMeshes; ++mi)
        {
            const aiMesh* m = scene->mMeshes[mi];
            if (!m) continue;

            MeshData mesh{};
            mesh.source_path = path;
            mesh.positions.reserve(m->mNumVertices);
            mesh.normals.reserve(m->mNumVertices);
            mesh.uvs.reserve(m->mNumVertices);

            for (unsigned int vi = 0; vi < m->mNumVertices; ++vi)
            {
                const aiVector3D p = m->mVertices[vi];
                mesh.positions.push_back(glm::vec3(p.x, p.y, p.z));

                if (m->HasNormals())
                {
                    const aiVector3D n = m->mNormals[vi];
                    mesh.normals.push_back(glm::vec3(n.x, n.y, n.z));
                }
                else
                {
                    mesh.normals.push_back(glm::vec3(0.0f, 1.0f, 0.0f));
                }

                if (m->HasTextureCoords(0))
                {
                    const aiVector3D uv = m->mTextureCoords[0][vi];
                    mesh.uvs.push_back(glm::vec2(uv.x, uv.y));
                }
                else
                {
                    mesh.uvs.push_back(glm::vec2(0.0f));
                }
            }

            for (unsigned int fi = 0; fi < m->mNumFaces; ++fi)
            {
                const aiFace& face = m->mFaces[fi];
                if (face.mNumIndices != 3) continue;
                mesh.indices.push_back((uint32_t)face.mIndices[0]);
                mesh.indices.push_back((uint32_t)face.mIndices[1]);
                mesh.indices.push_back((uint32_t)face.mIndices[2]);
            }

            if (!mesh.empty()) out.push_back(std::move(mesh));
        }

        return out;
    }

    inline MeshData load_mesh_assimp_first(const std::string& path, const MeshLoadOptions& opt = {})
    {
        const auto meshes = load_meshes_assimp(path, opt);
        if (meshes.empty()) return MeshData{};
        return meshes.front();
    }
}

