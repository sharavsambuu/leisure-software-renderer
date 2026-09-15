#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: primitive_import.hpp
    МОДУЛЬ: resources
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн resources модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/


#include <string>

#include "shs/geometry/primitives.hpp"
#include "shs/geometry/primitives_builders.hpp"
#include "shs/resources/resource_registry.hpp"

namespace shs
{
    inline MeshAssetHandle import_plane_primitive(ResourceRegistry& reg, const PlaneDesc& d, const std::string& key = {})
    {
        return reg.add_mesh(make_plane(d), key);
    }

    inline MeshAssetHandle import_sphere_primitive(ResourceRegistry& reg, const SphereDesc& d, const std::string& key = {})
    {
        return reg.add_mesh(make_sphere(d), key);
    }

    inline MeshAssetHandle import_box_primitive(ResourceRegistry& reg, const BoxDesc& d, const std::string& key = {})
    {
        return reg.add_mesh(make_box(d), key);
    }

    inline MeshAssetHandle import_cone_primitive(ResourceRegistry& reg, const ConeDesc& d, const std::string& key = {})
    {
        return reg.add_mesh(make_cone(d), key);
    }
}

