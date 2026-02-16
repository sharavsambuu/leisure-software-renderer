#pragma once

/*
    SHS RENDERER SAN

    FILE: units.hpp
    MODULE: core
    PURPOSE: Canonical unit conventions shared across renderer and Jolt integration.

    Base convention:
        - Distance: meter
        - Mass: kilogram
        - Time: second
        - Angle: radian
*/

#include <glm/glm.hpp>

namespace shs::units
{
    // Base SI units used by SHS runtime conventions.
    inline constexpr float meter = 1.0f;
    inline constexpr float kilogram = 1.0f;
    inline constexpr float second = 1.0f;
    inline constexpr float radian = 1.0f;

    // Common derived scales.
    inline constexpr float centimeter = 0.01f * meter;
    inline constexpr float millimeter = 0.001f * meter;
    inline constexpr float degree = 0.017453292519943295769f; // pi / 180

    // Standard Earth gravity magnitude in m/s^2.
    inline constexpr float gravity_mps2 = 9.81f;

    // SHS world convention is +Y up, so gravity points to -Y.
    inline constexpr glm::vec3 gravity_world_y_down()
    {
        return glm::vec3(0.0f, -gravity_mps2, 0.0f);
    }

    inline constexpr float meters_from_centimeters(float value_cm)
    {
        return value_cm * centimeter;
    }

    inline constexpr float meters_from_millimeters(float value_mm)
    {
        return value_mm * millimeter;
    }

    inline constexpr float radians_from_degrees(float value_deg)
    {
        return value_deg * degree;
    }

    inline constexpr float degrees_from_radians(float value_rad)
    {
        return value_rad / degree;
    }

    inline constexpr float mps_from_kph(float value_kph)
    {
        return value_kph * (1000.0f / 3600.0f);
    }

    inline constexpr float kph_from_mps(float value_mps)
    {
        return value_mps * (3600.0f / 1000.0f);
    }
}

