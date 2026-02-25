#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: units.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Рендерер болон Jolt физикийн интеграцийн хооронд хуваалцах үндсэн нэгжүүдийн стандартыг тодорхойлно.

    Үндсэн нэгжүүд:
        - Урт/Зай: метр (meter)
        - Жин: килограмм (kilogram)
        - Хугацаа: секунд (second)
        - Өнцөг: радиан (radian)
*/

#include <glm/glm.hpp>

namespace shs::units
{
    // SHS-ийн ажиллах үед ашиглагдах SI (Олон улсын нэгжийн систем) суурь нэгжүүд.
    inline constexpr float meter = 1.0f;
    inline constexpr float kilogram = 1.0f;
    inline constexpr float second = 1.0f;
    inline constexpr float radian = 1.0f;

    // Түгээмэл хэрэглэгдэх уламжлагдсан хэмжигдэхүүнүүд.
    inline constexpr float centimeter = 0.01f * meter;
    inline constexpr float millimeter = 0.001f * meter;
    inline constexpr float degree = 0.017453292519943295769f; // pi / 180

    // Дэлхийн татах хүчний хурдатгалын стандарт хэмжээ (м/с^2).
    inline constexpr float gravity_mps2 = 9.81f;

    // SHS ертөнцийн координатын системд +Y тэнхлэг дээшээ заадаг тул татах хүч нь -Y рүү чиглэнэ.
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

