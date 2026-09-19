#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: open_id_hash.hpp
    МОДУЛЬ: core
    ЗОРИЛГО: Open-id namespace-уудын ХУВААЛЦАХ content-addressing хууль
            (Constitution I §7 — No User Lock-In). `PassId` (2026-09-18) ба
            `ShaderId` (2026-09-18) хоёулаа нэрийг ижил offsets-оор id болгоно:
            id нь НЭР-ийн цэвэр функц тул процесс, translation unit, бүртгэлийн
            дарааллаас үл хамаарна.

    Яагаад нэг газар вэ: хоёр registry тус тусдаа хувилбар хадгалбал нэг нь
    drift хийхэд нэг нэр өөр id болж, яг тэр replay/determinism шинж (saved
    recipe, replay log, barrier table-ууд хүчинтэй үлдэх) чимээгүй эвдэрнэ.
    Rule of two: хоёр дахь (гурав дахь) хэрэглэгч ирэхэд механизмыг дээшлүүлж
    нэгтгэх — хуулах биш.

    Хамрах хүрээ: зөвхөн offsет тооцоолол. id-ийн диапазон, нэрийн хүснэгт,
    бүртгэлийн бодлого нь namespace бүрийн өөрийн (pass_id.hpp / shader_id.hpp)
    хэвээр үлдэнэ.
*/

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace core
    {
    // FNV-1a 32 with a final mix, folded into [0, range_capacity).
    //
    // Цэвэр функц: state байхгүй, iteration order байхгүй, ambient seed
    // байхгүй. Тиймээс нэг нэр хаана ч, хэзээ ч, ямар дарааллаар бүртгэгдсэн ч
    // нэг offset өгнө.
    //
    // `range_capacity` тэгээс их байх ЁСТОЙ (тэг бол modulo нь undefined).
    // Хоёр registry хоёулаа open диапазонаасаа гаргаж авдаг тул ≥1 гаран
    // (PassIdRegistry::capacity() == 64511, ShaderIdRegistry::capacity() — мөн).
    [[nodiscard]] constexpr std::uint16_t open_id_offset(
        std::string_view name, std::size_t range_capacity)
    {
        std::uint32_t h = 2166136261u;
        for (const char c : name)
        {
            h ^= static_cast<std::uint32_t>(static_cast<unsigned char>(c));
            h *= 16777619u;
        }
        h ^= h >> 16;
        h *= 0x7feb352du;
        h ^= h >> 15;
        return static_cast<std::uint16_t>(h % static_cast<std::uint32_t>(range_capacity));
    }
    } // inline namespace core
} // namespace shs
