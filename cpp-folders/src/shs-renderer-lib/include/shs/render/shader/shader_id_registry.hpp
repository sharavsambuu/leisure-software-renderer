#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: shader_id_registry.hpp
    МОДУЛЬ: shader
    ЗОРИЛГО: Open registered range-ийн shader-id registry (Constitution I §7 —
            No User Lock-In; arch/render_path_architecture.md §4 graduation
            req 6). `ShaderId` (shader_id.hpp) нь жижиг хаалттай builtin
            vocabulary; consumer/demo-ийн shader-ууд энэ registry-гээр орж,
            core-д гар хүрэхгүй.

    Энэ нь `PassIdRegistry`-ийн ГУРАВДАХ хэрэглэгч — ижил shape, өөр
    namespace. Offset тооцооллыг ХУВААЛЦАХ хуулиас авна
    (`shs/core/open_id_hash.hpp`), хуулбарлаагүй: механизм нэг, диапазон ба
    нэрийн хүснэгт namespace тус бүрийн өөрийн.

    id нь MINT-ORDER БИШ, CONTENT-ADDRESSED (`open_offset`): open id нь
    бүртгэгдсэн НЭР-ийн детерминит функц. Үүнээс гурван шинж гарна:

      1. Процесс болон translation unit хооронд тогтвортой — consumer shader
         -ийг нэрлэсэн хадгалагдсан recipe, replay log хүчинтэй үлдэнэ.
      2. Дарааллаас үл хамаарна — нэрүүдийг ямар ч дарааллаар intern хийсэн
         ижил id гарна, тиймээс бүртгэлийн дараалал төлөвлөгөөнд нэвтрэхгүй.
      3. Cross-registry aliasing байхгүй — `try_name(foreign_id)` нь зөвхөн
         тухайн id-г үүсгэсэн нэрийг буцаана, эс бөгөөс nullopt. (Mint-order
         id-ууд энэ шалгалтад унана: `base+0` нь "энд хамгийн түрүүнд
         бүртгэгдсэн нь" гэсэн үг тул өөр registry-д чимээгүй буруу shader
         руу заана.)

    Тииймээс цорын ганц алдааны горим нь collision бөгөөд тэр нь ЧАНГААР:
    хоёр ялгаатай нэр нэг slot-д буувал хоёр дахь `intern` nullopt буцаана,
    юу ч дарагдахгүй. Хадгалалт нь flat vector (ихэвчлэн < 100 consumer
    shader) тул registry нь copyable, comparable, хямд хэвээр — lookup үед
    хэшгүй, ambient global байхгүй (owner тус бүр нэг instance).
*/

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "shs/core/open_id_hash.hpp"
#include "shs/render/shader/shader_id.hpp"

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace render
    {
    class ShaderIdRegistry
    {
    public:
        // Хэдэн ялгаатай consumer-owned shader open range-д багтах вэ.
        static constexpr std::size_t capacity()
        {
            return static_cast<std::size_t>(kShaderIdOpenMax) -
                   static_cast<std::size_t>(kShaderIdOpenBase) + 1u;
        }

        // Нэрийн open range доторх детерминит offset. Хуваалцах хууль
        // (`core::open_id_offset`), тиймээс ижил нэр pass болон shader талд
        // ижил offset өгнө — санамсаргүй биш, нэг л хууль байх ёстой.
        static constexpr uint16_t open_offset(std::string_view name)
        {
            return core::open_id_offset(name, capacity());
        }

        // Нэрийг typed shader id болгоно. Total (throw байхгүй, хагас
        // state байхгүй):
        //   - builtin нэр ("blinn_phong") нь өөрийн builtin id болно —
        //     consumer хэзээ ч core shader-ийг сүүдэрлэж чадахгүй;
        //   - давтагдсан нэр => ижил id (idempotent);
        //   - шинэ нэр => open id болж бүртгэгдэнэ;
        //   - хоосон нэр, `kShaderIdOpenSpelling` эсвэл collision => nullopt.
        std::optional<ShaderId> intern(std::string_view name)
        {
            if (name.empty() || name == kShaderIdOpenSpelling) return std::nullopt;

            const ShaderId builtin = parse_shader_id(name);
            if (shader_id_is_builtin(builtin)) return builtin;

            const uint16_t offset = open_offset(name);
            const ShaderId id = static_cast<ShaderId>(static_cast<uint16_t>(kShaderIdOpenBase + offset));

            for (const auto& slot : slots_)
            {
                if (slot.first == offset)
                {
                    // Ижил нэр => ижил id. Өөр нэр => collision: татгалзана.
                    return slot.second == name ? std::optional<ShaderId>(id) : std::nullopt;
                }
            }

            slots_.emplace_back(offset, std::string(name));
            return id;
        }

        // Typed id-ийн нэр: builtin хүснэгт, эсвэл тухайн id-г үүсгэсэн нэр
        // (энэ registry бүртгэсэн). Unknown, диапазонаас гадуур id, мөн энэ
        // registry-д бүртгэгдээгүй id-д nullopt — чухал нь, foreign id нь
        // зөвхөн miss болно, өөр shader руу alias болохгүй.
        std::optional<std::string_view> try_name(ShaderId id) const
        {
            if (shader_id_is_builtin(id))
            {
                const std::string_view n = shader_id_builtin_name(id);
                return n.empty() ? std::nullopt : std::optional<std::string_view>(n);
            }
            if (!shader_id_is_open(id)) return std::nullopt;

            const uint16_t offset = static_cast<uint16_t>(
                static_cast<uint16_t>(id) - static_cast<uint16_t>(kShaderIdOpenBase));
            for (const auto& slot : slots_)
            {
                if (slot.first == offset) return std::string_view(slot.second);
            }
            return std::nullopt;
        }

        // builtin id, эсвэл энэ registry бүртгэсэн open id-д true.
        bool contains(ShaderId id) const { return try_name(id).has_value(); }

        // Зөвхөн энэ registry бүртгэсэн open id-д true.
        bool is_open(ShaderId id) const
        {
            return shader_id_is_open(id) && contains(id);
        }

        // Бүртгэлийн дарааллаар (детерминит; id-ууд нь өөрсдөө дарааллаас
        // хамааралгүй, `open_offset`-ыг үз).
        const std::vector<std::pair<uint16_t, std::string>>& registered() const { return slots_; }

        std::size_t open_count() const { return slots_.size(); }
        bool empty() const { return slots_.empty(); }
        void clear() { slots_.clear(); }

        // Value semantics, дарааллаас хамааралгүй: хоёр registry нь ижил
        // нэрүүдийг бүртгэсэн бол (тиймээс ижил content-addressed id-тай)
        // тэнцүү.
        bool operator==(const ShaderIdRegistry& other) const
        {
            if (slots_.size() != other.slots_.size()) return false;
            for (const auto& slot : slots_)
            {
                const ShaderId id =
                    static_cast<ShaderId>(static_cast<uint16_t>(kShaderIdOpenBase + slot.first));
                const std::optional<std::string_view> in_other = other.try_name(id);
                if (!in_other.has_value() || *in_other != std::string_view(slot.second)) return false;
            }
            return true;
        }

    private:
        // (open-range offset, бүртгэгдсэн нэр). Flat ба linear: ихэвчлэн
        // цөөн entry, registry-г copyable, comparable байлгана.
        std::vector<std::pair<uint16_t, std::string>> slots_{};
    };

    } // inline namespace render
} // namespace shs
