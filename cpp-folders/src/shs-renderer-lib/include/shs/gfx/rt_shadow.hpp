#pragma once

/*
    SHS РЕНДЕРЕР САН

    ФАЙЛ: rt_shadow.hpp
    МОДУЛЬ: gfx
    ЗОРИЛГО: Энэ файл нь shs-renderer-lib-ийн gfx модульд хамаарах төрөл/функцийн
            интерфэйс эсвэл хэрэгжүүлэлтийг тодорхойлно.
*/

#include <cstdint>
#include <vector>
#include <algorithm>

namespace shs {

struct RT_ShadowDepth {
    int w = 0;
    int h = 0;
    std::vector<float> depth;

    RT_ShadowDepth() = default;
    RT_ShadowDepth(int W, int H) { resize(W, H); }

    inline void resize(int W, int H) {
        w = W; h = H;
        depth.assign((size_t)w * (size_t)h, 1.0f);
    }

    inline void clear(float v = 1.0f) {
        std::fill(depth.begin(), depth.end(), v);
    }

    inline float* data() { return depth.data(); }
    inline const float* data() const { return depth.data(); }

    inline float& at(int x, int y) { return depth[(size_t)y * (size_t)w + (size_t)x]; }
    inline const float& at(int x, int y) const { return depth[(size_t)y * (size_t)w + (size_t)x]; }
};

} // namespace shs
