#pragma once
/*
    shs-renderer-lib
    Shadow RT type (depth-only)

    ТАЙЛБАР:
    - Depth нь float buffer (0..1 эсвэл view-space z г.м) аль ч байж болно
    - Гол нь жинхэнэ type байх, void* хэрэглэхгүй
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

