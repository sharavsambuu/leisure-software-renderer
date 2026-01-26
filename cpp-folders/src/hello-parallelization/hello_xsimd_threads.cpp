/*
    xsimd + std::thread олон цөм стресс тест

    Kernel:
      x = x * a + y * b
      y = y * c - x * d
      x = 1 / (1 + abs(x)) 

    - Thread-үүд хуваан ажиллуулна
    - Дотоод давталтдаа SIMD хэрэглэнэ (xsimd::batch<float>)
    - compute+memory стресс тест хийхийн тулд ITER удаа давтана
*/

#include <xsimd/xsimd.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

static inline float fast_nonlinear(float v) {
    v = std::fabs(v);
    return 1.0f / (1.0f + v);
}

static void kernel_chunk(float* x, float* y, size_t begin, size_t end, int iters)
{
    using batch = xsimd::batch<float>;
    constexpr size_t L = batch::size;

    const batch A(1.001f);
    const batch B(0.999f);
    const batch C(1.0005f);
    const batch D(0.9985f);
    const batch ONE(1.0f);

    size_t i = begin;

    for (int k = 0; k < iters; ++k) {
        i = begin;

        // SIMD бие
        for (; i + L <= end; i += L) {
            batch vx = batch::load_unaligned(x + i);
            batch vy = batch::load_unaligned(y + i);

            batch nx = vx * A + vy * B;
            batch ny = vy * C - nx * D;

            // nx = 1 / (1 + abs(nx))
            batch ax = xsimd::abs(nx);
            nx = ONE / (ONE + ax);

            nx.store_unaligned(x + i);
            ny.store_unaligned(y + i);
        }

        // tail скялар
        for (; i < end; ++i) {
            float nx = x[i] * 1.001f + y[i] * 0.999f;
            float ny = y[i] * 1.0005f - nx * 0.9985f;
            nx = fast_nonlinear(nx);
            x[i] = nx;
            y[i] = ny;
        }
    }
}

int main()
{
    const size_t N    = 1u << 24;   // 16,777,216 floats (массив бүрт ~64MB)
    const int    ITER = 4000;       // нэмж болно

    std::vector<float> x(N), y(N);

    // init
    for (size_t i = 0; i < N; ++i) {
        x[i] = float((i % 1024) - 512) * 0.001f;
        y[i] = float((i % 2048) - 1024) * 0.0007f;
    }

    const unsigned hw      = std::max(1u, std::thread::hardware_concurrency());
    const unsigned threads = hw; // Бүх CPU цөмийг хэрэглэх

    const size_t chunk     = (N + threads - 1) / threads;

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> pool;
    pool.reserve(threads);

    for (unsigned t = 0; t < threads; ++t) {
        const size_t begin = (size_t)t * chunk;
        const size_t end   = std::min(N, begin + chunk);
        if (begin >= end) break;

        pool.emplace_back([&, begin, end] {
            kernel_chunk(x.data(), y.data(), begin, end, ITER);
        });
    }

    for (auto& th : pool) th.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // checksum 
    double sumx = 0.0, sumy = 0.0;
    for (size_t i = 0; i < N; i += 4096) { // sample
        sumx += x[i];
        sumy += y[i];
    }

    std::printf("N=%zu ITER=%d threads=%u lanes=%zu time=%.2f ms checksum=(%.6f, %.6f)\n",
        N, ITER, (unsigned)pool.size(), xsimd::batch<float>::size, ms, sumx, sumy);

    return 0;
}
