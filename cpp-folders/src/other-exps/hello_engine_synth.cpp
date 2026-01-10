/*
  SDL2 AUDIO + VISUALIZER
  SUPER CAR 6-TRANSITION SHIFT + ENGINE START + IDLE LOCK + W/S PLAY MANUAL

  - Програм эхлэхэд engine start дуу сонсгоод idle дээр тогтворжино.
  - W-г жигд барьвал  : RPM өснө -> автоматаар upshift болно -> crack + thump сонсогдоно.
  - S-г жигд дарвал   : throttle бууна -> downshift зөөлөн.
  - Юу ч дарахгүй бол : speed өөрөө 0-рүү бууна, rpm idle дээр тогтоно.

  CONTROLS:
    W/S  : Throttle нэмэх/хасах
    ESC  : Quit
*/

#include <SDL2/SDL.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <atomic>
#include <cstring>

#if defined(__SSE__)
  #include <xmmintrin.h>
  #include <pmmintrin.h>
#endif

static constexpr float PI = 3.14159265358979323846f;

// Float <-> uint32 bitcast
static inline uint32_t f2u(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static inline float    u2f(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }

// Shared parameters (main -> audio)
static std::atomic<uint32_t> g_rpm_u        { f2u(900.0f) };
static std::atomic<uint32_t> g_throttle_u   { f2u(0.0f)   };
static std::atomic<uint32_t> g_load_u       { f2u(0.25f)  };
static std::atomic<uint32_t> g_torqueMul_u  { f2u(1.0f)   };
static std::atomic<uint32_t> g_shiftBurst_u { f2u(0.0f)   };

static inline float clamp01(float x) { return (x < 0.f) ? 0.f : (x > 1.f) ? 1.f : x; }
static inline float softclip(float x) {
    const float a = 1.5f;
    return x / (1.0f + a * std::fabs(x));
}

// 0..1 фаз wrap (x>=0 гэж үзнэ)
static inline float wrap01_pos(float x) {
    x -= (float)(int)x;
    return x;
}

// Fast sine LUT (phase01 нь 0..1 дотор байна гэж үзнэ)
struct SineLUT {
    static constexpr int N = 4096;
    float table[N + 1];

    SineLUT() {
        for (int i = 0; i <= N; ++i) {
            float t = float(i) / float(N);
            table[i] = std::sin(2.0f * PI * t);
        }
    }

    inline float sin01(float phase01) const {
        float x = phase01 * float(N);
        int   i = (int)x;
        float f = x - (float)i;
        return table[i] + (table[i + 1] - table[i]) * f;
    }
};

static SineLUT g_sin;

// RNG + helpers
struct XorShift32 {
    uint32_t s = 0x12345678u;
    inline uint32_t next_u32() {
        uint32_t x = s;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        s = x;
        return x;
    }
    inline float next_f01() { return (next_u32() >> 8) * (1.0f / 16777216.0f); }
    inline float next_f11() { return next_f01() * 2.0f - 1.0f; }
};

struct OnePoleLP {
    float y = 0.0f;
    float a = 0.1f;
    inline float process(float x) {
        y += a * (x - y);
        return y;
    }
};

struct Smooth {
    float v = 0.f;
    float a = 0.02f;
    inline float process(float target) { v += a * (target - v); return v; }
};

// Ring buffer (audio -> main) : бичигдсэн sample-уудын дараалсан уншилт
struct AudioRing {
    static constexpr uint32_t CAP = 1u << 16;
    float buf[CAP];

    alignas(64) std::atomic<uint32_t> w{0};
    alignas(64) uint32_t r = 0;

    inline void push(const float* x, int n) {
        uint32_t wi = w.load(std::memory_order_relaxed);
        for (int i = 0; i < n; ++i) buf[(wi + uint32_t(i)) & (CAP - 1)] = x[i];
        w.store(wi + uint32_t(n), std::memory_order_release);
    }

    inline uint32_t available() const {
        uint32_t wi = w.load(std::memory_order_acquire);
        return wi - r;
    }

    inline int read_block(float* out, int n) {
        uint32_t wi    = w.load(std::memory_order_acquire);
        uint32_t avail = wi - r;
        uint32_t take  = (avail < (uint32_t)n) ? avail : (uint32_t)n;

        for (uint32_t i = 0; i < take; ++i) out[i] = buf[(r + i) & (CAP - 1)];
        for (uint32_t i = take; i < (uint32_t)n; ++i) out[i] = 0.0f;

        r += take;
        return (int)take;
    }
};

static AudioRing g_ring;

// EngineSynth: Bugatti W16 (tuning)
struct EngineSynth {
    int   sampleRate     = 48000;

    float phase01        = 0.0f;
    float crackPhase01   = 0.0f;
    float thumpPhase01   = 0.0f;
    float starterPhase01 = 0.0f;

    float tStart         = 0.0f;
    float prevNoise      = 0.0f;

    int   cylinders      = 16;

    Smooth rpmSm, thrSm, loadSm;
    XorShift32 rng;
    OnePoleLP noiseLP;
    OnePoleLP mainLP;

    static constexpr int H = 8;
    static constexpr int LBINS = 32;
    float harmW[LBINS][H + 1];

    EngineSynth() {
        for (int b = 0; b < LBINS; ++b) {
            float load = float(b) / float(LBINS - 1);

            float bright = 0.18f + 0.70f * load;
            float exp = 1.25f + 2.60f * (1.0f - bright);

            harmW[b][0] = 0.0f;
            float s = 0.0f;
            for (int k = 1; k <= H; ++k) {
                float w = 1.0f / std::pow(float(k), exp);
                if (k >= 5) w *= 0.55f;
                if (k >= 7) w *= 0.60f;
                harmW[b][k] = w;
                s += w;
            }
            if (s > 1e-6f) {
                float inv = 1.0f / s;
                for (int k = 1; k <= H; ++k) harmW[b][k] *= inv;
            }
        }
    }

    inline float step(float rpmIn, float thrIn, float loadIn, float torqueMulIn, float shiftBurstIn) {
        float dt = 1.0f / float(sampleRate);
        tStart += dt;

        float rpm  = rpmSm.process(rpmIn);
        float thr  = thrSm.process(thrIn);
        float load = loadSm.process(loadIn);

        thr  = clamp01(thr);
        load = clamp01(load);

        float torqueMul = std::clamp(torqueMulIn, 0.0f, 1.15f);
        float burst     = clamp01(shiftBurstIn);

        float starter = 0.0f;
        float catchEnv = 0.0f;

        if (tStart < 0.55f) {
            float u = tStart / 0.55f;
            float wh = 160.0f + 120.0f * u;
            starterPhase01 = wrap01_pos(starterPhase01 + wh / float(sampleRate));
            starter = 0.13f * (1.0f - 0.35f * u) * g_sin.sin01(starterPhase01);
        }
        if (tStart >= 0.45f && tStart < 0.85f) {
            float u = (tStart - 0.45f) / 0.40f;
            catchEnv = clamp01(u);
        }

        float f0 = (rpm / 60.0f) * (0.5f * float(cylinders));
        f0 *= 0.50f;

        float jitter = (0.0010f + 0.0025f * load) * rng.next_f11();
        f0 *= (1.0f + jitter);

        phase01 = wrap01_pos(phase01 + f0 / float(sampleRate));

        int bin = int(load * float(LBINS - 1) + 0.5f);
        bin = std::clamp(bin, 0, LBINS - 1);

        float base = 0.0f;
        for (int k = 1; k <= H; ++k) {
            float phk = wrap01_pos(phase01 * float(k));
            base += harmW[bin][k] * g_sin.sin01(phk);
        }

        float n = rng.next_f11();
        noiseLP.a = 0.025f + 0.14f * thr;
        float ncol = noiseLP.process(n);

        float hp = ncol - prevNoise;
        prevNoise = ncol;

        float rpmNorm = std::fmin(rpm / 7000.0f, 1.0f);

        float drive = 0.24f + 0.76f * std::clamp(torqueMul, 0.0f, 1.0f);
        float hiss  = (0.006f + 0.040f * thr) * (0.25f + 0.75f * rpmNorm);

        float crackHz = 900.0f + 550.0f * thr + 350.0f * rpmNorm;
        crackPhase01  = wrap01_pos(crackPhase01 + crackHz / float(sampleRate));

        float crackTone  = g_sin.sin01(crackPhase01);
        float crackTone2 = g_sin.sin01(wrap01_pos(crackPhase01 * 1.55f));

        float thumpHz = 90.0f + 40.0f * thr + 20.0f * rpmNorm;
        thumpPhase01  = wrap01_pos(thumpPhase01 + thumpHz / float(sampleRate));
        float thump   = g_sin.sin01(thumpPhase01);

        float crack = 0.0f;
        crack += (0.060f * burst) * crackTone;
        crack += (0.030f * burst) * crackTone2;
        crack += (0.030f * burst) * hp;
        crack += (0.085f * burst) * thump;

        float amp = (0.050f + 0.30f * load + 0.15f * thr + 0.07f * rpmNorm) * drive;
        float noiseGain = (hiss + 0.020f * thr * (0.3f + 0.7f * load)) * drive;

        float x = amp * base + noiseGain * ncol + crack;

        if (tStart < 1.0f) {
            float blend = clamp01(catchEnv);
            x = (1.0f - blend) * (starter) + blend * x;
        }

        float grit = 0.62f + 1.05f * load;
        x = softclip(x * grit);

        mainLP.a = 0.022f + 0.28f * (0.25f + 0.75f * load);
        x = mainLP.process(x);

        return x;
    }
};

static void audio_cb(void* userdata, Uint8* stream, int len) {
    EngineSynth* synth = reinterpret_cast<EngineSynth*>(userdata);
    float* out  = reinterpret_cast<float*>(stream);
    int frames  = len / int(sizeof(float));

    float rpm   = u2f(g_rpm_u.load(std::memory_order_relaxed));
    float thr   = u2f(g_throttle_u.load(std::memory_order_relaxed));
    float load  = u2f(g_load_u.load(std::memory_order_relaxed));
    float tMul  = u2f(g_torqueMul_u.load(std::memory_order_relaxed));
    float burst = u2f(g_shiftBurst_u.load(std::memory_order_relaxed));

    for (int i = 0; i < frames; ++i) out[i] = synth->step(rpm, thr, load, tMul, burst);

    g_ring.push(out, frames);
}

// FFT (radix-2)
struct Complex { float re, im; };
static inline Complex c_add(Complex a, Complex b) { return {a.re + b.re, a.im + b.im}; }
static inline Complex c_sub(Complex a, Complex b) { return {a.re - b.re, a.im - b.im}; }
static inline Complex c_mul(Complex a, Complex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}

static void fft_radix2(std::vector<Complex>& a) {
    const int n = (int)a.size();

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        float ang = -2.0f * PI / float(len);
        Complex wlen{std::cos(ang), std::sin(ang)};
        for (int i = 0; i < n; i += len) {
            Complex w{1.0f, 0.0f};
            for (int j = 0; j < len / 2; ++j) {
                Complex u = a[i + j];
                Complex v = c_mul(a[i + j + len / 2], w);
                a[i + j] = c_add(u, v);
                a[i + j + len / 2] = c_sub(u, v);
                w = c_mul(w, wlen);
            }
        }
    }
}

// Visualizer
static void draw_wave(SDL_Renderer* r, const float* x, int n, int W, int H) {
    int mid = H / 2;
    float yscale = 0.45f * float(H);

    for (int px = 0; px < W - 1; ++px) {
        int i0 = (px * n) / W;
        int i1 = ((px + 1) * n) / W;
        int y0 = mid - (int)(x[i0] * yscale);
        int y1 = mid - (int)(x[i1] * yscale);
        SDL_RenderDrawLine(r, px, y0, px + 1, y1);
    }
}

struct SpectrumState {
    static constexpr int BARS = 48;

    std::vector<Complex> fftBuf;
    const float* hann = nullptr;

    float bars[BARS]{};
    float peaks[BARS]{};

    uint32_t lastFftMs = 0;
    uint32_t fftIntervalMs = 50;

    SpectrumState(int n, const float* hannWin) : fftBuf(n), hann(hannWin) {}

    void update_if_needed(const float* x, int n, uint32_t nowMs) {
        if (nowMs - lastFftMs < fftIntervalMs) {
            float peakFall = 0.012f;
            float barFall  = 0.030f;
            for (int i = 0; i < BARS; ++i) {
                bars[i]  = std::max(0.0f, bars[i]  - barFall);
                peaks[i] = std::max(0.0f, peaks[i] - peakFall);
                if (peaks[i] < bars[i]) peaks[i] = bars[i];
            }
            return;
        }
        lastFftMs = nowMs;

        for (int i = 0; i < n; ++i) fftBuf[i] = { x[i] * hann[i], 0.0f };
        fft_radix2(fftBuf);

        const int usable = n / 2;
        const float eps = 1e-9f;

        for (int b = 0; b < BARS; ++b) {
            float t0 = float(b) / float(BARS);
            float t1 = float(b + 1) / float(BARS);

            int k0 = (int)(std::pow(t0, 2.0f) * usable);
            int k1 = (int)(std::pow(t1, 2.0f) * usable);
            if (k1 <= k0) k1 = k0 + 1;
            if (k1 > usable) k1 = usable;

            float mag = 0.0f;
            for (int k = k0; k < k1; ++k) {
                float re = fftBuf[k].re, im = fftBuf[k].im;
                float m = std::sqrt(re * re + im * im);
                if (m > mag) mag = m;
            }

            float db = 20.0f * std::log10(mag + eps);
            float v = (db + 60.0f) / 60.0f;
            v = std::clamp(v, 0.0f, 1.0f);

            float rise = 0.35f;
            bars[b] = bars[b] + rise * (v - bars[b]);

            float peakFall = 0.010f;
            if (bars[b] > peaks[b]) peaks[b] = bars[b];
            else peaks[b] = std::max(0.0f, peaks[b] - peakFall);
        }
    }

    void draw(SDL_Renderer* r, int W, int H) const {
        int y1    = H - 10;
        int y0    = (int)(0.55f * H);
        int hSpec = y1 - y0;

        int barW = W / BARS;

        for (int b = 0; b < BARS; ++b) {
            int xpx = b * barW;

            int bh = (int)(bars[b] * float(hSpec));
            SDL_Rect rect{ xpx, y1 - bh, barW - 2, bh };
            SDL_RenderFillRect(r, &rect);

            int py = y1 - (int)(peaks[b] * float(hSpec));
            SDL_RenderDrawLine(r, xpx, py, xpx + barW - 3, py);
        }
    }
};

// Car model + 6-transition shift
struct CarModel {
    float rpmIdle          = 900.0f;
    float rpmRed           = 7200.0f;

    float rpmUpRate        = 11000.0f;
    float rpmDownRate      = 9000.0f;

    float maxAccel         = 9.5f;
    float drag             = 0.40f;
    float roll             = 0.85f;

    float wheelCirc        = 2.05f;
    float finalDrive       = 3.20f;

    float gearRatios[8]    = {0.0f, 3.00f, 2.05f, 1.55f, 1.20f, 0.98f, 0.82f, 0.68f};

    float upshiftRpmBase   = 6600.0f;
    float downshiftRpmBase = 1400.0f;

    float speed            = 0.0f;
    int   gear             = 1;
    bool  shifting         = false;
    bool  upshift          = true;
    int   pendingGear      = 1;

    float rpm              = 900.0f;

    float shiftT           = 0.0f;
    float shiftDur         = 0.14f;

    float torqueMul        = 1.0f;
    float shiftBurst       = 0.0f;

    float rpmBlend         = 0.28f;

    float sPreEnd          = 0.16f;
    float sCutEnd          = 0.30f;
    float sSwapAt          = 0.40f;
    float sRampEnd         = 0.60f;
    float sSetEnd          = 0.80f;

    static inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

    inline void compute_shift_points(float throttle, float& upRpm, float& downRpm) const {
        float t = clamp01(throttle);
        upRpm = lerp(3200.0f, upshiftRpmBase, t);
        downRpm = lerp(1100.0f, downshiftRpmBase, t);
    }

    inline float compute_rpm_target_from_speed() const {
        float wheelRPM = (speed / wheelCirc) * 60.0f;
        float ratio = gearRatios[gear] * finalDrive;
        float rt = rpmIdle + wheelRPM * ratio;
        return std::min(rt, rpmRed);
    }

    inline void start_shift(int newGear, bool isUpshift) {
        shifting    = true;
        upshift     = isUpshift;
        pendingGear = newGear;
        shiftT      = 0.0f;
        shiftDur    = upshift ? 0.120f : 0.160f;
        torqueMul   = 1.0f;
        shiftBurst  = 0.0f;
    }

    inline float tri_pulse(float x, float c, float w) const {
        float d = std::fabs(x - c);
        if (d >= w) return 0.0f;
        return 1.0f - (d / w);
    }

    inline void update_shift(float dt, float throttle) {
        shiftT += dt;
        float s = (shiftDur > 1e-6f) ? (shiftT / shiftDur) : 1.0f;
        if (s > 1.0f) s = 1.0f;

        shiftBurst = tri_pulse(s, sSwapAt, 0.028f);

        float preBoost = 1.0f + 0.06f * clamp01(s / sPreEnd);

        float cutMul = 1.0f;
        if (s <= sCutEnd) {
            float u = (s - sPreEnd) / (sCutEnd - sPreEnd);
            u = clamp01(u);
            cutMul = 1.0f - u;
        } else cutMul = 0.0f;

        float rampMul = 0.0f;
        if (s >= sSwapAt && s <= sRampEnd) {
            float u = (s - sSwapAt) / (sRampEnd - sSwapAt);
            u = clamp01(u);
            rampMul = u;
        } else if (s > sRampEnd) rampMul = 1.0f;

        float settleMul = 1.0f;
        if (s >= sRampEnd && s <= sSetEnd) {
            float u = (s - sRampEnd) / (sSetEnd - sRampEnd);
            u = clamp01(u);
            float wob    = std::sin(2.0f * PI * 2.0f * u) * std::exp(-3.5f * u);
            float wobAmp = 0.030f + 0.020f * clamp01(throttle);
            settleMul = 1.0f + wobAmp * wob;
        }

        float stabMul = 1.0f;
        if (s > sSetEnd) {
            float u = (s - sSetEnd) / (1.0f - sSetEnd);
            u = clamp01(u);
            stabMul = 1.0f + (settleMul - 1.0f) * (1.0f - u);
        }

        if (s < sPreEnd) torqueMul = preBoost;
        else if (s < sSwapAt) torqueMul = cutMul;
        else if (s < sRampEnd) torqueMul = rampMul;
        else if (s < sSetEnd) torqueMul = settleMul;
        else torqueMul = stabMul;

        if (gear != pendingGear && s >= sSwapAt) gear = pendingGear;

        if (shiftT >= shiftDur) {
            shifting   = false;
            torqueMul  = 1.0f;
            shiftBurst = 0.0f;
        }
    }

    void step(float dt, float throttle) {
        throttle = clamp01(throttle);

        if (shifting) update_shift(dt, throttle);

        float tMul  = std::clamp(torqueMul, 0.0f, 1.12f);
        float accel = throttle * maxAccel * tMul;

        float resist = drag * speed + roll;
        speed += (accel - resist) * dt;
        if (speed < 0.0f) speed = 0.0f;

        float rpmTarget = compute_rpm_target_from_speed();

        if (!shifting) {
            float upRpm, downRpm;
            compute_shift_points(throttle, upRpm, downRpm);

            if (rpmTarget > upRpm && gear < 7) start_shift(gear + 1, true);
            else if (rpmTarget < downRpm && gear > 1) start_shift(gear - 1, false);
        }

        if (shifting && torqueMul > 1.0f) {
            float bias = 1.0f + 0.012f * (torqueMul - 1.0f) / 0.06f;
            rpmTarget = std::min(rpmTarget * bias, rpmRed);
        }

        if (shifting) {
            rpm = rpm + rpmBlend * (rpmTarget - rpm);
            if (upshift) rpm = std::max(rpmIdle, rpm * 0.9988f);
        } else {
            float maxUp = rpmUpRate * dt;
            float maxDn = rpmDownRate * dt;
            if (rpm < rpmTarget) rpm = std::fmin(rpm + maxUp, rpmTarget);
            else                 rpm = std::fmax(rpm - maxDn, rpmTarget);
        }

        if (throttle < 0.02f && speed < 0.15f && !shifting) {
            rpm += 0.10f * (rpmIdle - rpm);
        }

        if (rpm < rpmIdle) rpm = rpmIdle;
    }
};

int main(int, char**) {
#if defined(__SSE__)
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
#endif

    if (SDL_Init(SDL_INIT_AUDIO | SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
        std::fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    const int W = 900, H = 420;
    SDL_Window* win = SDL_CreateWindow("Bugatti W16 - Masculine 6T (W/S)",
                                       SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                       W, H, SDL_WINDOW_SHOWN);
    if (!win) {
        std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* ren = SDL_CreateRenderer(win, -1,
        SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
    );
    if (!ren) {
        std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    SDL_AudioSpec want{}, have{};
    want.freq     = 48000;
    want.format   = AUDIO_F32SYS;
    want.channels = 1;
    want.samples  = 2048;
    want.callback = audio_cb;

    EngineSynth synth;
    synth.sampleRate = want.freq;
    want.userdata    = &synth;

    SDL_AudioDeviceID dev = SDL_OpenAudioDevice(
        nullptr, 0, &want, &have, SDL_AUDIO_ALLOW_SAMPLES_CHANGE
    );
    if (!dev) {
        std::fprintf(stderr, "SDL_OpenAudioDevice failed: %s\n", SDL_GetError());
        SDL_DestroyRenderer(ren);
        SDL_DestroyWindow(win);
        SDL_Quit();
        return 1;
    }

    SDL_PauseAudioDevice(dev, 0);

    // савалгаа харуулахад зориулсан тасралтгүй урсаж байдаг буфер
    const int SCOPE_N = 2048;
    const int HOP_N   = 256;

    std::vector<float> scope(SCOPE_N, 0.0f);
    std::vector<float> hop(HOP_N, 0.0f);

    const int VIS_N = 1024;
    std::vector<float> vis(VIS_N);

    std::vector<float> hann(VIS_N);
    for (int i = 0; i < VIS_N; ++i) {
        hann[i] = 0.5f - 0.5f * std::cos(2.0f * PI * float(i) / float(VIS_N - 1));
    }
    SpectrumState spec(VIS_N, hann.data());

    CarModel car;

    float thr = 0.0f;
    float thrTarget = 0.0f;
    float load = 0.35f;

    bool running = true;
    uint32_t lastTicks = SDL_GetTicks();
    uint32_t lastPrint = lastTicks;

    std::printf("Controls:\n");
    std::printf("  W/S: throttle (masculine bugatti)   ESC: quit\n");
    std::printf("Play:\n");
    std::printf("  - Hold W smoothly for upshifts.\n");
    std::printf("  - Release W / hold S to slow down; do nothing -> idle.\n\n");

    while (running) {
        uint32_t nowTicks = SDL_GetTicks();
        float dt = float(nowTicks - lastTicks) / 1000.0f;
        if (dt > 0.05f) dt = 0.05f;
        lastTicks = nowTicks;

        const Uint8* ks = SDL_GetKeyboardState(nullptr);
        bool wDown = ks[SDL_SCANCODE_W] != 0;
        bool sDown = ks[SDL_SCANCODE_S] != 0;

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            if (e.type == SDL_KEYDOWN && e.key.keysym.sym == SDLK_ESCAPE) running = false;
        }

        if (wDown && !sDown) thrTarget += 1.35f * dt;
        else if (sDown && !wDown) thrTarget -= 1.85f * dt;
        else {
            float decay = 1.80f * dt;
            if (thrTarget > 0.0f) thrTarget = std::max(0.0f, thrTarget - decay);
            else thrTarget = 0.0f;
        }
        thrTarget = clamp01(thrTarget);

        // Throttle smoothing
        thr += 0.12f * (thrTarget - thr);
        thr = clamp01(thr);

        car.step(dt, thr);

        g_rpm_u       .store(f2u(car.rpm       ), std::memory_order_relaxed);
        g_throttle_u  .store(f2u(thr           ), std::memory_order_relaxed);
        g_load_u      .store(f2u(load          ), std::memory_order_relaxed);
        g_torqueMul_u .store(f2u(car.torqueMul ), std::memory_order_relaxed);
        g_shiftBurst_u.store(f2u(car.shiftBurst), std::memory_order_relaxed);

        // Осциллоскопын буферт HOP_N sample-ийг дарааллаар нь нэмж урсгана
        g_ring.read_block(hop.data(), HOP_N);

        std::memmove(scope.data(), scope.data() + HOP_N, (SCOPE_N - HOP_N) * sizeof(float));
        std::memcpy (scope.data() + (SCOPE_N - HOP_N), hop.data(), HOP_N * sizeof(float));

        // FFT-д зориулж scope-ийн сүүлийн VIS_N sample-ийг авч хэрэглэнэ
        std::memcpy(vis.data(), scope.data() + (SCOPE_N - VIS_N), VIS_N * sizeof(float));
        spec.update_if_needed(vis.data(), VIS_N, nowTicks);

        SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
        SDL_RenderClear(ren);

        SDL_SetRenderDrawColor(ren, 220, 220, 220, 255);
        draw_wave(ren, scope.data(), SCOPE_N, W, H);

        SDL_SetRenderDrawColor(ren, 220, 220, 220, 255);
        spec.draw(ren, W, H);

        SDL_RenderPresent(ren);

        if (nowTicks - lastPrint > 1000) {
            lastPrint = nowTicks;
            float kmh = car.speed * 3.6f;
            std::printf("\rGear=%d  Shifting=%d  Speed=%6.1f km/h  RPM=%4.0f  Thr=%.2f  TorqueMul=%.2f  Crack=%.2f    ",
                        car.gear, car.shifting ? 1 : 0, kmh, car.rpm, thr, car.torqueMul, car.shiftBurst);
            std::fflush(stdout);
        }
    }

    std::printf("\n");

    SDL_CloseAudioDevice(dev);
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
