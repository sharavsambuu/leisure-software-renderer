/*
    Single translation unit owning the stb_image (PNG *read*) implementation
    for the AD4 known-answer test executables (t0/t1). Kept separate from
    adventures_stb.cpp (stb_image_write) so the demo binaries themselves do
    not link the decoder; the known-answer gates read the *_sw demo PNGs
    back for independent numerical checks.
*/

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "adventures_frame.hpp"

#include <string>
#include <vector>

namespace adventures
{
    // Loads a PNG as top-left-origin RGBA8. Declared in the known-answer
    // test TUs (this is the only definition — one stb_image TU per tree).
    bool ka_load_png_rgba(const std::string& path, std::vector<uint8_t>& rgba, int& w, int& h)
    {
        int comp = 0;
        uint8_t* data = stbi_load(path.c_str(), &w, &h, &comp, 4);
        if (data == nullptr) return false;
        rgba.assign(data, data + size_t(w) * size_t(h) * 4u);
        stbi_image_free(data);
        return true;
    }
} // namespace adventures
