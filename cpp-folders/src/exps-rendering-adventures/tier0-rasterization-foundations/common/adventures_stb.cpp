/*
    Single translation unit owning the stb_image_write implementation for the
    whole adventures tree (header-inclusion hygiene: exactly one TU defines
    STB_IMAGE_WRITE_IMPLEMENTATION).
*/

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "adventures_frame.hpp"

namespace adventures
{
    bool Frame::save_png(const std::string& path) const
    {
        const int stride_bytes = width * 4;
        return stbi_write_png(path.c_str(), width, height, 4, rgba.data(), stride_bytes) != 0;
    }
}
