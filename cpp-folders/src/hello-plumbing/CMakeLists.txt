project(HelloPlumbing LANGUAGES CXX)


file(COPY ../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


add_executable(HelloPassBasics hello_pass_basics.cpp)
target_link_libraries(HelloPassBasics PRIVATE shs::renderer)
if(MSVC)
    target_compile_options(HelloPassBasics PRIVATE /W4)
else()
    target_compile_options(HelloPassBasics PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(HelloPassBasics PRIVATE $<$<CONFIG:Release>:-O3>)
    target_compile_options(HelloPassBasics PRIVATE $<$<CONFIG:Debug>:-g>)
endif()
target_compile_features(HelloPassBasics PRIVATE cxx_std_20)

find_package(Vulkan QUIET)
find_program(GLSLANG_VALIDATOR NAMES glslangValidator glslangValidator.exe)

if(Vulkan_FOUND AND GLSLANG_VALIDATOR)
    set(VK_TRI_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
    set(VK_TRI_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")
    file(MAKE_DIRECTORY "${VK_TRI_OUT_DIR}")

    set(VK_TRI_VERT "${VK_TRI_SHADER_DIR}/vulkan_triangle.vert")
    set(VK_TRI_FRAG "${VK_TRI_SHADER_DIR}/vulkan_triangle.frag")
    set(VK_TRI_VERT_SPV "${VK_TRI_OUT_DIR}/vulkan_triangle.vert.spv")
    set(VK_TRI_FRAG_SPV "${VK_TRI_OUT_DIR}/vulkan_triangle.frag.spv")

    add_custom_command(
        OUTPUT "${VK_TRI_VERT_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_TRI_VERT}" -o "${VK_TRI_VERT_SPV}"
        DEPENDS "${VK_TRI_VERT}"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${VK_TRI_FRAG_SPV}"
        COMMAND ${GLSLANG_VALIDATOR} -V "${VK_TRI_FRAG}" -o "${VK_TRI_FRAG_SPV}"
        DEPENDS "${VK_TRI_FRAG}"
        VERBATIM
    )
    add_custom_target(HelloVulkanTriangleShaders DEPENDS "${VK_TRI_VERT_SPV}" "${VK_TRI_FRAG_SPV}")

    add_executable(HelloVulkanTriangle hello_vulkan_triangle.cpp)
    add_dependencies(HelloVulkanTriangle HelloVulkanTriangleShaders)
    target_link_libraries(HelloVulkanTriangle PRIVATE shs::renderer Vulkan::Vulkan)
    target_compile_definitions(HelloVulkanTriangle PRIVATE
        SHS_VK_TRIANGLE_VERT_SPV="${VK_TRI_VERT_SPV}"
        SHS_VK_TRIANGLE_FRAG_SPV="${VK_TRI_FRAG_SPV}"
    )
    if(MSVC)
        target_compile_options(HelloVulkanTriangle PRIVATE /W4)
    else()
        target_compile_options(HelloVulkanTriangle PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(HelloVulkanTriangle PRIVATE $<$<CONFIG:Release>:-O3>)
        target_compile_options(HelloVulkanTriangle PRIVATE $<$<CONFIG:Debug>:-g>)
    endif()
    target_compile_features(HelloVulkanTriangle PRIVATE cxx_std_20)
else()
    message(STATUS "HelloVulkanTriangle skipped (requires Vulkan SDK + glslangValidator)")
endif()
