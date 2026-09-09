# Exps software renderer — pure CPU learning path.
# Location: cpp-folders/src/exps-software-renderer/CMakeLists.txt
# Order: hello-3d-demos before hello-render-target (SDL2_image::Main alias
# poisoning guard, see hello-3d-demos/CMakeLists.txt).

add_subdirectory("hello-pixel-primitives")
add_subdirectory("hello-shaders")
add_subdirectory("hello-3d-primitives")
add_subdirectory("hello-3d-demos")
add_subdirectory("hello-render-target")
