project(HelloPlumbing LANGUAGES CXX)


file(COPY ../assets DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")


add_executable(HelloPassPlumbing hello_pass_plumbing.cpp)
target_link_libraries(HelloPassPlumbing PRIVATE shs::renderer)
if(MSVC)
    target_compile_options(HelloPassPlumbing PRIVATE /W4)
else()
    target_compile_options(HelloPassPlumbing PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(HelloPassPlumbing PRIVATE $<$<CONFIG:Release>:-O3>)
    target_compile_options(HelloPassPlumbing PRIVATE $<$<CONFIG:Debug>:-g>)
endif()
target_compile_features(HelloPassPlumbing PRIVATE cxx_std_20)

