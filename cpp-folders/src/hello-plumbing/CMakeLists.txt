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
