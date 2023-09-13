project(HelloParallelization)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
find_package(glm REQUIRED)
find_package(Assimp REQUIRED)
find_package(Boost REQUIRED COMPONENTS)

if(ASSIMP_FOUND)
  include_directories(${ASSIMP_INCLUDE_DIR})
endif() 
include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_IMAGE_INCLUDE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${stb_SOURCE_DIR}/)
include_directories(${SHSRenderer_SOURCE_DIR}/)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})
set(Boost_USE_STATIC_LIBS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")


add_executable(HelloNestedLoops hello_nested_loops.cpp
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
  )
target_link_libraries(HelloNestedLoops Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloThreadShaderJobs hello_thread_shader_jobs.cpp)
target_link_libraries(HelloThreadShaderJobs Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloThreadShaderZoned hello_thread_shader_zoned.cpp)
target_link_libraries(HelloThreadShaderZoned Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(PrimitiveLoops primitive_loops.cpp)
target_link_libraries(PrimitiveLoops Threads::Threads)

add_executable(PrimitiveThreads primitive_threads.cpp)
target_link_libraries(PrimitiveThreads Threads::Threads)

add_executable(PrimitiveFibers primitive_fibers.cpp)
target_link_libraries(PrimitiveFibers 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  Threads::Threads 
  )

add_executable(PrimitiveNestedFibers primitive_nested_fibers.cpp)
target_link_libraries(PrimitiveNestedFibers Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lpthread
  )


add_executable(HelloThreadJobSystem hello_thread_job_system.cpp)
target_link_libraries(HelloThreadJobSystem 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  Threads::Threads 
  )

add_executable(HelloThreadJobSystemLockless hello_thread_job_system_lockless.cpp)
target_link_libraries(HelloThreadJobSystemLockless 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  Threads::Threads 
  )

add_executable(HelloThreadJobSystemLocklessPriority hello_thread_job_system_lockless_priority.cpp)
target_link_libraries(HelloThreadJobSystemLocklessPriority 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  Threads::Threads 
  )

add_executable(HelloFiberJobSystemGraph hello_fiber_job_system_graph.cpp)
target_link_libraries(HelloFiberJobSystemGraph Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  -lpthread
  )

add_executable(HelloFiberJobSystemBoost hello_fiber_job_system_boost.cpp)
target_link_libraries(HelloFiberJobSystemBoost 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  Threads::Threads 
  )