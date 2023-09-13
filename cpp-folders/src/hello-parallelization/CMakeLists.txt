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

add_executable(HelloThreadJobs hello_thread_jobs.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
)
target_link_libraries(HelloThreadJobs Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(HelloThreadZoned hello_thread_zoned.cpp 
  ${SHSRenderer_SOURCE_DIR}/shs_renderer.hpp
)
target_link_libraries(HelloThreadZoned Threads::Threads ${SDL2_LIBRARIES} glm::glm)

add_executable(PrimitiveLoops primitive_loops.cpp)
target_link_libraries(PrimitiveLoops Threads::Threads)

add_executable(PrimitiveThreads primitive_threads.cpp)
target_link_libraries(PrimitiveThreads Threads::Threads)

add_executable(PrimitiveFibers primitive_fibers.cpp)
target_link_libraries(PrimitiveFibers Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lpthread
  )

add_executable(PrimitiveNestedFibers primitive_nested_fibers.cpp)
target_link_libraries(PrimitiveNestedFibers Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lpthread
  )

add_executable(HelloFiberJobSystem hello_fiber_job_system.cpp)
target_link_libraries(HelloFiberJobSystem Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  -lpthread
  )

add_executable(HelloFiberJobSystemLockless hello_fiber_job_system_lockless.cpp)
target_link_libraries(HelloFiberJobSystemLockless Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  -lpthread
  )

add_executable(HelloFiberJobSystemLocklessPriority hello_fiber_job_system_lockless_priority.cpp)
target_link_libraries(HelloFiberJobSystemLocklessPriority Threads::Threads 
  -lboost_fiber
  -lboost_thread
  -lboost_context 
  -lboost_system 
  -lboost_chrono
  -lpthread
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