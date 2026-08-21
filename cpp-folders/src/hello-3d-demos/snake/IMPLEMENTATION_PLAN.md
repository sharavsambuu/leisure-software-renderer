# Hello 3D Snake Demo — Implementation Plan

## Goals

Build a Vulkan-based 3D snake game demo using the `shs_renderer.hpp` software renderer pipeline. The scene is a **retro arcade tunnel** view where:

- A wireframe tunnel (cylindrical lattice) recedes into the distance
- A snake glides through it, head and body rendered as torus primitives with vertex colors
- Food spawns at random positions along the tunnel path; eating grows the tail
- The camera orbits around the snake from above

## Domain Architecture

```
snake/
├── CMakeLists.txt                 # defines Hello3DSnake target
│   └── include_directories("${CMAKE_CURRENT_LIST_DIR}/../")  → resolves relative includes
├── hello_3d_snake.cpp            # main: SDL2 window + game loop
└── domains/
    ├── config/difficulty.hpp     # Difficulty struct + level meta-table (replay-ready)
    └── matrix/
        ├── snake.contract.hpp    # core types: SnakeCommand, SnakeMovementFrame, etc.
        ├── snake.action.hpp      # reduces command list → movement frame
        ├── snake.event.hpp       # event enum: HEAD_MOVED, SELF_COLLISION, FOOD_EATEN
        └── snake.reducer.hpp     # applies actions to state (movement, collision, eating)
    └── spatial_fx/
        └── snake.plan.hpp        # camera params + lighting for tunnel rendering
```

## Key Design Decisions

- **Domain folder = responsibility:** `domains/matrix/*` holds the game logic; `domains/spatial_fx/*` holds FX/rendering setup.
- **Replay-ready difficulty table** in `config/difficulty.hpp` lets you swap levels mid-run without recompiling.
- **Relative includes resolve via CMake include paths:** `include_directories("${CMAKE_CURRENT_LIST_DIR}/../")` adds the parent of CMakeLists.txt to the search path, so `"snake.contract.hpp"` from any file under `domains/` resolves to `../matrix/snake.contract.hpp`.

## Build & Run

```bash
cd cpp-folders/build && cmake .. -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
    -DHello3DSnake_ENABLED=ON \
    && cmake --build . --target Hello3DSnake
./cpp-folders/build/src/hello-3d-demos/snake/Hello3DSnake
```

## Status

All domain headers and main entry point are now in place. The build system uses the same `include_directories(${HelloSHSRenderer_SOURCE_DIR}/)` pattern as all other demos, ensuring consistency across the entire `hello-3d-demos` suite.
