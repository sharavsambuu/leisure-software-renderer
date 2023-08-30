# First result

<p align="center">
  <img width="320" height="320" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/hello_wireframe_monkeyobj_canvas.png">
</p>


# Tasks to mess in the near future
    Realtime looper using SDL2
    Realtime canvas renderer on SDL2 window
    Map vertext coordinates
      - Map vertices from local obj space coordinates to scene space coordinates
      - Map vertices from scene space coordinates to camera space coordinates
      - Map vertices from camera space coordinates to the homogenous space coordinates
    Add a few more OBJ in the scene
    Rotate and Translate 3D model
    Rotate and Translate camera, drive through scene
    Control camera zoom in/out 
    Control camera frustum related params
    lists go on...


# About
    I'm just exercizing about software renderer and shaders in my freetime.
    I think using graphics api is too overrated and writing own software renderer is cool.
    For simplicity I'm gonna use SDL2, and all the rendering computation should happend 
    solely on the CPU, no accesselerators, otherwise there is no point to start this repo.
    The goal is to learn something.
  
    Might implement followings exercises
      - Simple rendering for primitives like pixels, lines
        - Triangle, quadrat, start etc
      - Some rasterizer for primitives like triangles, culling... currently no idea on how to implement
      - ZBuffer implementation
      - Render wireframes from obj file
      - Apply some shader logics
      - Render surface with following lightening models
        - Flat shading
        - Gouraud shading
        - Phong shading
        - Blinn-Phong shading
      - Light types
        - point light
        - spot light
        - directional light
        - some ambience
      - Texture mapping
      - Normal mapping
      - Shadow mapping
      - Ambient occlusion
      - Simple ways to parallelize computations using Threads on CPUs
      - Simple render buffer
        - Render to texture
      - Simple frame buffer
      - Simple ping-pong mechanism
      - Simple effect like depth of field
      - Gaussian blur
      - Toon mapping
      - Maybe other cool effects on the internet
      - Skybox
      - Environment mapping
      - IBL aka Image Based Lighting
      - PBR aka Physically Based Rendering
      - Maybe if my kung-fu skill good enough, might implement deferred rendering
      - Transparency
      - Python port using PyGame or PySDL2


# Libraries
    sudo apt install automake m4 libtool cmake build-essential
    sudo apt install libboost-all-dev
    sudo apt install libsdl2-dev
    sudo apt install libpng-dev
    sudo apt install libsdl2-image-dev
    sudo apt install libglm-dev
    sudo apt install libassimp-dev
    sudo ldconfig

    If you windows 10, you can use WSL2 with Ubuntu 20.04LTS, and everything is almost same.

# Other dependency installation
    pass


# Compilation steps
    cd cpp-folders && mkdir build && cd build
    cmake ..
    make
    cd src/hello-pixel-primitives && ./HelloPixel


# References
    Tinyrendere project
      - https://github.com/ssloy/tinyrenderer
      - https://github.com/ssloy/tinyrenderer/wiki
    SDL2 Tutorial and links
      - https://www.youtube.com/watch?v=gOXg1ImX5j0
      - https://github.com/aminosbh/sdl2-samples-and-projects
    Youtube video about software renderer
      - https://www.youtube.com/watch?v=A3KUGbkcKgc
    Bresenham's line drawing algorithm
      - https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
      - https://github.com/ssloy/tinyrenderer/wiki/Lesson-1:-Bresenham%E2%80%99s-Line-Drawing-Algorithm
    Assimp library for model loader
      - https://github.com/assimp/assimp
    Triangle rasterization
      - Youtube tutorial
        https://www.youtube.com/watch?v=k5wtuKWmV48
    Data Oriented Design
      - https://medium.com/mirum-budapest/introduction-to-data-oriented-programming-85b51b99572d
      
      





      

