# About
    I'm so bored, just exercizing about software renderer and shaders in my freetime.
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
    sudo apt install automake m4 libtool cmake
    sudo apt install libboost-all-dev && sudo ldconfig
    sudo apt install libsdl2-dev

# References
    Tinyrendere project
      - https://github.com/ssloy/tinyrenderer
      - https://github.com/ssloy/tinyrenderer/wiki
    SDL2 Tutorial
      - https://www.youtube.com/watch?v=gOXg1ImX5j0

      

