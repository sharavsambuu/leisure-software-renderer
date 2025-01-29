
# Tasks to mess around in the near future

    Switch to SDL3
    Update libraries
    Fix cmake stuffs on Windows 10
    Add notes on how configure dependencies on Windows 10
    Fix Race Conditions on the job classes
    Make Job System example doesn't crash
    Move on to the next original goals like 3D shading and stuffs

    Try to parallelize rasterization by subdividing large triangles based on certain threshold
        - recursively assemble rasterizable small triangle chunks into the list
        - submit them parallel rasterizer job system
    Fix the coordinate system transformation and convention mess, write a note.
    Try to synchronize several workers on some pre placed barrier mechanism, or use atomic counters
    Try to group different tasks with different atomic counters 
    Try to build concurrent job graph using array of atomic counters

    DONE Implement Thread based job system using lockless priority queue
    DONE Realtime looper using SDL2
    DONE Realtime canvas renderer on SDL2 window
    DONE Map vertext coordinates
      - Map vertices from local obj space coordinates to scene space coordinates
      - Map vertices from scene space coordinates to camera space coordinates
      - Map vertices from camera space coordinates to the homogenous space coordinates
    DONE Add a few more OBJ in the scene
    DONE Rotate and Translate 3D model
    DONE Rotate and Translate camera, drive through scene
    


# About

    I'm just exercizing about software renderer and shaders in my freetime.
    I think using graphics api is too overrated and writing own software renderer is cool.
    For simplicity I'm gonna use SDL2, and all the rendering computation should happend 
    solely on the CPU, no accelerators, otherwise there is no point to start this repo.
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
    
    Cool things would be nice to try out
      - Lens Flare
      - Depth of field
      - Motion Blur
      - Global Illumination
      - God rays
      - Volumetric Fog
      - Shadow maps, soft shadows
      - Screen Space Ambient Occlusion
      - Various Physically Based Rendering models
          - Cook-Torrance
          - Torrance-Sparrow
          - Beard-Maxwell
          - Oren-Nayar
          - Poulin-Fournier


# Libraries
    sudo apt install automake m4 libtool cmake build-essential
    sudo apt install libssl-dev
    sudo apt install libsdl2-dev
    sudo apt install libpng-dev
    sudo apt install libsdl2-image-dev
    sudo apt install libglm-dev
    sudo apt install libassimp-dev
    sudo ldconfig

    If you windows 10, you can use WSL2 with Ubuntu 20.04LTS, and everything is almost same.


# Compilation steps
    cd cpp-folders && mkdir build && cd build
    cmake ..
    make -j8
    cd src/hello-pixel-primitives && ./HelloPixel


# References
    
    
    Doom Eternal's multi threaded game engine called Id Tech 7
      - https://www.youtube.com/watch?v=UsmqWSZpgJY
      - https://advances.realtimerendering.com/s2020/RenderingDoomEternal.pdf
    Explained why industry is moving toward to Forward rendering variants and mentioned draw call becomes so cheap and BUS bandwith becomes bottleneck 
      - https://www.youtube.com/watch?v=n5OiqJP2f7w
    Parallelizing the Naughty Dog Engine Using Fibers
      - https://www.youtube.com/watch?v=HIVBhKj7gQU
      - https://www.createursdemondes.fr/wp-content/uploads/2015/03/parallelizing_the_naughty_dog_engine_using_fibers.pdf
    Tinyrenderer project
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
        https://www.youtube.com/watch?v=t7Ztio8cwqM
      - http://www.sunshine2k.de/coding/java/TriangleRasterization/TriangleRasterization.html
    Data Oriented Design
      - https://medium.com/mirum-budapest/introduction-to-data-oriented-programming-85b51b99572d
    Mentioned color spaces
      - https://thebookofshaders.com/06/
    IQ's articles
      - https://iquilezles.org/articles/
    Might be useful to parallelize pixel calculations
      - https://github.com/taskflow/taskflow
    Coroutine Job System
      - https://poniesandlight.co.uk/reflect/coroutines_job_system/
    Nice explanation about Model View Projection spaces
      - https://jsantell.com/model-view-projection/
    Rotation Transformation convention, mentioned positive rotation is counter clock-wise
      - https://www.youtube.com/watch?v=9egyFOt6PHM
     
      
      
# Result
<p><img width="320" height="320" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/hello_wireframe_monkeyobj_canvas.png"></p>
<p><img width="640" height="420" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/monkey-3d.png"></p>
<p><img width="640" height="360" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/mongolian-flag.png"></p>
<p><img width="320" height="320" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/fbm.png"></p>




      

