
# Tasks to mess around in the near future

    Revive Lockless job system experiments
    Revive priority based job system experiments
    Implement multiple parallel and sequencial job groups and graphs demo
        Try to synchronize several workers on some pre placed barrier mechanism, or use atomic counters
        Try to group different tasks with different atomic counters 
        Try to build concurrent job graph using array of atomic counters
    Fix Race Conditions on the job classes
    Make Job System example doesn't crash
    
    Try to parallelize rasterization by subdividing large triangles based on certain threshold
        - recursively assemble rasterizable small triangle chunks into the list
        - submit them parallel rasterizer job system
    

    DONE Fix the coordinate system transformation and convention mess, write a note.
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

    I'm just learning about software renderer and shaders in my freetime.
    I think using graphics API is overrated and writing own software renderer is cool.
    For simplicity I'm gonna use SDL2, and all the rendering computation should happend 
    solely on the CPU, maybe I can use OpenCL where it makes sense
    The goal is to learn something.
  
    Might implement followings exercises
      - Simple rendering for primitives like pixels, lines
        - Triangle, quadrat, stars etc
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


# On Ubuntu 24.04

    sudo apt install automake m4 libtool cmake build-essential autoconf autoconf-archive automake libtool-bin python3.12-venv python3.13-venv

    Download Lunarg Vulkan SDK first

    mv ~/Downloads/vulkansdk-linux-x86_64-1.4.341.1.tar.xz ~/vulkan
    cd ~/vulkan && tar -xvf vulkansdk-linux-x86_64-1.4.341.1.tar.xz
    
    vim ~/.bashrc
      add at bottom
      ```
      source ~/vulkan/1.4.341.1/setup-env.sh
      ```


    VCPKG installation on ubuntu 
    https://lindevs.com/install-vcpkg-on-ubuntu

    export VCPKG_ROOT="/opt/vcpkg"

    sudo vcpkg install sdl2[vulkan] --recurse
    sudo vcpkg install sdl2-image
    sudo vcpkg install --recurse sdl2-image[libjpeg-turbo]
    sudo vcpkg install glm
    sudo vcpkg install assimp

    
    Compilation steps on ubuntu 24.04

    cd cpp-folders && mkdir build && cd build
    export VCPKG_ROOT="/opt/vcpkg"
    cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
    make -j20
    cd src/hello-pixel-primitives && ./HelloPixel
    



# On MacOS, it is similar

    Vulkan installation
    https://github.com/MrVideo/ARMMoltenVKGuide
    https://vulkan-tutorial.com/Development_environment#page_MacOS
    https://vulkan.lunarg.com/sdk/home#mac

    xcode-select --install
    source ~/VulkanSDK/<version>/setup-env.sh
    export VULKAN_SDK=~/VulkanSDK/<version>/macOS
    export PATH="$VULKAN_SDK/bin:$PATH"
    vulkaninfo | head
    glslangValidator --version

    CMake Vulkan detection behavior in this repo
      - Global and automatic: shs-renderer-lib detects Vulkan + SDL2 Vulkan capability once, demos consume the shared result
      - Linux/Windows: uses normal find_package(Vulkan) + find_program(glslangValidator)
      - macOS: tries normal detection first, then falls back to VULKAN_SDK path if needed


    brew install vcpkg
    git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
    export VCPKG_ROOT="$HOME/vcpkg"

    Install libs on Apple Silicon

    vcpkg install "sdl2[vulkan]:arm64-osx" --recurse
    vcpkg install "sdl2-image[libjpeg-turbo]:arm64-osx"
    vcpkg install "glm:arm64-osx"
    vcpkg install "assimp:arm64-osx"


    cd cpp-folders && mkdir build && cd build
    export VCPKG_ROOT="$HOME/vcpkg"
    export VK_ICD_FILENAMES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
    export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"
    cmake .. -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
    make -j20
    cd src/hello-pixel-primitives && ./HelloPixel
    cd ../hello-plumbing && ./HelloPassBasics && ./HelloPassBasicsVulkan && ./HelloForwardPlusStressVulkan && ./HelloVulkanShadowTechniques
    # HelloPassBasicsVulkan is shader-based multi-pass Vulkan path:
    #   soft shadow map + PBR scene + camera/per-object motion blur + light shafts + lens flare + FXAA
    # Default shaders are in:
    #   cpp-folders/src/shs-renderer-lib/shaders/vulkan/pb_*.vert|frag
    # Forward+ Vulkan shader set also includes:
    #   cpp-folders/src/shs-renderer-lib/shaders/vulkan/fp_stress_scene.vert|frag
    #   cpp-folders/src/shs-renderer-lib/shaders/vulkan/fp_stress_shadow.vert
    #   cpp-folders/src/shs-renderer-lib/shaders/vulkan/fp_stress_light_cull.comp
    #   cpp-folders/src/shs-renderer-lib/shaders/vulkan/fp_stress_depth_reduce.comp



# On Windows 11

    VCPKG related environment variable, system properties -> environment variables -> system variables -> New...
    change VCPKG path according where you installed.

      CMAKE_TOOLCHAIN_FILE = C:\src\misc\vcpkg\scripts\buildsystems\vcpkg.cmake


    vcpkg install sdl2[vulkan] --recurse
    vcpkg install sdl2-image
    vcpkg install sdl2-image:x64-windows-static
    vcpkg install --recurse sdl2-image[libjpeg-turbo]
    vcpkg install libjpeg-turbo
    vcpkg install glm
    vcpkg install assimp


    Use CMake-GUI with Visual Studio 17 2022
    



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
    John Chapman blog
      - https://john-chapman-graphics.blogspot.com
     
      
      
# Result
<p><img width="320" height="320" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/hello_wireframe_monkeyobj_canvas.png"></p>
<p><img width="655" height="534" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/monkey-blinn-phong.png"></p>
<p><img width="688" height="511" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/monkey-toon-shading.png"></p>
<p><img width="1677" height="477" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/normal_zbuffer_debug.png"></p>
<p><img width="657" height="540" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/depth-of-field-monkeys.png"></p>
<p><img width="511" height="345" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/per-object-motion-blur-monkeys.png"></p>
<p><img width="711" height="574" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/multi-pass-per-object-motion-blur.png"></p>
<p><img width="836" height="719" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/multi-pass-motion-blur-dof-fxaa.png"></p>
<p><img width="437" height="317" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/pseudo-lens-flare.png"></p>
<p><img width="640" height="360" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/mongolian-flag.png"></p>
<p><img width="320" height="320" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/fbm.png"></p>
<p><img width="460" height="360" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/subaru.png"></p>
<p><img width="800" height="600" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/shadow-mapping-soft.png"></p>
<p><img width="1200" height="900" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/image-based-lighting.png"></p>
<p><img width="1200" height="900" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/physics-based-rendering-monkey.png"></p>
<p><img width="1200" height="900" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/physics-based-rendering-light-shafts.png"></p>
<p><img width="1200" height="900" src="https://raw.githubusercontent.com/sharavsambuu/leisure-software-renderer/master/images/physics-based-rendering-light-shafts-1.png"></p>




      
