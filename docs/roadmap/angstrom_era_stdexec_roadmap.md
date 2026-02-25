# Roadmap: The Angstrom-Era Software Renderer (stdexec)

This roadmap outlines the visionary, long-term trajectory for `shs-renderer-lib`'s software backend. It details the transition from a traditional thread pool to a massive, many-core dataflow architecture powered by C++ `stdexec` (Senders and Receivers), preparing the engine for the "Angstrom Era" of hardware.

## The Vision: Return of the PS3 Cell SPUs

In the early 2000s, the PlayStation 3 introduced the Cell Broadband Engine. It was notoriously difficult to program because developers could no longer rely on a single, monolithic CPU thread. They had to explicitly marshal data to and from tiny, localized Synergistic Processing Elements (SPUs) and dispatch pure compute jobs.

The hardware industry is rapidly iterating toward a modern, PC-scale equivalent of this architecture. In the "Angstrom Era" (representing the physical limits of atomic-scale silicon after the nanometer era), the line between CPU and GPU will blur entirely. We anticipate Systems on a Chip (SoCs) where thousands of smaller CPU cores share a massive pool of global RAM, negating the need for a discrete GPU over a PCIe bus.

In this future, traditional graphics paradigms (OpenGL, Vulkan fixed-function state machines) become a bottleneck. The software renderer is reborn not as a fallback, but as **the primary way to render**, utilizing pure C++ computation distributed across thousands of generalized cores. 

## The Core Technology: `stdexec`

`stdexec` (the reference implementation for the upcoming standard C++ `std::execution` framework) is the modern answer to the Cell processor's SPU dispatch dilemma. It abandons traditional OS-level thread-wrangling in favor of **Task Graphs** (Senders and Receivers).

Because the SHS Renderer is already deeply rooted in **Value-Oriented Programming (VOP)** and **Data-Oriented Design (DOD)**, translating our pure-function ECS loops, vertex transformations, and tile-binning rasterization algorithms into `stdexec` task nodes is a natural evolution.

## The Ternary Backend Strategy

Rather than fully replacing the current rendering backends, the Angstrom Era formally adopts a **Three-Pillar Architecture**. Each backend implements the same `IRenderBackend` interface but serves a distinctly different architectural purpose:

1. **`sw_backend` (The Reference Truth):** The pristine, "Classic" standard C++ software renderer. It acts as the ultimate truth for math, coordinate system conventions (LH/RH Jolt integrations), and pixel-perfect rendering logic, free from the noise of massive concurrency or complex API synchronization.
2. **`vk_backend` (The Silicon Heavyweight):** The state-of-the-art backend targeting modern discrete GPUs via explicit hardware acceleration.
3. **`stdexec_backend` (The Angstrom Vanguard):** The highly experimental, scalable dataflow backend orchestrating massive multi-core execution (mimicking the PS3 SPU dispatch). 

### The Feature Validation Loop
Every new feature (e.g., clustered shading, soft shadows) must survive this pipeline:
*   **Step 1:** Correctness and perfect pixel math validated on `sw_backend`.
*   **Step 2:** Performance and API execution validated on `vk_backend`.
*   **Step 3:** Scaling and lock-free dataflow execution validated on `stdexec_backend`.

## Phase 1: Prototype and `stdexec` Integration

1.  **Drop-In `stdexec` Framework**: Introduce the `stdexec` reference library to the codebase alongside the existing `ThreadPoolJobSystem`.
2.  **Granular Task Senders**: Refactor core software rendering loops (e.g., triangle setup, edge function evaluation, tile shading) from block loops into `stdexec` Sender chains.
3.  **Tile-Based Rendering Rewrite**: Ensure the software rasterizer is strictly tile-based (e.g., 16x16 or 32x32 pixel tiles). Each tile becomes an independent job node in the `stdexec` graph, guaranteeing zero contention when writing to the localized color/depth spans.

## Phase 2: Wait-Free Data Pipelines

1.  **DAG (Directed Acyclic Graph) Scheduling**: Replace all manual threading barriers (`WaitGroup`) with explicit dependency chains using `stdexec::then`, `stdexec::when_all`, and `stdexec::let_value`.
2.  **Pure Dataflow Representation**: The entire frame from Input Reduction -> Culling -> Rasterization -> Blit must be represented as a single, immutable execution graph submitted to the `stdexec` scheduler at the start of the frame.
3.  **Linear Memory Bounding**: Every Sender in the graph must receive pre-allocated `std::span` outputs. The task graph must guarantee a zero-allocation (`malloc`-free) execution pipeline.

## Phase 3: Hardware Scalability Testing

1.  **Massive Multi-Core Topologies**: Benchmarking the software renderer's `stdexec` scheduler on high-core-count CPUs (HEDT, Threadripper, server-grade ARM instances) to measure linear scalability.
2.  **GPU Scheduler Backend (Experimental)**: Utilizing compiler toolchains like NVIDIA HPC (`nvc++`), swap the standard CPU thread-pool scheduler for a GPU scheduler. Test if the precise same C++ `stdexec` Task Graph can be natively compiled and dispatched to a discrete GPU as a massive compute shader, blurring the line between software and hardware rendering.

## The Angstrom-Era Guarantee

By aligning the engine logic with `stdexec` and strict Data-Oriented Design, `shs-renderer-lib` ensures its survival and dominance in the post-GPU era. When hardware inevitably pivots to thousand-core unified APUs, the engine will instantly and natively scale across all available silicon without dropping a single frame to OS thread starvation or lock contention.
