# Roadmap: The Angstrom-Era Software Renderer (Virtual SPU)

This roadmap outlines the visionary, long-term trajectory for `shs-renderer-lib`'s software backend. It details the transition to a massive, many-core "Virtual SPU" architecture powered by C++20 `jthreads` and stackless coroutines, preparing the engine for the "Angstrom Era" of hardware.

## The Vision: Return of the PS3 Cell SPUs

In the early 2000s, the PlayStation 3 introduced the Cell Broadband Engine. It was notoriously difficult to program because developers could no longer rely on a single, monolithic CPU thread. They had to explicitly marshal data to and from tiny, localized Synergistic Processing Elements (SPUs) and dispatch pure compute jobs.

The hardware industry is rapidly iterating toward a modern, PC-scale equivalent of this architecture. In the "Angstrom Era," the line between CPU and GPU will blur entirely. I anticipate Systems on a Chip (SoCs) where thousands of smaller CPU cores share a massive pool of global RAM.

In this future, the software renderer is reborn as a **Massive Virtual SPU Array**. Each core is modeled not as a transient task, but as a persistent, cooperative **Virtual SPU** (represented by a `std::jthread`) that "occupies" a core and waits for work to be dispatched through a coroutine-based job system.

## The Philosophy: General-Purpose vs. ASICs

A fundamental pillar of the Angstrom Era is the bet that **General-Purpose Silicon beats Fixed-Function Silicon (ASICs)** at scale.

1.  **Software-Defined Rendering**: GPUs rely on fixed ASICs (rasterizers, samplers) that are "frozen" in silicon. By using `jthreads` on general-purpose cores, we gain 100% flexibility to invent new rendering paradigms without hardware restrictions.
2.  **Unified Resource Pool**: Instead of hitting "black box" bottlenecks in a fixed-function pipeline, the Virtual SPU model treats all silicon as a unified pool that can be rebalanced dynamically (e.g., spinning up more Culling workers if that stage becomes the bottleneck).
3.  **ASIC-Like Performance**: By using **Data-Oriented Design (DOD)** and persistent workers, we achieve the throughput of specialized hardware while retaining the programmability of C++.
4.  **Always-Busy / Non-Blocking Execution**: Adopting the **id Tech** philosophy where workers never "wait." A Virtual SPU never hits a blocking barrier; it either processes a task or suspends a coroutine to immediately pick up the next available job piece.

## The Execution Unit: Virtual SPUs (`std::jthread`)

The **Virtual SPU** is the heart of the execution. We model these units using C++20 `std::jthread`. Unlike traditional threads, these units are:

- **Persistent**: A Virtual SPU "lives" for the duration of the frame or application, eliminating the overhead of frequent OS-level context switching for transient tasks.
- **Cooperative**: Using `std::stop_token` to handle graceful interruption and state-safe shutdown.
- **Cache-Aligned**: By pinning Virtual SPUs to specific hardware cores, we maintain extreme cache locality, ensuring that regional tile data stays in the L1/L2 caches of the "owning" core.

## The Scheduler: Stackless Coroutines

Rather than relying on a complex external dataflow library, we use **C++20 Stackless Coroutines** as the primary scheduling primitive.

In our architecture, the frame graph is expressed as a series of awaited tasks. This allows us to write "sequential-looking" code that actually executes as a wait-free DAG:

1.  **Job Composition**: Coroutines allow us to suspend execution at dependency boundaries (e.g., waiting for all culling tiles to complete).
2.  **Mailbox Dispatch**: The scheduler "pushes" job handles into the mailboxes of idle Virtual SPUs.
3.  **Non-Blocking / Always-Busy Pipelines**: The combination of `jthreads` and coroutines enables a zero-allocation, lock-free, and non-blocking data pipeline. If a job hits a dependency, it yields control, allowing the Virtual SPU to stay 100% busy on other available work.

## The Dual Backend Strategy

Rather than maintaining a separate "reference" and "experimental" software path, the Angstrom Era adopts a **Binary Backend Architecture**. Since the primary software renderer is built to scale across all available silicon, it becomes the definitive truth and the performance vanguard simultaneously.

1. **`sw_backend` (The Angstrom Core):** The high-performance software renderer mapping frame logic onto a dedicated pool of persistent `jthread` Virtual SPUs. It serves as both the pixel-perfect reference and the many-core performance path.
2. **`vk_backend` (The Silicon Heavyweight):** The state-of-the-art backend targeting modern discrete GPUs via hardware acceleration.

### The Feature Validation Loop
Every new feature (e.g., clustered shading, soft shadows) must survive this pipeline:
*   **Step 1 (Consistency):** Logic and math validated across the **Angstrom Core** (`sw_backend`).
*   **Step 2 (Hardware):** API execution and performance validated on the GPU (`vk_backend`).

## Phase 1: Virtual SPU Infrastructure

1.  **Drop-In `VirtualSPUPool`**: Implement a pool of persistent `std::jthread` workers with lock-free inbound mailboxes.
2.  **Coroutine Job Bridge**: Establish the boilerplate for stackless coroutine handles that can be submitted to the Virtual SPU pool.
3.  **Tile-Based Dispatch**: Rewrite the software rasterizer to be strictly tile-based, where each tile is an independent coroutine task.

## Phase 2: Wait-Free Data Pipelines

1.  **Structured Concurrency**: Replace manual threading barriers with structured coroutine `awaits`.
2.  **Memory Bounding**: Every coroutine task must receive pre-allocated `std::span` outputs, ensuring a strictly zero-allocation pipeline.

## Phase 3: Hardware Scalability Testing

1.  **High-Core-Count Topologies**: Benchmarking the `jthread` scheduler on 64+, 128+, and 256+ core topologies (Threadripper/Epyc) to ensure linear scaling.
2.  **Core Affinity Tuning**: Researching OS-specific core pinning strategies to minimize cross-CCX (Core Complex) latency.

## The Angstrom-Era Guarantee

By aligning the engine logic with persistent `jthreads` and stackless coroutines, `shs-renderer-lib` ensures that when hardware inevitably pivots to thousand-core unified APUs, the engine will instantly and natively scale across all available silicon with the same precision and control once reserved for the PS3 Cell SPUs.
