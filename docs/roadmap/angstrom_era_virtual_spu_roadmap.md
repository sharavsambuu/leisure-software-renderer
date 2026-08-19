

---

# Roadmap: The Angstrom-Era Software Renderer (Virtual SPU)

This roadmap outlines the visionary, long-term trajectory for **`shs-renderer-lib`’s** software backend. It details the transition to a massive, many-core **"Virtual SPU"** architecture powered by C++20 `std::jthread`, wide SIMD vector pipelines, Data-Oriented Design (DOD), and a structured coroutine DAG, preparing the engine for the "Angstrom Era" of hardware.

---

## The Vision: Return of the PS3 Cell SPUs
In the early 2000s, the PlayStation 3 introduced the Cell Broadband Engine. It was notoriously difficult to program because developers could no longer rely on a single, monolithic CPU thread. They had to explicitly marshal data to and from tiny, localized Synergistic Processing Elements (SPUs) and dispatch pure compute jobs with software-managed scratchpads (Local Stores).

The hardware industry is rapidly iterating toward a modern, PC-scale equivalent of this architecture. In the "Angstrom Era," the line between CPU and GPU blurs entirely through unified-memory Systems on a Chip (SoCs) where hundreds to thousands of cores share a massive pool of global RAM.

In this future, the software renderer is reborn as a **Massive Virtual SPU Array**. Each core is modeled not as a transient OS task, but as a persistent, cooperative **Virtual SPU** (represented by a `std::jthread` and dedicated SIMD vector registers) that "occupies" a physical core, manages its own cache-line-aligned local scratchpad, and executes work dispatched through a zero-allocation, coroutine-based job system.

---

## The Philosophy: General-Purpose vs. ASICs
A fundamental pillar of the Angstrom Era is the bet that **Flexible, Vector-Driven General Silicon beats Fixed-Function Silicon (ASICs) at scale**.

* **Software-Defined Rendering:** GPUs rely on fixed ASICs (rasterizers, samplers, fixed blend units) that are "frozen" in silicon. By using persistent Virtual SPUs on high-throughput vector cores, we gain 100% flexibility to invent new rendering paradigms (such as Visibility Buffers, software meshlet culling, and custom subpixel rasterization) without hardware restrictions.
* **Zero-Copy Engine Integration:** Traditional GPU pipelines suffer from heavy marshalling, driver translation layers, and API command-ring stalls. By running the renderer on the same unified silicon as the game's simulation, the Virtual SPUs read contiguous Entity Component System (ECS) arrays directly with **zero translation layers and zero memory copies**.
* **Deterministic Frame Pacing:** Eliminating GPU drivers and hardware state machine black-boxes eliminates runtime pipeline compilation hitches, driver memory defragmentation stalls, and frame pacing jitter. Every frame executes with microsecond-level predictability.
* **Unified Resource Pool:** Instead of hitting "black box" bottlenecks in a fixed-function pipeline, the Virtual SPU model treats all silicon as a unified pool that can be rebalanced dynamically (e.g., reallocating 80% of workers to geometry culling or physics simulation if that stage becomes the frame bottleneck).
* **ASIC-Like Performance through VOP & DOD:** By combining Data-Oriented Design (DOD) with Vector-Oriented Programming (VOP) across native vector lanes (AVX-512, ARM SVE2), we achieve the raw throughput of specialized hardware while retaining the full expressive power of C++.
* **Always-Busy / Non-Blocking Execution:** Adopting the id Tech philosophy where workers never "wait." A Virtual SPU never hits a blocking OS barrier; it either processes a stream of micro-tasks or yields at pass boundaries to immediately pick up the next available execution piece.

---

## The Technical Anchor: Coherence and Ownership
Even in a world of abundant cores, the physical constraint remains the **Memory Hierarchy and Interconnect Bandwidth**. To prevent 1,000 cores from collapsing into a Coherence Storm, the architecture enforces four strict disciplines:

1. **Ownership:** Each core (Virtual SPU) has absolute, exclusive ownership over its regional data domain (a specific screen tile and its local depth/visibility buffer). Cross-core writes are strictly forbidden.
2. **Isolation (Emulated Local Stores):** Every Virtual SPU is allocated a dedicated, cache-line-aligned ($64\text{ B}/128\text{ B}$) scratchpad arena. Work is executed purely within the core's private L1/L2 cache. Finished frame tiles bypass intermediate cache thrashing via **Non-Temporal Streaming Stores** (`_mm512_stream_*` / `STNP`).
3. **Locality & SoA Layouts:** Data is strictly marshaled as Structure-of-Arrays (SoA) and Array-of-Structures-of-Arrays (AoSoA), ensuring every byte pulled into an L1 cache line is directly consumed by SIMD vector instructions.
4. **Deterministic Scheduling:** The frame DAG ensures that data flows through the system in a predictable, contention-free sequence with zero false sharing.

---

## The Execution Unit: Virtual SPUs (`std::jthread`)
The Virtual SPU is the heart of the execution. We model these units using C++20 `std::jthread` combined with wide SIMD execution abstractions. Unlike traditional threads, these units are:

* **Singular & Unified:** **Exactly one persistent thread pool exists** (1 `std::jthread` per hardware core). There are no competing secondary pools for coroutines or background tasks, eliminating CPU oversubscription and OS context switches.
* **Persistent:** A Virtual SPU "lives" for the entire duration of the engine instance, completely eliminating thread-creation overhead and maintaining warm cache residency for owned scratchpad arenas.
* **Vector-Driven:** Each Virtual SPU processes geometric clusters, edge equations, and shading samples across 8-wide, 16-wide, or 32-wide SIMD lanes using Vector-Oriented Programming.
* **Cooperative:** Using `std::stop_token` to handle graceful interruption and state-safe shutdown.
* **Cache-Aligned & Pinned:** By pinning Virtual SPUs to specific hardware cores (and respecting NUMA / Performance vs. Efficient core topologies), we maintain extreme locality, ensuring that regional tile data stays in the L1/L2 caches of the "owning" core.

---

## The Scheduler: Stackless Coroutines on a Unified Job Core
Rather than creating separate execution engines, the scheduler uses a **Layered Two-Tier Model** where coroutines serve as an ergonomic syntax layer directly on top of the unified work-stealing job pool:

* **The Unified Job Core:** All executable units are represented as lightweight `Job` payloads (`void(*execute)(void*)`). A coroutine resumption (`coroutine_handle<>::resume`) is fundamentally treated as just another standard `Job` pushed into the work-stealing queue.
* **Two-Tier Layering:**
  * **Macro Tier (Coroutine Frame DAG):** High-level rendering passes are written as sequential `co_await` operations (`co_await DepthPass()`, `co_await VisibilityPass()`).
  * **Micro Tier (Lock-Free Work Stealing):** When an awaited pass executes, it fans out hundreds of raw micro-tasks (tile chunks, meshlets) to the persistent `VirtualSPUPool`. An atomic latch tracks progress; once the counter reaches zero, the completing worker submits the parent coroutine's resumption handle back into the unified queue.
* **Non-Blocking / Always-Busy Execution:** Workers never block on mutexes or OS conditionals. When a worker drains its local queue, it steals micro-tasks from neighboring workers, ensuring 100% vector unit saturation.

---

## The Modern Rendering Paradigm: Meshlets & Visibility Buffer
To maximize the throughput of our Virtual SPUs, the engine adopts a modern, bandwidth-efficient rendering pipeline:

* **Cluster-Based Geometry (Meshlets):** Geometries are partitioned into fixed-size meshlets (64 vertices, 128 triangles), allowing full SIMD frustum culling, normal-cone backface culling, and Hierarchical-Z (HZB) occlusion rejection *before* rasterization.
* **Fixed-Point Tile Rasterization:** Rasterization is performed using $16.16$ fixed-point arithmetic with SIMD edge functions, evaluating full pixel blocks simultaneously with subpixel precision.
* **Compact Visibility Buffer (V-Buffer):** Instead of allocating heavy G-Buffers across main memory, the rasterizer writes a compact 64-bit visibility payload (Meshlet ID, Primitive ID, Barycentrics) into the tile's L1/L2 scratchpad.
* **Tile-Based In-Cache Shading:** Shading, material evaluation, and lighting are evaluated strictly once per visible pixel directly within the tile’s local cache before streaming the final resolved color to RAM.

---

## The Dual Backend Strategy
Rather than maintaining a separate "reference" and "experimental" software path, the Angstrom Era adopts a **Binary Backend Architecture**. Since the primary software renderer is built to scale across all available silicon, it becomes both the definitive truth and the performance vanguard:

* **`sw_backend` (The Angstrom Core):** The high-performance software renderer mapping frame logic onto a dedicated pool of persistent vector `std::jthread` Virtual SPUs. It serves as both the pixel-perfect reference and the many-core performance path.
* **`vk_backend` (The Silicon Heavyweight):** The state-of-the-art backend targeting modern discrete GPUs via hardware acceleration and Vulkan 1.3+ compute/mesh pipelines.

---

## The Feature Validation Loop
Every new feature (e.g., clustered shading, soft shadows, visibility passes) must survive this pipeline:

* **Step 1 (Consistency):** Logic, math, and bit-exact fixed-point rasterization validated across the Angstrom Core (`sw_backend`).
* **Step 2 (Hardware):** API execution, memory layout, and hardware baseline validated on the GPU (`vk_backend`).

---

## Phased Implementation Roadmap

### Phase 1: Unified Virtual SPU Infrastructure & Local Stores
- [ ] Implement the singular, persistent `std::jthread` worker pool (`VirtualSPUPool`) with hardware core pinning and NUMA awareness.
- [ ] Build the underlying zero-allocation lock-free work-stealing job deque (`Job{ void(*)(void*), void* }`).
- [ ] Build thread-local, cache-line-aligned ($64\text{ B}/128\text{ B}$) linear **Scratchpad Allocators** to emulate SPU Local Stores.
- [ ] Establish the cross-platform SIMD vector abstraction (AVX2 / AVX-512 / ARM NEON & SVE2).
- [ ] Implement Non-Temporal Streaming Store resolve routines (`_mm512_stream_*` / `STNP`) to bypass cache pollution.

### Phase 2: Coroutine Bridge, DOD Ingestion & Meshlets
- [ ] Implement the C++20 Coroutine Awaiter Bridge (`ParallelForBatch` / atomic completion latches that submit `coroutine_handle::resume` as a standard `Job`).
- [ ] Implement zero-copy SoA / AoSoA data streamers bridging ECS components directly to Virtual SPUs.
- [ ] Implement geometry pre-clustering into uniform **Meshlets** (64v / 128p).
- [ ] Build SIMD cluster-level frustum, normal-cone, and Hierarchical-Z (HZB) occlusion culling.

### Phase 3: Fixed-Point Rasterizer & Visibility Buffer
- [ ] Build a 16-wide fixed-point SIMD tile rasterizer with subpixel precision.
- [ ] Implement the compact 64-bit Visibility Buffer output pipeline entirely within thread-local scratchpads.
- [ ] Implement in-cache Tile-Based Deferred Shading (TBDR) and SIMD-accelerated texture block decompression (BCn / ASTC).
- [ ] Integrate C++23 `std::mdspan` for Morton (Z-curve) swizzled texture and framebuffer access.

### Phase 4: Hardware Scalability Testing & Validation
- [ ] Benchmark the Virtual SPU scheduler across 32, 64, 128, and 256+ core topologies (Threadripper / EPYC / High-Density APUs) to ensure linear scaling.
- [ ] Tune core affinity for heterogeneous architectures (Performance cores for SIMD rasterization vs. Efficient cores for async tasks/culling).
- [ ] Run cross-backend automated image regression tests between `sw_backend` and `vk_backend`.

---

## The Angstrom-Era Guarantee
By aligning the engine logic with persistent `std::jthread` Virtual SPUs, a singular unified work-stealing core, explicit Local Store memory discipline, wide SIMD execution, Data-Oriented Design, and stackless coroutine DAGs, **`shs-renderer-lib`** ensures that when hardware pivots to thousand-core unified APUs, the engine will scale proportionally with core availability under bounded memory constraints—bringing the precision, control, and throughput once reserved for the PS3 Cell SPUs into the modern era of computing.