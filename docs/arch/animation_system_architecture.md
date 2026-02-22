# Architecture: Animation System

An animation system is a specialized pipeline that handles the deformation of geometry over time. In a modern engine, this involves skeletal hierarchies, skinning, blend trees, and procedural adjustments like IK.

## 1. Skeletal Hierarchy & Skinning
- **Skeleton (Rig)**: A tree structure of "Joints" or "Bones." Each bone has a transform relative to its parent.
- **Skinned Mesh**: Vertex data that includes **Bone IDs** and **Weights** (typically max 4 bones per vertex).
- **Skinning Matrix**: The "Final" transform for a vertex, calculated by blending the current transforms of its influencing bones.

## 2. Animation Data (Tracks)
- **Keyframes**: Stored as Quaternions (Rotation), Vectors (Translation), and Scalars (Scale).
- **Sampling**: Linearly interpolating (LERP) or Spherically interpolating (SLERP) between keyframes based on the current time.
- **Compression**: Techniques like ACL (Animation Compression Library) to reduce memory footprint.

## 3. The Animation Graph (Blend Trees)
Higher-level logic for smooth transitions.
- **Sampling**: Playing a single clip.
- **Liner Blending**: Mixing two clips (e.g., Walk to Run) based on a speed variable.
- **Additives**: Layering a "Recoil" animation on top of an "Idle" animation.
- **State Machines**: Managing logic transitions (e.g., Idle -> Jump -> Fall -> Land).

## 4. Transform & Procedural Animation
Animation isn't just about bones; it's about controlling all aspects of the transform over time.

### A. Spline & Path Following
- **Trajectory Control**: Using Bezier or Hermite splines to move entities (like a butterfly or a floating platform).
- **Camera Tracks**: Defining complex paths for cinematic fly-throughs.
- **Orientation**: Calculating the "Tangent" and "Up" vectors of the spline to ensure the model faces the right direction.

### B. Procedural "Jiggle" & Secondary Motion
- **Oscillators**: Using Sin/Cos waves for subtle breathing or floating effects.
- **Shake/Jiggle**: Adding high-frequency noise to a camera to simulate impacts or hand-held cameras.
- **Spring Physics**: Simple point-mass springs for hair or antenna movement without a full physics engine.

## 5. Cinematics & Cutscenes (The "Timeline")
A system to orchestrate multiple animations over a fixed duration.
- **Track System**: A timeline containing multiple tracks:
    - **Animation Track**: Triggers skeletal clips.
    - **Camera Track**: Swaps active cameras or controls FoV.
    - **Event Track**: Fires Lua events (e.g., spawn a particle at T=5.0s).
    - **Sound Track**: Syncs audio playback to the visuals.

## 6. Morph Targets (Blend Shapes)
For animations that can't be represented by bones (e.g., facial expressions, speech, or muscle bulging).
- **Vertex Deltas**: Storing a second (or third) set of vertex positions.
- **Weighting**: Linearly interpolating between the "Base" mesh and the "Target" mesh based on a scalar weight.
- **GPU Implementation**: Morphing is usually done in the **Compute Shader** or **Vertex Shader** before skinning.

## 7. Ragdoll & Physics Integration
Transitioning from scripted "art" to chaotic "simulated" movement.
- **Physical Bone Mapping**: Linking each skeletal bone to a rigid body collider in the physics engine (like Jolt).
- **The "Flip"**: At a specific moment (e.g., character dies), the `SkinningSystem` stops following animation tracks and starts following the physics-driven colliders.
- **Blended Ragdoll**: Mixing a 50% "Hit Reaction" animation with a 50% physics impulse for realistic staggering.

## 8. ECS Integration
In your "Component-First" design, animation is just another processor.
- **AnimatorComponent**: Stores current time, clip ID, and bone transforms.
- **SkinningSystem**: Computes the final skinning matrices (often in parallel) and uploads them to a GPU buffer (SSBO or UBO).
- **Vertex Shader**: Samples the bone matrices to transform `in_Position` before lighting.

## 6. Workflow: Blender to Engine
- **Export**: Use glTF 2.0 as the intermediate format (it has excellent support for skins and animations).
- **JSON Metadata**: Define the "Animation Controller" in your scene JSON, mapping state names to clip paths.
