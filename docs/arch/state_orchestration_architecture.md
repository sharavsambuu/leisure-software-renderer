# Architecture: State Machine Orchestration (Behavior & Logic)

As the engine grows, you need a "Director" to coordinate the complex interactions between Physics, Animation, AI, and Gameplay. This is achieved through a **State Orchestration System**.

## 1. Orchestration Patterns
Depending on the complexity, you can use several patterns:
- **Hierarchical Finite State Machines (HFSM)**: Best for high-level game states (Menu → Loading → Gameplay → Paused).
- **Behavior Trees (BT)**: Best for complex AI logic where decisions are based on a tree of conditions and actions.
- **Utility AI**: Best for choosing the "best" action based on a score (e.g., "Should I attack or heal?").

## 2. The "Director" System (Blackboards)
To allow different state machines to talk to each other, you need a **Blackboard**.
- **Blackboard**: A shared data container (string/variant maps) where systems can post global context.
- **Example**: The `AISystem` posts `PlayerIsVulnerable = true` to the blackboard. The `OrchestrationSystem` sees this and triggers the `Attack` state in the `AnimationController`.

## 3. Layered State Composition
In your "Component-First" design, the State Machine is just another **Component**.
- **BehaviorComponent**: Stores a handle to a behavior tree or state machine definition.
- **BlackboardComponent**: Stores local entity data (e.g., "CurrentTargetEntityID").

## 4. AI as an Orchestration Problem
You are 100% correct: **AI is the "Intelligence" layer of Orchestration.** It shouldn't be a siloed system; it should be the primary consumer of the Behavior Tree/HFSM logic.

### A. The Input: AI Perception (Senses)
Before orchestrating, the AI needs data. This is handled by **Sensor Components**.
- **Sight Sensor**: A frustum-based check against the `VisibilitySystem`.
- **Hearing Sensor**: Listens for "Noise Events" posted to the Blackboard by Physics or Scripting.
- **Result**: Sensors update the **Blackboard** (e.g., `TargetSpotted = true`).

### B. The Decision: Orchestration (The Brain)
The AI uses the Behavior Tree to evaluate the Blackboard data.
- **Example Node**: `HasPathToPlayer?` -> `TRUE` -> `MoveToPlayerAction`.

### C. The Output: System Dispatch
Once a decision is made, the Orchestrator "directs" other systems:
- **Pathfinding**: Calls `NavigationSystem.FindPath(target_pos)`.
- **Movement**: Calls `MovementProcessor.SetTarget(path_nodes)`.
- **Animation**: Calls `AnimationSystem.SetVariable("Speed", 5.0)`.

### D. Movement & Navigation (NavMesh)
AI doesn't just "think"; it needs to "go" somewhere.

1.  **The NavMesh (The "Map")**: A simplified polygonal representation of all walkable surfaces in the scene.
2.  **Navigation System (The "Navigator")**:
    - **Pathfinding**: Using A* on the NavMesh to find a series of waypoints from point A to B.
    - **Spatial Queries**: "Find the nearest point on the NavMesh" or "Is there a clear line-of-sight on the ground?"
3.  **The Orchestration Link**:
    - In the **Behavior Tree**, a `MoveTo` node calls the `NavigationSystem.FindPath()`.
    - Once a path is found, the node enters a "Running" state and directs the **Movement Processor** (Physics) to follow the waypoints.

### E. Building AI with Lua Trees
You've hit on the most flexible way to build non-player characters: **Hierarchical State Composition**.

1.  **The Spine (C++ Orchestrator)**: Handles the high-level tree traversal and blackboard checks.
2.  **The Meat (Lua States)**: Each "State" or "Action" is a Lua table or function.
    - `OnEnter()`: Triggered when the AI enters the state (e.g., play an "Aggro" sound).
    - `OnUpdate(dt)`: Runs every tick (e.g., steer toward the player).
    - `OnExit()`: Cleanup when leaving.

This allows you to compose a "Tree of State Machines" where a high-level state (e.g., "Combat") is itself a sub-tree of more specific Lua-scripted states (e.g., "Dodge", "Attack", "Reload").

## 5. The Orchestration Lifecycle
1.  **Check Conditions (Processors)**: Systems (Physics, AI) update the Blackboard.
2.  **Evaluate Logic (Orchestrator)**: The `BehaviorSystem` traverses the tree/FSM logic.
3.  **Dispatch Actions (Dispatchers)**: The orchestrator sends commands to other systems:
    - `AnimationSystem.Play("Roar")`
    - `PhysicsSystem.ApplyImpulse(JumpForce)`
    - `LuaSystem.FireEvent("BossPhaseTwo")`

## 5. Workflow: Blender & JSON
- **Blender Binding**: Tag an "Empty" or a "Character" with `behavior: "boss_ai.json"`.
- **Logic Visualizer**: Ideally, you would have a separate graph editor for the state machine rules, which exports to the JSON format.
- **Scripted Leaves**: While the *structure* of the state machine is data-driven, the individual **Leaf Nodes** (the actions) can be written in **Lua**.

## 6. Technical Advantages
- **Decoupling**: The "Animation" doesn't need to know about "Physics"; only the "Orchestrator" knows how to link them.
- **Debugging**: You can pause the game and see exactly which node in the Behavior Tree is currently active.
- **Moddability**: Changing the "Flow" of the game is just a matter of editing a JSON file, not recompiling C++.
