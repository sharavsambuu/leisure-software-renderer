# Event System: The Nervous System

While ECS is great for constant, per-frame updates (like gravity or rendering), an **Event System** handles one-off signals that happen sporadically. It is the "Nervous System" that tells distant parts of the engine: *"Something just happened!"*

## 1. Why do we need it?
Imagine an `Explosion` entity.
- **Without Events**: Every system in the engine would have to check every frame: *"Was there an explosion near me?"* (Very slow).
- **With Events**: The `PhysicsSystem` detects the explosion and broadcasts an `OnExplosion` event. Only systems that care (like `SoundSystem` or `ParticleSystem`) listen and respond.

## 2. Two Types of Events

### A. Immediate Events (The Callback)
Used for direct, local reactions.
- **Example**: A `Button` component fires an `OnClick` callback.
- **Pros**: Fast and direct.
- **Cons**: Can create "Spaghetti code" if too many things are clicking each other directly.

### B. Queued Events (The Message Bus)
Used for decoupling complex systems.
- **Example**: The `CombatSystem` posts a `DamageEvent(target, amount)`. It doesn't know who is listening.
- **The Flow**:
    1. A script or system "Publishes" an event to a central queue.
    2. The **Event System** stores it.
    3. During the **Event Phase** of the frame, all "Subscribers" receive the message.
- **The Win**: The `CombatSystem` doesn't need to know that a `UI_System` exists to update the health bar—it just shouts into the void, and the UI hears it.

## 3. ECS-Native Events (Tag Components)
In a pure ECS engine, an event can simply be a **Temporary Component**.
- **The Pattern**: To "Damage" an entity, you add a `PendingDamageComponent` to it.
- **The System**: A `DamageSystem` looks for that tag, applies the math to the `HealthComponent`, and then **deletes** the tag.
- **Benefit**: This makes events perfectly serializable and thread-safe, just like any other data.

## 4. Lua Integration
The Event System is the primary way Lua talks to C++ without knowing its internals:
- **C++ to Lua**: When a player dies, C++ broadcasts `player_died`. All Lua scripts subscribed to that name wake up and run their logic.
- **Lua to C++**: A script can fire a `request_save` event, which the C++ **Foundation Layer** picks up and handles.

---

> [!TIP]
> **Performance Rule**: Don't use events for things that happen every frame (like movement). Use per-frame System updates for that. Use events for **Transitions** (Start, Stop, Die, Win, Collide).
