# Action-Based Input: The "Agency" Layer

Most engines make the mistake of checking `if (KeyDown(SPACE))` inside their player code. We're avoiding that by using **Semantic Actions**.

## The Abstraction
The entity doesn't care *who* is telling it to jump. It just has an `InputComponent` that stores a list of currently active actions (Jump, Shoot, Interact).

The source of these actions is the **Agency Layer**:
1. **Local Agency**: Maps your Keyboard/Mouse to actions. (The Human player).
2. **AI Agency**: Logic trees that write "Jump" into the component. (The Bot).
3. **Network Agency**: Deserialized packets from another client. (The Multiplayer Buddy).

## Why bother with this?
- **Multiplayer Proxy**: On your screen, another player is just a normal entity. The only difference is that their `InputComponent` is being driven by a network socket instead of a keyboard.
- **Easy Replays**: Recording a gameplay session is just saving a timestamped stream of actions. Playing it back is just feeding those same actions into the `InputComponent`.
- **Debugging**: Want to test a physics trigger? You can "fake" a jump action through the console without even touching your controller.

## Integration
- **InputSystem**: Reads the raw hardware and updates the `InputComponent`.
- **MovementSystem**: Reads the `InputComponent` and tells Jolt Physics what to do.

This decoupling means you can change your controls (e.g., adding Gamepad support) without breaking your character movement code.

---

> [!IMPORTANT]
> **Action Priority**: If both an AI and a Local Player try to write to the same entity (e.g., in a possession mechanic), the system needs a priority check to decide which "Agency" wins.
