#pragma once

// ============================================================================
// fps/domains/matrix/fps.event.hpp — shared event vocabulary re-export
// Single include site for pods/edges that consume CombatEvents.
// ============================================================================

#include "fps.contract.hpp"

namespace fps::matrix {
    // EventType / CombatEvent live in fps.contract.hpp; this header exists so
    // consumers can express intent ("I consume events") and to keep a stable
    // include name if the vocabulary ever moves to its own file.
    using matrix::CombatEvent;
    using matrix::EventType;
}