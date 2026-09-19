#pragma once

/*
    SHS RENDERER SAN

    FILE: step_shape.hpp
    MODULE: domains (shared core value utility; R3, ROP-3.2)
    PURPOSE: The Step-shaped concept. R3 ruling (2026-09-18, rop_hardening_todo
             §7): keep per-pod named steps (FsmStep, InputStep, FrameStep,
             RenderPathStep) and compose by SHAPE — a concept, not a concrete
             shared type — with zero renames of working code. A gateway rim
             returns a by-value batch summary; this concept states what every
             such summary must be, and each existing step carries a
             static_assert pinning itself here.

             Deliberately minimal, per the ROP-3.2 pause note ("a subtly wrong
             concept is worse than per-pod types — it must neither reject a
             legal step nor accept a broken one"):
             - is_object: a step is a value, never a reference or function;
             - default-constructible: the zero batch yields the zero step;
             - copy-constructible: the rim returns by value and the pod test
               kit re-runs the same batch over a fresh copy;
             - equality_comparable: value semantics — the kit's replay /
               empty-log proofs rely on operator== (K1.1 spike contract).
             No trivial-copyability demand: a legal future step may carry a
             small value payload (a generation counter today, a diagnostic
             string tomorrow) without failing the shape.
*/

#include <concepts>
#include <type_traits>

namespace shs
{
// namespace-cutover: inline compatibility wrapper (step 7)
    inline namespace core
    {
        template<typename S>
        concept StepShape =
            std::is_object_v<S>                     // a value, not a reference/function
            && std::is_default_constructible_v<S>   // the zero-batch step: S step{}
            && std::is_copy_constructible_v<S>      // returned by value; kit re-runs batches
            && std::equality_comparable<S>;         // kit replay/empty-log proofs need ==
    } // namespace core
} // namespace shs
