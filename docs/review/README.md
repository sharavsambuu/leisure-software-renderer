# Architecture & Code Reviews

This directory contains formal technical, architectural, and performance reviews of the subsystems within the `leisure-software-renderer` repository.

## Index of Reviews

| Review Document | Target Subsystem | Review Date | Focus Areas |
| :--- | :--- | :---: | :--- |
| [**`shs-renderer-lib` Review**](2026-09-18_antigravity_shs_renderer_lib_review.md) | `cpp-folders/src/shs-renderer-lib` | 2026-09-18 | KDBA compliance, C++23 baseline, contract guardrails (Rule 17), DVO backbone, dual-backend parity, rasterizer performance analysis. |
| [**Constitutions & Laws Review**](2026-09-18_antigravity_constitutions_and_laws_review.md) | `docs/spec/` (Constitutions I, II, III & Annexes) | 2026-09-18 | Philosophical coherence, statutory hierarchy, mechanical enforcement gates, DOD/DVO/FP synthesis, and governance critique. |
| [**Independent Review (`deepseek4.1`)**](2026-09-18_deepseek4.1.md) | `cpp-folders/src/shs-renderer-lib` (working practice, post-RP-2) | 2026-09-18 | Auditability vs refactorability, verification freshness (red gate at pristine `HEAD`, flaky gate), doc↔code fidelity, identity-namespace consistency. |

---

## Review Standards & Guidelines

All reviews in this directory follow the project's constitutional laws and guidelines:
1. **Constitution I**: Math, units, Left-Handed (LH) coordinate systems, and NASA/JPL vertical alignment style ([`docs/spec/conventions.md`](../spec/conventions.md)).
2. **Constitution II**: Kleisli Domain Boundary Architecture (KDBA), Core 4 per Domain Value Object (DVO), and contract guardrails placement ([`docs/spec/value_oriented_programming.md`](../spec/value_oriented_programming.md)).
3. **Constitution III**: Data-Oriented Design (DOD), Structure-of-Arrays (SoA), generational handles, and dual-tier memory separation ([`docs/spec/dod_ecs_architecture.md`](../spec/dod_ecs_architecture.md)).
4. **Terminology Law**: Strict usage of "Domain Value Object" / "DVO" across live documentation; retirement of deprecated terms ([`docs/spec/domain_value_object_law.md`](../spec/domain_value_object_law.md)).
5. **Law Budget (gate-at-adoption norm)**: every newly adopted rule must name the mechanical gate that enforces it at adoption time (Constitution II §2.2 item 5); gateless rules are guidelines, not law, until their gate lands ([`docs/spec/value_oriented_programming.md`](../spec/value_oriented_programming.md)).
