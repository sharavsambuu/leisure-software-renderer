#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"
pipeline_dir="${lib_root}/include/shs/execution/pipeline"
domains_dir="${lib_root}/include/shs/domains"

if command -v rg >/dev/null 2>&1; then
  search_cmd=(rg -n)
else
  search_cmd=(grep -R -nE)
fi

planner_files=(
  "${pipeline_dir}/frame_graph.hpp"
  "${pipeline_dir}/render_path_barrier_plan.hpp"
  "${pipeline_dir}/render_path_capabilities.hpp"
  "${pipeline_dir}/render_path_compiler.hpp"
  "${pipeline_dir}/render_path_interfaces.hpp"
  "${pipeline_dir}/render_path_resource_plan.hpp"
  "${pipeline_dir}/render_path_standard_pass_routing.hpp"
)

failed=0

check_pattern() {
  local pattern="$1"
  local label="$2"
  local output
  if output="$("${search_cmd[@]}" "${pattern}" "${planner_files[@]}" 2>/dev/null)"; then
    echo "[kdba-boundary] FAIL: ${label}"
    echo "${output}"
    failed=1
  else
    echo "[kdba-boundary] OK: ${label}"
  fi
}

check_pattern '#include[[:space:]]+[<"]shs/rhi/drivers/' "planner headers include backend driver headers"
check_pattern '#include[[:space:]]+[<"]shs/rhi/sync/' "planner headers include runtime sync headers"
check_pattern 'dynamic_cast[[:space:]]*<' "planner headers use dynamic_cast policy branching"

# --- P0.5 pod-first tree restructure checks ---

# Named-module pilot: a strict leaf dependency allowlist, not a directory exemption.
# Expand only alongside the manifest, tests and the owning module's boundary rules.
canonical_camera="${lib_root}/include/shs/camera/convention.hpp"
legacy_camera="${domains_dir}/camera/convention.hpp"
if [[ ! -f "${canonical_camera}" ]] || ! cmp -s "${legacy_camera}" <(printf '%s\n' \
  '#pragma once' '' \
  '// Compatibility include: definitions live in the camera-owned canonical header.' \
  '#include "shs/camera/convention.hpp"'); then
  echo "[kdba-boundary] FAIL: camera convention compatibility mapping drift"
  exit 1
fi
camera_dependencies="$(grep -E '^[[:space:]]*#[[:space:]]*include' "${canonical_camera}" || true)"
if [[ "${camera_dependencies}" != $'#include <glm/glm.hpp>\n#include <glm/gtc/matrix_transform.hpp>' ]] || \
   grep -qE 'rand\(|srand\(|std::chrono|std::time\(|getenv\(|random_|SDL_|fopen\(' "${canonical_camera}"; then
  echo "[kdba-boundary] FAIL: canonical camera convention is not a pure GLM leaf"
  exit 1
fi
for header in $(find "${lib_root}/include/shs/camera" -type f | sort); do
  if [[ "${header}" != "${canonical_camera}" ]]; then
    echo "[kdba-boundary] FAIL: unregistered canonical camera header ${header}"
    exit 1
  fi
done
if grep -R -nF 'shs/domains/camera/convention.hpp' "${lib_root}/include"; then
  echo "[kdba-boundary] FAIL: library headers use retired camera convention path"
  exit 1
fi
echo "[kdba-boundary] OK: canonical camera leaf and single-hop compatibility header"

# Facade sanity: every migration facade must forward to exactly one existing
# canonical header, which must not be the facade itself (guards self-include).
facade_count=0
for facade in $(find "${lib_root}/include/shs" -name '*.hpp' | sort); do
  rel="${facade#"${lib_root}/include/"}"
  case "${rel}" in
    shs/camera/convention.hpp) continue ;; # validated above, never a facade
    shs/domains/*|shs/execution/*|shs/core/*|shs/memory/*|shs/containers/*|shs/rhi/*) continue ;;
  esac
  facade_count=$((facade_count + 1))
  target="$(grep -oE '#include[[:space:]]*"[^"]+"' "${facade}" | grep -oE '"[^"]+"' | tr -d '"' | head -1)"
  if [[ -z "${target}" ]]; then
    echo "[kdba-boundary] FAIL: facade ${rel} has no forwarding #include"
    failed=1
    continue
  fi
  if [[ ! -f "${lib_root}/include/${target}" ]]; then
    echo "[kdba-boundary] FAIL: facade ${rel} forwards to missing ${target}"
    failed=1
    continue
  fi
  if [[ "${target}" == "${rel}" ]]; then
    echo "[kdba-boundary] FAIL: facade ${rel} forwards to itself (include cycle)"
    failed=1
    continue
  fi
done
echo "[kdba-boundary] OK: ${facade_count} migration facades forward to existing canonical headers"

# Domain direction law: headers under shs/domains/ must never directly include
# execution zones (canonical include text). Legacy-path includes that resolve
# through migration facades are counted as advisory until P5 canonicalization.
#
# Sanctioned carve-out (roadmap P1): shs/domains/renderpath/ is the contract
# seam — its contract re-exports the recipe/plan/capabilities spine from
# execution/pipeline/. No other domain pod may include execution zones.
#
# Known P5 debt (grandfathered INFO, tracked by the Core 4 completeness item):
# these pre-canon domain headers predate the direction law and still reach into
# execution zones directly. New headers must never join this list — the check
# below FAILs on any file not grandfathered here.
forbidden_domains_include='shs/(execution|pipeline|passes|rhi|sw_render|platform|shader|app|job)/'
# Grandfather list evicted R2 (P2.1-P2.4 resolved; any hit below is a hard FAIL).
# shs/rhi/* remains a canonical-continue ONLY as the P3-pending monolith marker.
grandfathered_execution_includes=()
legacy_count=0
for h in $(find "${domains_dir}" -name '*.hpp' | sort); do
  rel="${h#"${lib_root}/include/"}"
  case "${rel}" in
    shs/domains/renderpath/*)
      echo "[kdba-boundary] INFO: ${rel} is the renderpath contract seam (P1-sanctioned execution re-exports)"
      continue
      ;;
  esac
  hits="$("${search_cmd[@]}" "#include[[:space:]]*[<\"]${forbidden_domains_include}" "${h}" 2>/dev/null || true)"
  if [[ -n "${hits}" ]]; then
    grandfathered=0
    for gf in "${grandfathered_execution_includes[@]}"; do
      if [[ "${rel}" == "${gf}" ]]; then
        grandfathered=1
        break
      fi
    done
    if [[ "${grandfathered}" -eq 1 ]]; then
      echo "[kdba-boundary] INFO: ${rel} carries grandfathered execution-zone includes (grandfathered Core 4 debt — tracked in docs/backlog/domain_pod_hardening_backlog.md)"
    else
      echo "[kdba-boundary] FAIL: domain header ${rel} directly includes an execution zone"
      echo "${hits}"
      failed=1
    fi
  fi
  n="$(grep -cE '#include[[:space:]]*[<\"]shs/(pipeline|passes|rhi|sw_render|platform|shader|app|job)/' "${h}" 2>/dev/null || true)"
  legacy_count=$((legacy_count + n))
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
echo "[kdba-boundary] OK: domain headers carry no direct execution-zone includes"
echo "[kdba-boundary] INFO: ${legacy_count} legacy-path includes in domains/ resolve via migration facades (canonicalization tracked in docs/backlog/kdba_conformance_backlog.md)"

# §7.2 rule 5 (roadmap P1.5 DoD): no node-based containers in hot-state zones.
# Hot-state zones are the shared primitive utilities (memory/, containers/,
# frame/) and the domain pods' state headers; cold string-keyed registries in
# domains/resources + domains/gfx are flagged INFO until their P5 migration.
node_container_pattern='std::(list|map|set|unordered_map|unordered_set)[[:space:]]*<'
hot_state_dirs=(
  "${lib_root}/include/shs/memory"
  "${lib_root}/include/shs/containers"
  "${lib_root}/include/shs/frame"
)
for d in "${hot_state_dirs[@]}"; do
  [[ -d "${d}" ]] || continue
  for h in $(find "${d}" -name '*.hpp' | sort); do
    rel="${h#"${lib_root}/include/"}"
    hits="$("${search_cmd[@]}" "${node_container_pattern}" "${h}" 2>/dev/null || true)"
    if [[ -n "${hits}" ]]; then
      echo "[kdba-boundary] FAIL: node-based container in hot-state zone ${rel} (§7.2 rule 5)"
      echo "${hits}"
      failed=1
    fi
  done
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
echo "[kdba-boundary] OK: zero node-based containers in hot-state zones (§7.2 rule 5)"

cold_registry_hits="$(grep -rnE 'std::(list|map|set|unordered_map|unordered_set)[[:space:]]*<' \
  "${lib_root}/include/shs/domains" 2>/dev/null | wc -l)"
echo "[kdba-boundary] INFO: ${cold_registry_hits} node-container uses in domains/ (cold registries; migrate to FlatMap — tracked in docs/backlog/kdba_conformance_backlog.md)"

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

# Semantic purity (R3 P4.3): no ambient entropy/time, no platform IO in domains/.
# NOTE: bare 'canvas' deliberately NOT gated (too generic; zero hits today but
# future math comments would false-positive). SDL/fopen are the enforced IO set.
entropy_hits="$(grep -rnE 'rand\(|srand\(|std::chrono|std::time\(|getenv\(|random_' \
  "${lib_root}/include/shs/domains" 2>/dev/null || true)"
if [[ -n "${entropy_hits}" ]]; then
  echo "[kdba-boundary] FAIL: ambient entropy/time in domains/ (dt arrives as input)"
  echo "${entropy_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no ambient entropy/time in domains/"
fi

unordered_hits="$(grep -rnE 'unordered_(map|set)' \
  $(find "${domains_dir}" -name '*.gateway.hpp' | sort) 2>/dev/null || true)"
if [[ -n "${unordered_hits}" ]]; then
  echo "[kdba-boundary] FAIL: unordered container in gateway path (hash order breaks replay)"
  echo "${unordered_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no unordered containers in gateway paths"
fi

pio_hits="$(grep -rnE 'SDL_[A-Z]|<SDL2/|fopen\(' \
  "${lib_root}/include/shs/domains" 2>/dev/null || true)"
if [[ -n "${pio_hits}" ]]; then
  echo "[kdba-boundary] FAIL: platform IO token in domains/ (edges only)"
  echo "${pio_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no platform IO tokens in domains/"
fi

# KDBA Kleisli doctrine (amendment 2026-09-16, Constitution II §8): the monad
# rides at chunk/batch level. A container of per-element expected values
# breaks cache alignment and auto-vectorization — FAIL on sight.
expected_vec_hits="$(grep -rnE 'vector<[[:space:]]*std::expected' \
  "${lib_root}/include/shs/domains" 2>/dev/null || true)"
if [[ -n "${expected_vec_hits}" ]]; then
  echo "[kdba-boundary] FAIL: per-element expected container in domains/ (monad rides at chunk level, §8)"
  echo "${expected_vec_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no per-element expected containers in domains/"
fi

# Closed event facts (Rule 12): events carry enums/ids/quantities, never
# std::string members. Comment mentions are fine; member declarations fail.
stringy_hits="$(grep -rnE 'std::string[[:space:]]+[A-Za-z_][A-Za-z0-9_]*;' \
  $(find "${domains_dir}" -name '*.event.hpp' | sort) 2>/dev/null || true)"
if [[ -n "${stringy_hits}" ]]; then
  echo "[kdba-boundary] FAIL: std::string member in event fact (closed payloads only, Rule 12)"
  echo "${stringy_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: event facts carry no std::string members"
fi

# KDBA phantom-flag ban (Rule 12, Constitution II §8): persistent PODs carry
# zero transitional flags. Any hit below is a hard FAIL.
phantom_hits="$(grep -rniE 'is_pending|is_trading|is_locked|retry_count|is_validating|is_payment_pending|is_rolling_back' \
  $(find "${domains_dir}" -name '*.contract.hpp' | sort) 2>/dev/null || true)"
if [[ -n "${phantom_hits}" ]]; then
  echo "[kdba-boundary] FAIL: phantom flag in persistent POD contract (transient belongs in SagaContext, Rule 12)"
  echo "${phantom_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no phantom flags in pod contracts"
fi

# KDBA monolith-decomposition tracker (Constitution II §8): switch-case sites
# in gateway.hpp are INFO-tracked, not FAIL — decomposition is the next
# breaking-parts phase. New switch sites should justify themselves.
mono_hits="$(grep -rnE 'switch[[:space:]]*\(' \
  $(find "${domains_dir}" -name '*.gateway.hpp' | sort) 2>/dev/null || true)"
if [[ -n "${mono_hits}" ]]; then
  echo "[kdba-boundary] INFO: switch-case sites in gateways (monolith-decomposition backlog — decompose into Kleisli arrows; Constitution II §6.6)"
  echo "${mono_hits}"
else
  echo "[kdba-boundary] OK: no switch-case sites in gateways"
fi

# Event-flow catalog sync (R5b P4.5): every *Event struct in a pod
# event.hpp must appear in docs/pods/EVENT_FLOW.md (tables mirror the
# in-code name tables, which feed the P6 overlay labels).
event_names="$(grep -rhE 'struct (Fsm[A-Za-z0-9_]+|[A-Za-z0-9_]+Event)\b' \
  $(find "${domains_dir}" -name '*.event.hpp' | sort) 2>/dev/null \
  | grep -oE '(Fsm[A-Za-z0-9_]+|[A-Za-z0-9_]+Event)\b' | sort -u)"
flow_doc="${lib_root}/../../../docs/pods/EVENT_FLOW.md"
drift=0
for ev in ${event_names}; do
  if ! grep -q "${ev}" "${flow_doc}" 2>/dev/null; then
    echo "[kdba-boundary] FAIL: event ${ev} missing from docs/pods/EVENT_FLOW.md"
    drift=1
    failed=1
  fi
done
if [[ "${drift}" -eq 0 ]]; then
  echo "[kdba-boundary] OK: EVENT_FLOW.md covers all pod events"
fi

# KDBA error-rail catalog (Constitution II §8 + Rule 12, hardening 2026-09-16):
# every closed error enum (*Error|*Rejection|*Reason) declared in a pod
# event/contract header must appear in docs/pods/ERROR_FLOW.md — the failure
# rail is a first-class vocabulary too (Rule 11: one error family per context).
error_names="$(grep -rhE 'enum class [A-Za-z0-9_]*(Error|Rejection|Reason)\b' \
  $(find "${domains_dir}" \( -name '*.event.hpp' -o -name '*.contract.hpp' \) | sort) 2>/dev/null \
  | grep -oE '[A-Za-z0-9_]*(Error|Rejection|Reason)\b' | sort -u)"
error_doc="${lib_root}/../../../docs/pods/ERROR_FLOW.md"
edrift=0
for err in ${error_names}; do
  if ! grep -q "${err}" "${error_doc}" 2>/dev/null; then
    echo "[kdba-boundary] FAIL: error enum ${err} missing from docs/pods/ERROR_FLOW.md"
    edrift=1
    failed=1
  fi
done
if [[ "${edrift}" -eq 0 ]]; then
  echo "[kdba-boundary] OK: ERROR_FLOW.md covers all pod error enums"
fi

# ---------------------------------------------------------------------------
# KDBA vocabulary gates (Constitution II §6.6 Pod Identifier Law).
#
# These gates locate their targets BY FILENAME GLOB. If a rename lands without
# updating this script, the globs resolve to an empty file set and GNU grep
# re-scopes `-r` to the working directory — the gate then silently enforces
# against the WRONG tree instead of failing loudly. (Observed for real when
# *.reducer.hpp -> *.gateway.hpp landed: the monolith tracker began reporting
# switch sites in execution/pipeline and rhi/drivers.) The non-vacuity guard
# below makes that failure mode impossible.
# ---------------------------------------------------------------------------
core4_roles=(contract command event gateway)
pod_dirs=()
while IFS= read -r d; do
  pod_dirs+=("${d}")
done < <(find "${domains_dir}" -mindepth 1 -maxdepth 1 -type d | sort)

# (1) Non-vacuity: every pod carries the full Core 4 under canonical names.
vacuity=0
for pod_dir in "${pod_dirs[@]}"; do
  pod="$(basename "${pod_dir}")"
  for role in "${core4_roles[@]}"; do
    if [[ ! -f "${pod_dir}/${pod}.${role}.hpp" ]]; then
      echo "[kdba-boundary] FAIL: pod ${pod} is missing ${pod}.${role}.hpp (Core 4 file law §6.2)"
      vacuity=1
      failed=1
    fi
  done
done
for glob in '*.gateway.hpp' '*.event.hpp' '*.contract.hpp'; do
  n="$(find "${domains_dir}" -name "${glob}" | wc -l)"
  if [[ "${n}" -eq 0 ]]; then
    echo "[kdba-boundary] FAIL: glob ${glob} matched zero files under domains/ (gate would enforce the wrong tree)"
    vacuity=1
    failed=1
  fi
done
if [[ "${vacuity}" -eq 0 ]]; then
  echo "[kdba-boundary] OK: non-vacuity — ${#pod_dirs[@]} pods carry the full Core 4 (contract/command/event/gateway)"
fi

# (2) Paradigm-token ban: the abandoned monolith-reducer vocabulary must not
#     reappear in the pod layer (§6.6: <Pod>Command variant + *Intent tokens,
#     <pod>_gateway entry point).
legacy_hits="$(grep -rnE '\breduce_|\breducers?\b|\b[A-Z][A-Za-z]*Action\b|\.reducer\.hpp|\.action\.hpp' \
  "${domains_dir}" "${lib_root}/tests" 2>/dev/null || true)"
if [[ -n "${legacy_hits}" ]]; then
  echo "[kdba-boundary] FAIL: legacy reducer/action vocabulary in domains/ (use <pod>_gateway + <Pod>Command/*Intent)"
  echo "${legacy_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no legacy reducer/action vocabulary in domains/"
fi

# (3) Gateway presence: every pod exposes its <pod>_gateway entry point.
missing_gw=0
for pod_dir in "${pod_dirs[@]}"; do
  pod="$(basename "${pod_dir}")"
  gw_file="${pod_dir}/${pod}.gateway.hpp"
  [[ -f "${gw_file}" ]] || continue
  defs="$(grep -nE "^[[:space:]]*(inline[[:space:]]+)?[A-Za-z_][A-Za-z_:<>0-9, ]*[[:space:]]+${pod}_gateway[[:space:]]*\\(" \
    "${gw_file}" 2>/dev/null || true)"
  if [[ -z "${defs}" ]]; then
    echo "[kdba-boundary] FAIL: ${pod}.gateway.hpp exposes no ${pod}_gateway entry point (§6.3 public gateway)"
    missing_gw=1
    failed=1
  fi
done
if [[ "${missing_gw}" -eq 0 ]]; then
  echo "[kdba-boundary] OK: every pod gateway exposes its <pod>_gateway entry point"
fi

# (4) Kleisli-shape gate (K6.1, Run A 2026-09-17): once a pod lands the house
#     shape (gateway returns a value Step instead of the retired writer
#     signature `void <pod>_gateway(..., pmr::vector<Event>&)`), the writer
#     shape must not regrow in that pod, and every pod must be either
#     Kleisli-migrated or explicitly grandfathered (P1.1 facade-case
#     precedent). Grandfather list = pods scheduled for Run B/C of the
#     consolidated run plan (docs/backlog/kdba_conformance_backlog.md).
#     Reassessment note (2026-09-17 banner): the original K6.1 wording gated
#     on `inline void reduce_*`, which the §6.6 naming migration already bans
#     outright — the enforceable regrowth vector is the writer SIGNATURE, so
#     this gate targets it instead.
kleisli_migrated_pods=(camera frame geometry gfx input lighting logic renderpath resources scene sky)
kleisli_grandfathered_pods=()
writer_gate=0
for pod_dir in "${pod_dirs[@]}"; do
  pod="$(basename "${pod_dir}")"
  gw_file="${pod_dir}/${pod}.gateway.hpp"
  [[ -f "${gw_file}" ]] || continue
  if printf '%s\n' "${kleisli_migrated_pods[@]}" | grep -qx -- "${pod}"; then
    writer_hits="$(grep -nE "void[[:space:]]+${pod}_gateway[[:space:]]*\(" "${gw_file}" 2>/dev/null || true)"
    if [[ -n "${writer_hits}" ]]; then
      echo "[kdba-boundary] FAIL: ${pod} is Kleisli-migrated but still exposes the writer signature (void ${pod}_gateway)"
      echo "${writer_hits}"
      failed=1
      writer_gate=1
    fi
  elif printf '%s\n' "${kleisli_grandfathered_pods[@]}" | grep -qx -- "${pod}"; then
    : # scheduled for Run B/C; the writer shape is tolerated until its run lands
  else
    echo "[kdba-boundary] FAIL: pod ${pod} is neither Kleisli-migrated nor grandfathered (register it in check_kdba_boundaries.sh)"
    failed=1
    writer_gate=1
  fi
done
if [[ "${writer_gate}" -eq 0 ]]; then
  echo "[kdba-boundary] OK: Kleisli-shape gate — ${#kleisli_migrated_pods[@]}/${#pod_dirs[@]} pod(s) hold the house shape; ${#kleisli_grandfathered_pods[@]} grandfathered"
fi

# (5) Switch-monolith gate (K6.2, Run C): no `switch` over a command/action
#     discriminator inside a pod gateway — dispatch is std::visit + if
#     constexpr over the closed variant (Rule 2 as amended). Pure enum
#     MAPPINGS inside a gateway (renderpath's technique_mode_for /
#     map_rejection / apply_runtime_toggle) are not action dispatch and stay
#     legal; the discriminator pattern (switch over .type/.kind) is what is
#     banned.
gw_files=()
while IFS= read -r f; do gw_files+=("${f}"); done < <(find "${domains_dir}" -name '*.gateway.hpp' | sort)
monolith_hits="$(grep -nE 'switch[[:space:]]*\([[:space:]]*[A-Za-z_]+(\.type|\.kind)[[:space:]]*\)' "${gw_files[@]}" 2>/dev/null || true)"
if [[ -n "${monolith_hits}" ]]; then
  echo "[kdba-boundary] FAIL: switch over a command/action discriminator in a gateway (Rule 2 amended — use std::visit over the closed variant)"
  echo "${monolith_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no switch-monolith dispatch in pod gateways (K6.2)"
fi

# (6) Silent-drop gate (K6.3, Run C): the team answered K3.2 with FACTS — a
#     consumed command never emits nothing. The consume-and-skip shape
#     (`continue;` inside a gateway) is therefore banned: a named arrow emits
#     its fact, or returns (the documented legacy mirrors live inside arrows,
#     never in the assembly loop).
drop_hits="$(grep -n 'continue;' "${gw_files[@]}" 2>/dev/null || true)"
if [[ -n "${drop_hits}" ]]; then
  echo "[kdba-boundary] FAIL: consume-and-skip 'continue' in a gateway (K3.2 house answer is FACTS — emit the fact or return from the arrow)"
  echo "${drop_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no silent-drop 'continue' in pod gateways (K6.3)"
fi

# Final enforcement gate (2026-09-16 hardening): every FAIL above must fail
# the script. Negative-test proven: the tail-section gates (platform IO,
# entropy, expected-vector, stringy events, phantom flags, event/error catalog
# drift) previously printed FAIL but exited 0 — CI-binding now.
if [[ "${failed}" -ne 0 ]]; then
  echo "[kdba-boundary] boundary violations detected"
  exit 1
fi

echo "[kdba-boundary] all checks passed"
