#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"
# Forwarder retirement (migration step-7 item 4): the legacy shs/domains/ +
# shs/execution/ trees are GONE. Pod gates scan the canonical owner tree
# directly; the renderpath planner seam lives in shs/renderpath/planning/.
pipeline_dir="${lib_root}/include/shs/renderpath/planning"
pod_scan_dirs=()
for _m in app camera geometry input lighting logic render renderpath resources scene sky task platform render/frame render/targets; do
  [[ -d "${lib_root}/include/shs/${_m}" ]] && pod_scan_dirs+=("${lib_root}/include/shs/${_m}")
done
# Purity gates (entropy/platform-IO/expected-vector) guard the VALUE tier only.
# Execution/adapter tiers (renderpath/execution, resources/adapters, app, task,
# platform) legitimately touch time and IO — they are the sanctioned edges.
pod_purity_dirs=()
for _m in camera geometry input lighting logic scene sky render/frame render/targets renderpath/planning resources/storage; do
  [[ -d "${lib_root}/include/shs/${_m}" ]] && pod_purity_dirs+=("${lib_root}/include/shs/${_m}")
done

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

# Reviewed named-module migrations are enforced by manifest, not path exemptions.
# Unmigrated zones retain every existing KDBA check below.
"${PYTHON:-python3}" "${script_dir}/check_header_migration.py"

# Domain direction law: value-tier pod headers must never directly include
# adapter/execution zones (canonical include text). The legacy facade-advisory
# counting retired together with the forwarder tree (migration step-7 item 4).
#
# Sanctioned carve-out (roadmap P1): the renderpath pod's ROOT seam files
# (shs/renderpath/*.hpp — contract re-exports the recipe/plan/capabilities
# spine from renderpath/planning + renderpath/execution). Subdirectories are
# execution-tier and get no carve-out. No other value-tier pod may include an
# adapter/execution zone.
forbidden_execution_include='shs/(renderpath/execution|renderpath/planning|resources/adapters|rhi|app|task|platform|render/software|render/shader|job)/'
for h in $(find "${pod_purity_dirs[@]}" -name '*.hpp' | sort); do
  rel="${h#"${lib_root}/include/"}"
  case "${rel}" in
    shs/renderpath/*)
      # Root seam only — subdirs (planning/, execution/) are execution-tier.
      [[ "$(dirname "${rel}")" == shs/renderpath ]] || continue
      echo "[kdba-boundary] INFO: ${rel} is the renderpath contract seam (P1-sanctioned planning/execution re-exports)"
      continue
      ;;
  esac
  hits="$("${search_cmd[@]}" "#include[[:space:]]*[<\"]${forbidden_execution_include}" "${h}" 2>/dev/null || true)"
  if [[ -n "${hits}" ]]; then
    echo "[kdba-boundary] FAIL: value-tier pod header ${rel} directly includes an adapter/execution zone"
    echo "${hits}"
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
echo "[kdba-boundary] OK: value-tier pod headers carry no direct adapter/execution-zone includes"

# §7.2 rule 5 (roadmap P1.5 DoD): no node-based containers in hot-state zones.
# Hot-state zones are the shared primitive utilities (memory/, containers/,
# frame/) and the domain pods' state headers; cold string-keyed registries in
# the resources + gfx cold registries are flagged INFO until their migration.
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
  "${pod_scan_dirs[@]}" 2>/dev/null || true | wc -l)"
echo "[kdba-boundary] INFO: ${cold_registry_hits} node-container uses in pod zones (cold registries; migrate to FlatMap — tracked in docs/backlog/kdba_conformance_backlog.md)"

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

# Semantic purity (R3 P4.3): no ambient entropy/time, no platform IO in pod zones.
# NOTE: bare 'canvas' deliberately NOT gated (too generic; zero hits today but
# future math comments would false-positive). SDL/fopen are the enforced IO set.
entropy_hits="$(grep -rnE 'rand\(|srand\(|std::chrono|std::time\(|getenv\(|random_' \
  "${pod_purity_dirs[@]}" 2>/dev/null || true)"
if [[ -n "${entropy_hits}" ]]; then
  echo "[kdba-boundary] FAIL: ambient entropy/time in pod zones (dt arrives as input)"
  echo "${entropy_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no ambient entropy/time in pod zones"
fi

unordered_hits="$(grep -rnE 'unordered_(map|set)' \
  $(find "${pod_scan_dirs[@]}" -name '*.gateway.hpp' | sort) 2>/dev/null || true)"
if [[ -n "${unordered_hits}" ]]; then
  echo "[kdba-boundary] FAIL: unordered container in gateway path (hash order breaks replay)"
  echo "${unordered_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no unordered containers in gateway paths"
fi

pio_hits="$(grep -rnE 'SDL_[A-Z]|<SDL2/|fopen\(' \
  "${pod_purity_dirs[@]}" 2>/dev/null || true)"
if [[ -n "${pio_hits}" ]]; then
  echo "[kdba-boundary] FAIL: platform IO token in pod zones (edges only)"
  echo "${pio_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no platform IO tokens in pod zones"
fi

# KDBA Kleisli doctrine (amendment 2026-09-16, Constitution II §8): the monad
# rides at chunk/batch level. A container of per-element expected values
# breaks cache alignment and auto-vectorization — FAIL on sight.
expected_vec_hits="$(grep -rnE 'vector<[[:space:]]*std::expected' \
  "${pod_purity_dirs[@]}" 2>/dev/null || true)"
if [[ -n "${expected_vec_hits}" ]]; then
  echo "[kdba-boundary] FAIL: per-element expected container in pod zones (monad rides at chunk level, §8)"
  echo "${expected_vec_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no per-element expected containers in domains/"
fi

# Closed event facts (Rule 12): events carry enums/ids/quantities, never
# std::string members. Comment mentions are fine; member declarations fail.
stringy_hits="$(grep -rnE 'std::string[[:space:]]+[A-Za-z_][A-Za-z0-9_]*;' \
  $(find "${pod_scan_dirs[@]}" -name '*.event.hpp' | sort) 2>/dev/null || true)"
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
  $(find "${pod_scan_dirs[@]}" -name '*.contract.hpp' | sort) 2>/dev/null || true)"
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
  $(find "${pod_scan_dirs[@]}" -name '*.gateway.hpp' | sort) 2>/dev/null || true)"
if [[ -n "${mono_hits}" ]]; then
  echo "[kdba-boundary] INFO: switch-case sites in gateways (monolith-decomposition backlog — decompose into Kleisli arrows; Constitution II §6.6)"
  echo "${mono_hits}"
else
  echo "[kdba-boundary] OK: no switch-case sites in gateways"
fi

# Event-flow catalog sync (R5b P4.5): every *Event struct in a pod
# event.hpp must appear in docs/pods/EVENT_FLOW.md (tables mirror the
# in-code name tables, which feed the P6 overlay labels).
event_files="$(find "${pod_scan_dirs[@]}" -name '*.event.hpp' | sort)"
event_names=""
if [[ -n "${event_files}" ]]; then
  event_names="$(grep -rhE 'struct (Fsm[A-Za-z0-9_]+|[A-Za-z0-9_]+Event)\b' \
    ${event_files} 2>/dev/null \
    | grep -oE '(Fsm[A-Za-z0-9_]+|[A-Za-z0-9_]+Event)\b' | sort -u || true)"
fi
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
error_files="$(find "${pod_scan_dirs[@]}" \( -name '*.event.hpp' -o -name '*.contract.hpp' \) | sort)"
error_names=""
if [[ -n "${error_files}" ]]; then
  error_names="$(grep -rhE 'enum class [A-Za-z0-9_]*(Error|Rejection|Reason)\b' \
    ${error_files} 2>/dev/null \
    | grep -oE '[A-Za-z0-9_]*(Error|Rejection|Reason)\b' | sort -u || true)"
fi
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
# Pod enumeration over the canonical owner tree (forwarder tree retired):
# <pod>:<canonical home> pairs — frame lives under render/, gfx under targets/.
core4_pod_homes=(
  camera camera geometry geometry input input lighting lighting logic logic
  renderpath renderpath resources resources scene scene sky sky
  frame render/frame gfx render/targets
)
pod_dirs=()
pod_names=()
for (( _pi=0; _pi<${#core4_pod_homes[@]}; _pi+=2 )); do
  _d="${lib_root}/include/shs/${core4_pod_homes[_pi+1]}"
  if [[ -d "${_d}" ]]; then
    pod_names+=("${core4_pod_homes[_pi]}")
    pod_dirs+=("${_d}")
  fi
done

# (1) Non-vacuity: every pod carries the full Core 4 under canonical names.
#     Exception (identity-gateway retirement, migration step 4.5): pods whose
#     command vocabulary is EMPTY by law (§6.1) and which own no applied state
#     carry no gateway file at all — contract/command/event stay, and the
#     pinned empty vocabulary is the drift guard (spec §2.7). Reintroducing a
#     gateway for a listed pod requires removing it from the list together
#     with a real command vocabulary and a §2.2 law amendment.
retired_identity_gateways=(camera geometry gfx lighting resources scene sky)
vacuity=0
for _pi in "${!pod_dirs[@]}"; do
  pod_dir="${pod_dirs[_pi]}"
  pod="${pod_names[_pi]}"
  for role in "${core4_roles[@]}"; do
    if [[ "${role}" == "gateway" ]] \
      && printf '%s\n' "${retired_identity_gateways[@]}" | grep -qx -- "${pod}"; then
      continue
    fi
    if [[ ! -f "${pod_dir}/${pod}.${role}.hpp" ]]; then
      echo "[kdba-boundary] FAIL: pod ${pod} is missing ${pod}.${role}.hpp (Core 4 file law §6.2)"
      vacuity=1
      failed=1
    fi
  done
done
for glob in '*.gateway.hpp' '*.event.hpp' '*.contract.hpp'; do
  n="$(find "${pod_scan_dirs[@]}" -name "${glob}" | wc -l)"
  if [[ "${n}" -eq 0 ]]; then
    echo "[kdba-boundary] FAIL: glob ${glob} matched zero files in pod zones (gate would enforce the wrong tree)"
    vacuity=1
    failed=1
  fi
done
if [[ "${vacuity}" -eq 0 ]]; then
  echo "[kdba-boundary] OK: non-vacuity — ${#pod_dirs[@]} pods carry the Core 4 file law (${#retired_identity_gateways[@]} with retired identity gateways carry contract/command/event only, step 4.5)"
fi

# (2) Paradigm-token ban: the abandoned monolith-reducer vocabulary must not
#     reappear in the pod layer (§6.6: <Pod>Command variant + *Intent tokens,
#     <pod>_gateway entry point).
legacy_hits="$(grep -rnE '\breduce_|\breducers?\b|\b[A-Z][A-Za-z]*Action\b|\.reducer\.hpp|\.action\.hpp' \
  "${pod_scan_dirs[@]}" "${lib_root}/tests" 2>/dev/null || true)"
if [[ -n "${legacy_hits}" ]]; then
  echo "[kdba-boundary] FAIL: legacy reducer/action vocabulary in pod zones (use <pod>_gateway + <Pod>Command/*Intent)"
  echo "${legacy_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no legacy reducer/action vocabulary in pod zones"
fi

# (3) Gateway presence: every pod exposes its <pod>_gateway entry point.
#     Retired identity pods (step 4.5) must NOT expose one — an identity
#     gateway regrowth is a drift violation, not a conformance win.
missing_gw=0
for _pi in "${!pod_dirs[@]}"; do
  pod_dir="${pod_dirs[_pi]}"
  pod="${pod_names[_pi]}"
  if printf '%s\n' "${retired_identity_gateways[@]}" | grep -qx -- "${pod}"; then
    if find "${pod_scan_dirs[@]}" -name "${pod}.gateway.hpp" 2>/dev/null | grep -q .; then
      echo "[kdba-boundary] FAIL: pod ${pod} identity gateway was retired (migration step 4.5); reintroduce only with a real command vocabulary (§2.2 amendment law)"
      failed=1
    fi
    continue
  fi
  gw_file=""
  while IFS= read -r f; do
    if ! grep -q 'Compatibility include: definitions live in' "${f}" 2>/dev/null; then
      gw_file="${f}"; break
    fi
  done < <(find "${pod_scan_dirs[@]}" -name "${pod}.gateway.hpp" | sort)
  [[ -n "${gw_file}" ]] || continue
  defs="$(grep -nE "^[[:space:]]*(inline[[:space:]]+)?[A-Za-z_][A-Za-z_:<>0-9, ]*[[:space:]]+${pod}_gateway[[:space:]]*\\(" \
    "${gw_file}" 2>/dev/null || true)"
  if [[ -z "${defs}" ]]; then
    # Step 4.1 refinement: the gateway entry point may live anywhere in the
    # pod's canonical headers (the input pod's public entry is the
    # translation gateway, input_latch_gateway in value_input_latch.hpp;
    # its application half moved to the app orchestrator).
    pod_entry="$(grep -rnE "^[[:space:]]*(inline[[:space:]]+)?[A-Za-z_][A-Za-z_:<>0-9, ]*[[:space:]]+${pod}_[A-Za-z0-9_]*_gateway[[:space:]]*\(" \
      $(find "${pod_scan_dirs[@]}" -path "*${pod}*" -name '*.hpp' 2>/dev/null) 2>/dev/null | grep -v 'Compatibility include' || true)"
    if [[ -n "${pod_entry}" ]]; then
      echo "[kdba-boundary] INFO: ${pod} gateway entry point lives in a sibling pod header (canonical seam, §6.3)"
    else
      echo "[kdba-boundary] FAIL: ${pod}.gateway.hpp exposes no ${pod}_gateway entry point (§6.3 public gateway)"
      missing_gw=1
      failed=1
    fi
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
kleisli_migrated_pods=(frame input logic renderpath)
kleisli_grandfathered_pods=()
writer_gate=0
for _pi in "${!pod_dirs[@]}"; do
  pod_dir="${pod_dirs[_pi]}"
  pod="${pod_names[_pi]}"
  gw_file=""
  while IFS= read -r f; do
    if ! grep -q 'Compatibility include: definitions live in' "${f}" 2>/dev/null; then
      gw_file="${f}"; break
    fi
  done < <(find "${pod_scan_dirs[@]}" -name "${pod}.gateway.hpp" | sort)
  [[ -n "${gw_file}" ]] || continue
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
while IFS= read -r f; do gw_files+=("${f}"); done < <(find "${pod_scan_dirs[@]}" -name '*.gateway.hpp' -exec grep -L 'Compatibility include: definitions live in' {} \; | sort)
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

# (7) Contract-guardrail seam gate (Constitution II Rule 17 + Forbidden
#     Pattern 7, C3.1 residual closed 2026-09-17): Core 4 seam files state
#     their edge law with the SHS_ contract macros only. Raw assert() and
#     native C++26 contract syntax are banned until the sanctioned C++26
#     switch run (C4.3) rewrites sites mechanically (bridge rule: single-
#     expression, side-effect-free conditions).
seam_contract_files=()
while IFS= read -r f; do seam_contract_files+=("${f}"); done < <(find "${pod_scan_dirs[@]}" \( -name '*.contract.hpp' -o -name '*.command.hpp' -o -name '*.event.hpp' -o -name '*.gateway.hpp' \) | sort)
raw_contract_hits="$(grep -nE '#include[[:space:]]*[<"](cassert|assert\.h)[>"]|\bassert[[:space:]]*\(|\b(pre|post|contract_assert)[[:space:]]*\(' "${seam_contract_files[@]}" 2>/dev/null || true)"
if [[ -n "${raw_contract_hits}" ]]; then
  echo "[kdba-boundary] FAIL: raw assert / native contract syntax in a Core 4 seam file (use SHS_PRE / SHS_POST / SHS_CONTRACT_ASSERT from shs/core/contract_guardrails.hpp, Rule 17)"
  echo "${raw_contract_hits}"
  failed=1
else
  echo "[kdba-boundary] OK: no raw assert / native contract syntax in Core 4 seams (SHS_ macros only, Rule 17)"
fi

# (8) Guardrail placement gate (P1, Rule 17 as amended 2026-09-17):
#     invariants live in the *type*, edge law lives at the *seam*.
#     SHS_PRE/SHS_POST only in *.gateway.hpp/*.contract.hpp;
#     SHS_CONTRACT_ASSERT only in *.contract/command/event.hpp + the
#     pure-leaf-value allowlist (see check_contract_placement.sh).
if ! bash "${script_dir}/check_contract_placement.sh"; then
  failed=1
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
