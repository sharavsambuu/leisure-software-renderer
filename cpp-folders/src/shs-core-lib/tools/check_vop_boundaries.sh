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
    echo "[vop-boundary] FAIL: ${label}"
    echo "${output}"
    failed=1
  else
    echo "[vop-boundary] OK: ${label}"
  fi
}

check_pattern '#include[[:space:]]+[<"]shs/rhi/drivers/' "planner headers include backend driver headers"
check_pattern '#include[[:space:]]+[<"]shs/rhi/sync/' "planner headers include runtime sync headers"
check_pattern 'dynamic_cast[[:space:]]*<' "planner headers use dynamic_cast policy branching"

# --- P0.5 pod-first tree restructure checks ---

# Facade sanity: every migration facade must forward to exactly one existing
# canonical header, which must not be the facade itself (guards self-include).
facade_count=0
for facade in $(find "${lib_root}/include/shs" -name '*.hpp' | sort); do
  rel="${facade#"${lib_root}/include/"}"
  case "${rel}" in
    shs/domains/*|shs/execution/*|shs/core/*|shs/memory/*|shs/containers/*) continue ;;
  esac
  facade_count=$((facade_count + 1))
  target="$(grep -oE '#include[[:space:]]*"[^"]+"' "${facade}" | grep -oE '"[^"]+"' | tr -d '"' | head -1)"
  if [[ -z "${target}" ]]; then
    echo "[vop-boundary] FAIL: facade ${rel} has no forwarding #include"
    failed=1
    continue
  fi
  if [[ ! -f "${lib_root}/include/${target}" ]]; then
    echo "[vop-boundary] FAIL: facade ${rel} forwards to missing ${target}"
    failed=1
    continue
  fi
  if [[ "${target}" == "${rel}" ]]; then
    echo "[vop-boundary] FAIL: facade ${rel} forwards to itself (include cycle)"
    failed=1
    continue
  fi
done
echo "[vop-boundary] OK: ${facade_count} migration facades forward to existing canonical headers"

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
grandfathered_execution_includes=(
  "shs/domains/camera/free_camera.hpp"            # platform edge; P5 Core 4
  "shs/domains/scene/system_processors.hpp"       # pluggable_pipeline seam; P5 legacy-seam audit
  "shs/domains/input/value_actions.hpp"           # app runtime_state; P5 Core 4
  "shs/domains/input/command.hpp"                 # app runtime_state; P5 Core 4
  "shs/domains/sky/skybox_renderer.hpp"           # job/parallel_for edge; P5 Core 4
)
legacy_count=0
for h in $(find "${domains_dir}" -name '*.hpp' | sort); do
  rel="${h#"${lib_root}/include/"}"
  case "${rel}" in
    shs/domains/renderpath/*)
      echo "[vop-boundary] INFO: ${rel} is the renderpath contract seam (P1-sanctioned execution re-exports)"
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
      echo "[vop-boundary] INFO: ${rel} carries grandfathered execution-zone includes (known P5 Core 4 debt)"
    else
      echo "[vop-boundary] FAIL: domain header ${rel} directly includes an execution zone"
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
echo "[vop-boundary] OK: domain headers carry no direct execution-zone includes"
echo "[vop-boundary] INFO: ${legacy_count} legacy-path includes in domains/ resolve via migration facades (canonicalized in P5)"

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
      echo "[vop-boundary] FAIL: node-based container in hot-state zone ${rel} (§7.2 rule 5)"
      echo "${hits}"
      failed=1
    fi
  done
done
if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi
echo "[vop-boundary] OK: zero node-based containers in hot-state zones (§7.2 rule 5)"

cold_registry_hits="$(grep -rnE 'std::(list|map|set|unordered_map|unordered_set)[[:space:]]*<' \
  "${lib_root}/include/shs/domains" 2>/dev/null | wc -l)"
echo "[vop-boundary] INFO: ${cold_registry_hits} node-container uses in domains/ (cold registries; migrate to FlatMap in P5)"

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "[vop-boundary] all checks passed"
