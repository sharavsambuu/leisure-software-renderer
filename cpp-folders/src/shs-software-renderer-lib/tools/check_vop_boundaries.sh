#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
lib_root="$(cd "${script_dir}/.." && pwd)"
pipeline_dir="${lib_root}/include/shs/pipeline"

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

if [[ "${failed}" -ne 0 ]]; then
  exit 1
fi

echo "[vop-boundary] all checks passed"
