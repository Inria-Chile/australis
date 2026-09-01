#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: node_preflight.sh --project-dir PATH --run-dir PATH' \
    '  --command-file FILE --universe-file FILE --pending-file FILE' \
    '  --completed-file FILE --active-file FILE' \
    '  --allocation-evidence-file FILE --active-evidence-file FILE' \
    '  --concurrency-evidence-file FILE --requirements-file FILE' \
    '  --probe-file FILE [--require-gpu]'
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

require_value() {
  test "$2" -ge 2 || die "$1 requires a value"
}

require_readable_file() {
  test -f "$1" && test -r "$1" || die "file is not readable: $1"
}

project_dir=''
run_dir=''
command_file=''
universe_file=''
pending_file=''
completed_file=''
active_file=''
allocation_evidence_file=''
active_evidence_file=''
concurrency_evidence_file=''
requirements_file=''
probe_file=''
require_gpu=0

while (($#)); do
  case "$1" in
    --project-dir) require_value "$1" "$#"; project_dir="$2"; shift 2 ;;
    --run-dir) require_value "$1" "$#"; run_dir="$2"; shift 2 ;;
    --command-file) require_value "$1" "$#"; command_file="$2"; shift 2 ;;
    --universe-file) require_value "$1" "$#"; universe_file="$2"; shift 2 ;;
    --pending-file) require_value "$1" "$#"; pending_file="$2"; shift 2 ;;
    --completed-file) require_value "$1" "$#"; completed_file="$2"; shift 2 ;;
    --active-file) require_value "$1" "$#"; active_file="$2"; shift 2 ;;
    --allocation-evidence-file) require_value "$1" "$#"; allocation_evidence_file="$2"; shift 2 ;;
    --active-evidence-file) require_value "$1" "$#"; active_evidence_file="$2"; shift 2 ;;
    --concurrency-evidence-file) require_value "$1" "$#"; concurrency_evidence_file="$2"; shift 2 ;;
    --requirements-file) require_value "$1" "$#"; requirements_file="$2"; shift 2 ;;
    --probe-file) require_value "$1" "$#"; probe_file="$2"; shift 2 ;;
    --require-gpu) require_gpu=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

for required_name in project_dir run_dir command_file universe_file pending_file completed_file active_file \
  allocation_evidence_file active_evidence_file concurrency_evidence_file requirements_file probe_file; do
  test -n "${!required_name}" || die "--${required_name//_/-} is required"
done
test -n "${OAR_JOB_ID:-}" || die 'OAR_JOB_ID is unset; run this only inside a valid OAR job'
test -n "${OAR_NODEFILE:-}" || die 'OAR_NODEFILE is unset; cannot prove node membership'
require_readable_file "$OAR_NODEFILE"

node_fqdn="$(hostname -f 2>/dev/null || hostname)"
node_short="${node_fqdn%%.*}"
case "$node_short" in
  access|frontend|sophia|nancy|grenoble|lille|luxembourg|lyon|nantes|rennes|strasbourg|toulouse)
    die "refusing to run on Grid'5000 frontend/control host: $node_fqdn"
    ;;
esac

allocated=0
while IFS= read -r allocated_node; do
  if [[ "$allocated_node" == "$node_fqdn" || "${allocated_node%%.*}" == "$node_short" ]]; then
    allocated=1
    break
  fi
done <"$OAR_NODEFILE"
test "$allocated" -eq 1 || die "host $node_fqdn is not listed in OAR_NODEFILE"

test -d "$project_dir" || die "project directory is unavailable: $project_dir"
project_dir="$(cd "$project_dir" && pwd -P)"
run_parent="$(dirname "$run_dir")"
test -d "$run_parent" || die "run parent does not exist: $run_parent"
run_parent="$(cd "$run_parent" && pwd -P)"
run_dir="$run_parent/$(basename "$run_dir")"
case "$run_dir" in
  "$project_dir"/*) ;;
  *) die 'run directory must be inside the project directory' ;;
esac
test ! -e "$run_dir" || die "refusing to overwrite existing run directory: $run_dir"

for required in "$command_file" "$universe_file" "$pending_file" "$completed_file" "$active_file" \
  "$allocation_evidence_file" "$active_evidence_file" "$concurrency_evidence_file" \
  "$requirements_file" "$probe_file"; do
  require_readable_file "$required"
done
test -s "$command_file" || die 'command file is empty'
test -s "$universe_file" || die 'task universe is empty'
test -s "$allocation_evidence_file" || die 'allocation evidence is empty'
test -s "$active_evidence_file" || die 'active-task evidence is empty'
test -s "$concurrency_evidence_file" || die 'cross-node concurrency evidence is empty'
test -s "$requirements_file" || die 'resource requirements are empty'
test -s "$probe_file" || die 'small-probe evidence is empty'

scratch_dir="$(mktemp -d)"
run_created=0
cleanup() {
  rm -rf -- "$scratch_dir"
  if test "$run_created" -eq 1; then
    rm -rf -- "$run_dir"
  fi
}
trap cleanup EXIT

# Snapshot every mutable input once; validate and archive only these bytes.
install -m 0600 "$command_file" "$scratch_dir/command.input"
install -m 0600 "$universe_file" "$scratch_dir/universe.input"
install -m 0600 "$pending_file" "$scratch_dir/pending.input"
install -m 0600 "$completed_file" "$scratch_dir/completed.input"
install -m 0600 "$active_file" "$scratch_dir/active.input"
install -m 0600 "$allocation_evidence_file" "$scratch_dir/allocation.input"
install -m 0600 "$active_evidence_file" "$scratch_dir/active-evidence.input"
install -m 0600 "$concurrency_evidence_file" "$scratch_dir/concurrency-evidence.input"
install -m 0600 "$requirements_file" "$scratch_dir/requirements.input"
install -m 0600 "$probe_file" "$scratch_dir/probe.input"
install -m 0600 "$OAR_NODEFILE" "$scratch_dir/oar-nodes.input"
command_file="$scratch_dir/command.input"
universe_file="$scratch_dir/universe.input"
pending_file="$scratch_dir/pending.input"
completed_file="$scratch_dir/completed.input"
active_file="$scratch_dir/active.input"
allocation_evidence_file="$scratch_dir/allocation.input"
active_evidence_file="$scratch_dir/active-evidence.input"
concurrency_evidence_file="$scratch_dir/concurrency-evidence.input"
requirements_file="$scratch_dir/requirements.input"
probe_file="$scratch_dir/probe.input"
oar_nodefile_original="$OAR_NODEFILE"

env_value() {
  local file="$1"
  local key="$2"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print}' "$file"
}
require_env_schema() {
  local file="$1"
  shift
  awk -F= 'NF && $0 !~ /^#/ {if ($1 == "" || seen[$1]++) exit 1}' "$file" || \
    die "duplicate or empty evidence key in $file"
  local key
  for key in "$@"; do
    test "$(grep -c "^${key}=" "$file")" -eq 1 || die "evidence requires exactly one $key in $file"
    test -n "$(env_value "$file" "$key")" || die "evidence value is empty for $key in $file"
  done
}
require_uint() {
  [[ "$2" =~ ^[0-9]+$ ]] || die "$1 must be a nonnegative integer"
}
require_fresh_epoch() {
  local label="$1"
  local checked="$2"
  require_uint "$label" "$checked"
  local now age
  now="$(date -u +%s)"
  age=$((now - checked))
  test "$age" -ge -60 && test "$age" -le 900 || die "$label is outside the 15-minute freshness window"
}

require_env_schema "$allocation_evidence_file" job_id user site node state gpu_indices \
  checked_epoch_utc walltime_seconds remaining_walltime_seconds
test "$(env_value "$allocation_evidence_file" job_id)" = "$OAR_JOB_ID" || \
  die 'allocation evidence job_id does not match OAR_JOB_ID'
test "$(env_value "$allocation_evidence_file" user)" = "$(id -un)" || \
  die 'allocation evidence user does not match current user'
evidence_node="$(env_value "$allocation_evidence_file" node)"
test -n "$evidence_node" || die 'allocation evidence has no node'
[[ "$evidence_node" == "$node_fqdn" || "${evidence_node%%.*}" == "$node_short" ]] || \
  die 'allocation evidence node does not match current host'
case "$(env_value "$allocation_evidence_file" state)" in
  Running|running|RUNNING) ;;
  *) die 'allocation evidence does not record a running job' ;;
esac
require_fresh_epoch checked_epoch_utc "$(env_value "$allocation_evidence_file" checked_epoch_utc)"
require_uint walltime_seconds "$(env_value "$allocation_evidence_file" walltime_seconds)"
require_uint remaining_walltime_seconds "$(env_value "$allocation_evidence_file" remaining_walltime_seconds)"
total_walltime="$(env_value "$allocation_evidence_file" walltime_seconds)"
remaining_walltime="$(env_value "$allocation_evidence_file" remaining_walltime_seconds)"
test "$total_walltime" -gt 0 && test "$remaining_walltime" -gt 0 && \
  test "$remaining_walltime" -le "$total_walltime" || die 'allocation walltime values are inconsistent'

test "$(sed -n '1p' "$concurrency_evidence_file")" = $'job_id\thost\tstate\tchecked_epoch_utc\tnotes' || \
  die 'concurrency evidence has an invalid header'
awk -F'\t' 'NR > 1 {if (NF != 5 || $1 == "" || $2 == "" || $3 !~ /^[Rr][Uu][Nn][Nn][Ii][Nn][Gg]$/ || $4 !~ /^[0-9]+$/) exit 1; key=$1 FS $2; if (seen[key]++) exit 1; rows++} END {exit !(rows > 0)}' \
  "$concurrency_evidence_file" || die 'concurrency evidence rows are invalid or duplicated'
while IFS=$'\t' read -r evidence_job evidence_host evidence_state evidence_epoch evidence_notes; do
  test "$evidence_job" = job_id && continue
  require_fresh_epoch concurrency_checked_epoch "$evidence_epoch"
done <"$concurrency_evidence_file"
awk -F'\t' -v job="$OAR_JOB_ID" -v host="$node_fqdn" 'NR > 1 && $1 == job && $2 == host {found=1} END {exit !found}' \
  "$concurrency_evidence_file" || die 'concurrency evidence omits the current job/node'

test "$(sed -n '1p' "$active_evidence_file")" = $'task_id\tjob_id\thost\tpid_or_lock\tchecked_epoch_utc' || \
  die 'active-task evidence has an invalid header'

normalize_tasks() {
  awk '
    NF && $0 !~ /^[[:space:]]*#/ {
      sub(/^[[:space:]]+/, ""); sub(/[[:space:]]+$/, "");
      if (length($0)) print
    }
  ' "$1" | LC_ALL=C sort -u
}
normalize_tasks "$universe_file" >"$scratch_dir/universe.txt"
normalize_tasks "$pending_file" >"$scratch_dir/pending.txt"
normalize_tasks "$completed_file" >"$scratch_dir/completed.txt"
normalize_tasks "$active_file" >"$scratch_dir/active.txt"
test -s "$scratch_dir/universe.txt" || die 'normalized task universe is empty'
test -s "$scratch_dir/pending.txt" || die 'pending task set is empty; nothing should be launched'

if comm -12 "$scratch_dir/pending.txt" "$scratch_dir/completed.txt" | grep -q .; then
  die 'a task appears in both pending and completed sets'
fi
if comm -12 "$scratch_dir/pending.txt" "$scratch_dir/active.txt" | grep -q .; then
  die 'a task appears in both pending and active sets'
fi
if comm -12 "$scratch_dir/completed.txt" "$scratch_dir/active.txt" | grep -q .; then
  die 'a task appears in both completed and active sets'
fi
LC_ALL=C sort -u "$scratch_dir/pending.txt" "$scratch_dir/completed.txt" "$scratch_dir/active.txt" \
  >"$scratch_dir/classified.txt"
cmp -s "$scratch_dir/universe.txt" "$scratch_dir/classified.txt" || \
  die 'completed, active, and pending sets do not exactly cover the task universe'

awk -F'\t' 'NR > 1 {if (NF != 5 || $1 == "" || $2 == "" || $3 == "" || $4 == "" || $5 !~ /^[0-9]+$/) exit 1; if (seen[$1]++) exit 1; print $1}' \
  "$active_evidence_file" | LC_ALL=C sort -u >"$scratch_dir/active-evidence-tasks.txt" || \
  die 'active-task evidence rows are invalid or duplicated'
cmp -s "$scratch_dir/active.txt" "$scratch_dir/active-evidence-tasks.txt" || \
  die 'active-task evidence does not exactly match active.txt'
while IFS=$'\t' read -r active_task active_job active_host active_owner active_epoch; do
  test "$active_task" = task_id && continue
  require_fresh_epoch active_checked_epoch "$active_epoch"
  awk -F'\t' -v job="$active_job" -v host="$active_host" \
    'NR > 1 && $1 == job && $2 == host {found=1} END {exit !found}' \
    "$concurrency_evidence_file" || die "active task references an unchecked job/node: $active_task"
done <"$active_evidence_file"

require_env_schema "$requirements_file" workload command_uses_project_code workers required_cpu_cores required_ram_mib \
  required_gpu_memory_mib_per_process required_disk_mib required_walltime_seconds \
  ram_headroom_percent vram_headroom_percent
for numeric_key in workers required_cpu_cores required_ram_mib required_gpu_memory_mib_per_process \
  required_disk_mib required_walltime_seconds ram_headroom_percent vram_headroom_percent; do
  require_uint "$numeric_key" "$(env_value "$requirements_file" "$numeric_key")"
done
case "$(env_value "$requirements_file" command_uses_project_code)" in true|false) ;; *) die 'command_uses_project_code must be true or false' ;; esac
require_env_schema "$probe_file" status validated peak_gpu_memory_mib peak_ram_mib duration_seconds output_checksum
test "$(env_value "$probe_file" status)" = passed || die 'small probe did not pass'
test "$(env_value "$probe_file" validated)" = true || die 'small probe lacks successful validation'
for numeric_key in peak_gpu_memory_mib peak_ram_mib duration_seconds; do
  require_uint "$numeric_key" "$(env_value "$probe_file" "$numeric_key")"
done

workers="$(env_value "$requirements_file" workers)"
required_cpu="$(env_value "$requirements_file" required_cpu_cores)"
required_ram="$(env_value "$requirements_file" required_ram_mib)"
required_gpu_mem="$(env_value "$requirements_file" required_gpu_memory_mib_per_process)"
required_disk="$(env_value "$requirements_file" required_disk_mib)"
required_walltime="$(env_value "$requirements_file" required_walltime_seconds)"
ram_headroom="$(env_value "$requirements_file" ram_headroom_percent)"
vram_headroom="$(env_value "$requirements_file" vram_headroom_percent)"
probe_gpu_mem="$(env_value "$probe_file" peak_gpu_memory_mib)"
probe_ram="$(env_value "$probe_file" peak_ram_mib)"
test "$workers" -gt 0 || die 'workers must be positive'
test "$required_cpu" -gt 0 || die 'required_cpu_cores must be positive'
test "$ram_headroom" -ge 15 && test "$ram_headroom" -le 100 && \
  test "$vram_headroom" -ge 15 && test "$vram_headroom" -le 100 || \
  die 'RAM and VRAM headroom percentages must be between 15 and 100'
test "$required_walltime" -le "$(env_value "$allocation_evidence_file" remaining_walltime_seconds)" || \
  die 'required walltime exceeds recorded remaining walltime'
if test "$required_gpu_mem" -gt 0 || test "$probe_gpu_mem" -gt 0; then
  test "$require_gpu" -eq 1 || die 'nonzero GPU memory evidence requires --require-gpu'
fi
if test "$require_gpu" -eq 1; then
  test "$required_gpu_mem" -gt 0 || test "$probe_gpu_mem" -gt 0 || \
    die '--require-gpu requires positive required or measured GPU memory'
fi

if command -v nproc >/dev/null 2>&1; then
  available_cpu="$(nproc)"
elif command -v sysctl >/dev/null 2>&1; then
  available_cpu="$(sysctl -n hw.ncpu)"
else
  die 'cannot determine allocated CPU count'
fi
require_uint available_cpu "$available_cpu"
test "$required_cpu" -le "$available_cpu" && test "$workers" -le "$available_cpu" || \
  die 'CPU requirements exceed allocated CPU count'

if command -v free >/dev/null 2>&1; then
  available_ram="$(free -m | awk '/^Mem:/ {print $7; exit}')"
elif test -r /proc/meminfo; then
  available_ram="$(awk '/^MemAvailable:/ {print int($2/1024); exit}' /proc/meminfo)"
else
  die 'cannot determine available RAM'
fi
require_uint available_ram "$available_ram"
required_ram_with_headroom=$(( (required_ram > probe_ram ? required_ram : probe_ram) * (100 + ram_headroom) / 100 ))
test "$required_ram_with_headroom" -le "$available_ram" || die 'RAM requirement plus headroom exceeds available RAM'

available_disk="$(df -Pm "$project_dir" | awk 'NR == 2 {print $4; exit}')"
require_uint available_disk "$available_disk"
test "$required_disk" -le "$available_disk" || die 'disk requirement exceeds available project filesystem space'

gpu_available=0
: >"$scratch_dir/nvidia-smi.stderr"
if command -v nvidia-smi >/dev/null 2>&1 && \
  nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu \
    --format=csv,noheader,nounits >"$scratch_dir/gpus.csv" 2>"$scratch_dir/nvidia-smi.stderr"; then
  test -s "$scratch_dir/gpus.csv" && gpu_available=1
else
  printf '%s\n' 'no usable NVIDIA GPU detected' >"$scratch_dir/gpus.csv"
fi

cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-unset}"
if test "$require_gpu" -eq 1; then
  test "$gpu_available" -eq 1 || die 'GPU required but nvidia-smi found no usable GPU'
  [[ "$cuda_visible_devices" != 'unset' && -n "$cuda_visible_devices" ]] || \
    die 'GPU work requires explicit nonempty CUDA_VISIBLE_DEVICES'
fi
if [[ "$cuda_visible_devices" != 'unset' && -n "$cuda_visible_devices" ]]; then
  test "$gpu_available" -eq 1 || die 'CUDA_VISIBLE_DEVICES is set but no usable GPU was detected'
  [[ "$cuda_visible_devices" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    die 'CUDA_VISIBLE_DEVICES must contain comma-separated physical GPU indices'
  IFS=',' read -r -a selected_gpus <<<"$cuda_visible_devices"
  test -z "$(printf '%s\n' "${selected_gpus[@]}" | LC_ALL=C sort | uniq -d)" || \
    die 'CUDA_VISIBLE_DEVICES contains duplicate GPU indices'
  evidence_gpu_indices="$(env_value "$allocation_evidence_file" gpu_indices)"
  if test "$require_gpu" -eq 1; then
    test -n "$evidence_gpu_indices" || die 'allocation evidence has no gpu_indices for GPU work'
  fi
  for selected in "${selected_gpus[@]}"; do
    awk -F',' -v wanted="$selected" '{gsub(/[[:space:]]/, "", $1); if ($1 == wanted) found=1} END {exit !found}' \
      "$scratch_dir/gpus.csv" || die "selected GPU index is unavailable: $selected"
    printf '%s\n' "$evidence_gpu_indices" | tr ',' '\n' | grep -Fx -- "$selected" >/dev/null || \
      die "selected GPU is absent from allocation evidence: $selected"
    free_vram="$(awk -F',' -v wanted="$selected" '
      {gsub(/[[:space:]]/, "", $1); if ($1 == wanted) {gsub(/[[:space:]]/, "", $5); print $5; exit}}
    ' "$scratch_dir/gpus.csv")"
    require_uint free_vram "$free_vram"
    required_vram_with_headroom=$(( (required_gpu_mem > probe_gpu_mem ? required_gpu_mem : probe_gpu_mem) * (100 + vram_headroom) / 100 ))
    test "$required_vram_with_headroom" -le "$free_vram" || \
      die "GPU $selected lacks required free VRAM plus headroom"
  done
  if ! nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader >"$scratch_dir/gpu-compute-processes.csv" 2>"$scratch_dir/gpu-processes.stderr"; then
    test "$require_gpu" -ne 1 || die 'cannot verify existing GPU compute processes'
  fi
  for selected in "${selected_gpus[@]}"; do
    selected_uuid="$(awk -F',' -v wanted="$selected" '
      {gsub(/[[:space:]]/, "", $1); if ($1 == wanted) {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); print $2; exit}}
    ' "$scratch_dir/gpus.csv")"
    if awk -F',' -v uuid="$selected_uuid" '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); if ($1 == uuid) found=1} END {exit !found}' \
      "$scratch_dir/gpu-compute-processes.csv"; then
      die "selected GPU $selected already has a compute process"
    fi
  done
elif test "$require_gpu" -eq 1; then
  die 'GPU work requires CUDA_VISIBLE_DEVICES'
fi

mkdir "$run_dir"
run_created=1
printf '%s\n' "owner=$(id -un) job_id=$OAR_JOB_ID pid=$$" >"$run_dir/.g5k-preflight-incomplete"
mkdir "$run_dir/environment" "$run_dir/tasks" "$run_dir/evidence" \
  "$run_dir/logs" "$run_dir/pids" "$run_dir/locks" "$run_dir/checkpoints" "$run_dir/exports"
install -m 0644 "$command_file" "$run_dir/command.txt"
install -m 0644 "$scratch_dir/universe.txt" "$run_dir/tasks/universe.txt"
install -m 0644 "$scratch_dir/pending.txt" "$run_dir/tasks/pending.txt"
install -m 0644 "$scratch_dir/completed.txt" "$run_dir/tasks/completed.txt"
install -m 0644 "$scratch_dir/active.txt" "$run_dir/tasks/active.txt"
install -m 0644 "$allocation_evidence_file" "$run_dir/evidence/allocation.env"
install -m 0644 "$active_evidence_file" "$run_dir/evidence/active-tasks.tsv"
install -m 0644 "$concurrency_evidence_file" "$run_dir/evidence/concurrency.tsv"
install -m 0644 "$requirements_file" "$run_dir/evidence/requirements.env"
install -m 0644 "$probe_file" "$run_dir/evidence/probe.env"
install -m 0644 "$scratch_dir/oar-nodes.input" "$run_dir/evidence/oar-nodes.txt"
install -m 0644 "$scratch_dir/gpus.csv" "$run_dir/environment/gpus.csv"
install -m 0644 "$scratch_dir/nvidia-smi.stderr" "$run_dir/environment/nvidia-smi.stderr"
if test -f "$scratch_dir/gpu-compute-processes.csv"; then
  install -m 0644 "$scratch_dir/gpu-compute-processes.csv" "$run_dir/environment/gpu-compute-processes.csv"
fi

if command -v lscpu >/dev/null 2>&1; then lscpu >"$run_dir/environment/lscpu.txt"; fi
if command -v nproc >/dev/null 2>&1; then
  nproc >"$run_dir/environment/allocated_cpu_count.txt"
elif command -v sysctl >/dev/null 2>&1; then
  sysctl -n hw.ncpu >"$run_dir/environment/allocated_cpu_count.txt"
else
  printf '%s\n' 'CPU count unavailable' >"$run_dir/environment/allocated_cpu_count.txt"
fi
if command -v taskset >/dev/null 2>&1; then taskset -pc $$ >"$run_dir/environment/cpu_affinity.txt"; fi
if command -v free >/dev/null 2>&1; then
  free -h >"$run_dir/environment/memory.txt"
elif test -r /proc/meminfo; then
  sed -n '1,20p' /proc/meminfo >"$run_dir/environment/memory.txt"
else
  printf '%s\n' 'memory inventory unavailable' >"$run_dir/environment/memory.txt"
fi
df -h "$project_dir" >"$run_dir/environment/filesystem.txt"
ps -u "$(id -un)" -o pid,ppid,etime,pcpu,pmem,rss,command >"$run_dir/environment/user_processes.txt"
if test "$gpu_available" -eq 1; then
  nvidia-smi pmon -c 1 >"$run_dir/environment/gpu-processes.txt" 2>&1 || true
fi
printf '%s\n' \
  "PWD=$(pwd -P)" \
  "PATH=${PATH:-unset}" \
  "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset}" \
  "CONDA_PREFIX=${CONDA_PREFIX:-unset}" \
  "VIRTUAL_ENV=${VIRTUAL_ENV:-unset}" \
  "CUDA_VISIBLE_DEVICES=$cuda_visible_devices" \
  "OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}" \
  "MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset}" \
  >"$run_dir/environment/selected_environment.env"
if command -v python >/dev/null 2>&1; then
  { command -v python; python --version; } >"$run_dir/environment/python.txt" 2>&1
  python -m pip freeze >"$run_dir/environment/python-packages.txt" 2>&1 || true
else
  printf '%s\n' 'python unavailable' >"$run_dir/environment/python.txt"
fi

git_commit='not-a-git-worktree'
git_dirty='unknown'
if git -C "$project_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git_commit="$(git -C "$project_dir" rev-parse HEAD)"
  git -C "$project_dir" status --porcelain=v1 >"$run_dir/environment/git-status.txt"
  git -C "$project_dir" diff --binary >"$run_dir/environment/git-diff.patch"
  git -C "$project_dir" ls-files --others --exclude-standard >"$run_dir/environment/git-untracked-files.txt"
  if test -s "$run_dir/environment/git-status.txt"; then
    git_dirty=true
    test "$(env_value "$requirements_file" command_uses_project_code)" = false || \
      die 'project worktree is dirty while command_uses_project_code=true'
  else
    git_dirty=false
  fi
fi
pending_count="$(wc -l <"$run_dir/tasks/pending.txt" | tr -d ' ')"
completed_count="$(wc -l <"$run_dir/tasks/completed.txt" | tr -d ' ')"
active_count="$(wc -l <"$run_dir/tasks/active.txt" | tr -d ' ')"
created_utc="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

printf '%s\n' \
  'schema=g5k-preflight-v3' \
  "created_utc=$created_utc" \
  "user=$(id -un)" \
  "hostname=$node_fqdn" \
  "oar_job_id=$OAR_JOB_ID" \
  "oar_nodefile=$oar_nodefile_original" \
  "project_dir=$project_dir" \
  "run_dir=$run_dir" \
  "git_commit=$git_commit" \
  "git_dirty=$git_dirty" \
  "cuda_visible_devices=$cuda_visible_devices" \
  "gpu_detected=$gpu_available" \
  "available_cpu_cores=$available_cpu" \
  "available_ram_mib=$available_ram" \
  "available_disk_mib=$available_disk" \
  "required_walltime_seconds=$required_walltime" \
  "pending_count=$pending_count" \
  "completed_count=$completed_count" \
  "active_count=$active_count" \
  'preflight_complete=true' \
  >"$run_dir/manifest.env"

(
  cd "$run_dir"
  find . -type f ! -name SHA256SUMS ! -name .g5k-preflight-incomplete -print0 | \
    LC_ALL=C sort -z | xargs -0 sha256sum >SHA256SUMS
)
rm -- "$run_dir/.g5k-preflight-incomplete"
run_created=0
rm -rf -- "$scratch_dir"
trap - EXIT
printf 'Preflight manifest created without launching work: %s\n' "$run_dir/manifest.env"
