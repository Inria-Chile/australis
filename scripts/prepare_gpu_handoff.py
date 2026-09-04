#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from polarfunc.provenance import sha256_file, utc_now, write_json


MODEL="facebook/esm2_t33_650M_UR50D"; REVISION="08e4846e537177426273712802403f7ba8261b6c"


def selected_fasta(path:Path,targets:dict[str,set[str]],outputs:dict[str,Path])->tuple[dict[str,set[str]],dict[str,int]]:
    handles={key:path.open("w") for key,path in outputs.items()}; seen={key:set() for key in targets}; counts={key:0 for key in targets}; identifier=None; lines=[]
    def emit():
        if identifier is None:return
        sequence="".join(lines)
        for key,ids in targets.items():
            if identifier in ids: handles[key].write(f">{identifier}\n{sequence}\n"); seen[key].add(identifier); counts[key]+=1
    try:
        with path.open() as source:
            for line in source:
                if line.startswith(">"):
                    emit(); identifier=line[1:].split()[0]; lines=[]
                else: lines.append(line.strip())
            emit()
    finally:
        for handle in handles.values(): handle.close()
    for key,ids in targets.items():
        if seen[key]!=ids or counts[key]!=len(ids): raise RuntimeError(f"FASTA selection mismatch {key}: missing={len(ids-seen[key])} records={counts[key]:,} expected={len(ids):,}")
    return seen,counts


def main()->int:
    started=utc_now(); core_path=ROOT/"manifests/benchmark_v1_core_50k.parquet"; matches=sorted((ROOT/"manifests").glob("benchmark_v1_extended_*.parquet"));
    if len(matches)!=1: raise RuntimeError(matches)
    extended_path=matches[0]; core=pd.read_parquet(core_path); extended=pd.read_parquet(extended_path); n=len(extended); gpu_dir=ROOT/"artifacts/gpu"; manifest_dir=ROOT/"manifests/gpu"; report_dir=ROOT/"reports/gpu"; jobs=ROOT/"jobs"
    paths={"core_manifest":manifest_dir/"gpu_manifest_core_50k.parquet","extended_manifest":manifest_dir/f"gpu_manifest_extended_{n}.parquet","transfer":manifest_dir/"gpu_transfer_manifest.tsv","core_fna":gpu_dir/"core_50k.fna","core_faa":gpu_dir/"core_50k.faa","extended_fna":gpu_dir/f"extended_{n}.fna","extended_faa":gpu_dir/f"extended_{n}.faa","handoff":report_dir/"GPU_HANDOFF_UPDATED.md","stage":jobs/"stage_gpu_payload_to_sophia.sh","cache_md":report_dir/"esm2_cache_plan.md","cache_sh":jobs/"prepare_esm2_cache_sophia.sh","genome":report_dir/"genomeocean_defer_decision.md","m10":ROOT/"manifests/corrective/W2C-10/manifest.json","m11":ROOT/"manifests/corrective/W2C-11/manifest.json","m12":ROOT/"manifests/corrective/W2C-12/manifest.json"}
    for path in paths.values():
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    columns=["CDHit_ID","AGC_ID","split_group","split","fraction","function_label","ecology_label","R2_caret","length_nt","protein_length","sequence_sha256","protein_sha256","benchmark_version"]
    core[columns].to_parquet(paths["core_manifest"],index=False,compression="zstd"); extended[columns].to_parquet(paths["extended_manifest"],index=False,compression="zstd")
    targets={"core":set(core.CDHit_ID.astype(str)),"extended":set(extended.CDHit_ID.astype(str))}; nt_seen,nt_counts=selected_fasta(ROOT/"artifacts/sequences/candidate_pool_v1.fna",targets,{"core":paths["core_fna"],"extended":paths["extended_fna"]}); aa_seen,aa_counts=selected_fasta(ROOT/"artifacts/sequences/candidate_pool_v1.faa",targets,{"core":paths["core_faa"],"extended":paths["extended_faa"]})
    payload=[paths[x] for x in ("core_manifest","extended_manifest","core_fna","core_faa","extended_fna","extended_faa")]; transfer=pd.DataFrame([{"path":str(path.relative_to(ROOT)),"size_bytes":path.stat().st_size,"sha256":sha256_file(path)} for path in payload]); transfer.to_csv(paths["transfer"],sep="\t",index=False)
    paths["stage"].write_text("""#!/usr/bin/env bash
set -euo pipefail
SOURCE_ROOT=${SOURCE_ROOT:?Set SOURCE_ROOT to the polarfunc repository}
DEST_ROOT=${DEST_ROOT:?Set DEST_ROOT to Sophia-visible compute-node storage}
MODE=${MODE:-dry-run}
test -f "$SOURCE_ROOT/manifests/gpu/gpu_transfer_manifest.tsv"
mkdir -p "$DEST_ROOT/artifacts/gpu" "$DEST_ROOT/manifests/gpu"
flags=(-a --partial --info=progress2)
test "$MODE" = dry-run && flags+=(--dry-run)
rsync "${flags[@]}" "$SOURCE_ROOT/artifacts/gpu/" "$DEST_ROOT/artifacts/gpu/"
rsync "${flags[@]}" "$SOURCE_ROOT/manifests/gpu/" "$DEST_ROOT/manifests/gpu/"
if test "$MODE" = execute; then
  (cd "$SOURCE_ROOT" && sha256sum $(tail -n +2 manifests/gpu/gpu_transfer_manifest.tsv | cut -f1)) > "$DEST_ROOT/source.sha256"
  (cd "$DEST_ROOT" && sha256sum -c source.sha256)
fi
"""); paths["stage"].chmod(0o750)
    paths["cache_sh"].write_text(f"""#!/usr/bin/env bash
set -euo pipefail
HF_HOME=${{HF_HOME:?Set HF_HOME to Sophia-visible compute-node storage}}
MODE=${{MODE:-dry-run}}
export HF_HOME TRANSFORMERS_CACHE="$HF_HOME/transformers"
command=(huggingface-cli download {MODEL} --revision {REVISION})
printf 'HF_HOME=%s\\nCOMMAND=' "$HF_HOME"; printf '%q ' "${{command[@]}}"; printf '\\n'
test "$MODE" = execute && "${{command[@]}}"
"""); paths["cache_sh"].chmod(0o750)
    paths["cache_md"].write_text(f"# ESM2 cache pre-staging plan\n\nPinned checkpoint: `{MODEL}` revision `{REVISION}`. Run `prepare_esm2_cache_sophia.sh` only inside a Sophia compute allocation with `HF_HOME` on storage visible to Musa. Dry-run is the default. Checkpoint preparation is distinct from GPU validation; no cache path is currently claimed ready.\n")
    paths["genome"].write_text("# GenomeOcean defer decision\n\nStatus remains `BLOCKED_CACHE_ACCESS`. The model is deferred, not scientifically rejected. Existing mean and last-valid-token pooling runners are retained. Retry only after a trusted checkpoint/config location is provided. GenomeOcean is nonblocking for INACH and remains a post-ESM2 NeurIPS DNA priority.\n")
    paths["handoff"].write_text(f"""# Corrective GPU handoff

- Core payload: 50,000 proteins.
- Extended payload: {n:,} proteins.
- Rennes storage was not mounted on the observed Musa node; use the parameterized staging script from an allocated compute node.
- ESM2: `{MODEL}` revision `{REVISION}`.

## Required order
1. Stage and verify SHA256.
2. Prepare checkpoint cache on Sophia-visible storage.
3. Run 100-sequence H100 smoke, mixed precision off.
4. Run 10k throughput test.
5. Run core, then extended, in two deterministic shards.

No foundation-model biological result exists yet.
""")
    checks10={"manifest_ids_match_fasta":bool(all(nt_seen[key]==targets[key] and aa_seen[key]==targets[key] and nt_counts[key]==len(targets[key]) and aa_counts[key]==len(targets[key]) for key in targets)),"core_nested":bool(set(core.CDHit_ID).issubset(set(extended.CDHit_ID))),"manifest_ids_unique":bool(core.CDHit_ID.is_unique and extended.CDHit_ID.is_unique),"checksums_complete":bool(len(transfer)==6 and transfer.sha256.str.len().eq(64).all()),"checksums_verified":bool(all(sha256_file(ROOT/row.path)==row.sha256 for row in transfer.itertuples())),"stage_dry_run_default":"MODE=${MODE:-dry-run}" in paths["stage"].read_text(),"no_hardcoded_sophia_destination":"DEST_ROOT=${DEST_ROOT" in paths["stage"].read_text()}
    cache_text=paths["cache_md"].read_text(); genome_text=paths["genome"].read_text()
    checks11={"pinned_model":MODEL in cache_text,"pinned_revision":REVISION in cache_text,"checkpoint_vs_gpu_validation_explicit":"distinct from GPU validation" in cache_text,"no_cache_claim":"no cache path is currently claimed ready" in cache_text}; checks12={"blocked_status_explicit":"BLOCKED_CACHE_ACCESS" in genome_text,"retry_condition_explicit":"trusted checkpoint" in genome_text}
    if not all((*checks10.values(),*checks11.values(),*checks12.values())): raise RuntimeError("GPU preparation validation failed")
    def out(path:Path):return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    common={"started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv),"inputs":[{"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows} for path in (core_path,extended_path)],"affected_previous_tasks":[],"notes":[]}
    write_json(paths["m10"],{"task_id":"W2C-10","status":"PASS",**common,"parameters":{"extended_n":n},"outputs":[out(path) for path in [*payload,paths["transfer"],paths["handoff"],paths["stage"]]],"validation":{"passed":True,"checks":checks10},"scientific_decision":"GPU_PAYLOAD_FROZEN_STAGING_DESTINATION_PARAMETERIZED"})
    write_json(paths["m11"],{"task_id":"W2C-11","status":"PASS",**common,"parameters":{"model":MODEL,"revision":REVISION},"outputs":[out(paths["cache_md"]),out(paths["cache_sh"])],"validation":{"passed":True,"checks":checks11},"scientific_decision":"CACHE_PREPARATION_SCRIPT_READY_NOT_EXECUTED"})
    write_json(paths["m12"],{"task_id":"W2C-12","status":"PASS",**common,"parameters":{},"outputs":[out(paths["genome"])],"validation":{"passed":True,"checks":checks12},"scientific_decision":"DEFER_GENOMEOCEAN_UNTIL_TRUSTED_CHECKPOINT"})
    print(json.dumps({"W2C-10":checks10,"W2C-11":checks11,"W2C-12":checks12,"extended_n":n},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
