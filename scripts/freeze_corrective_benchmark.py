#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from polarfunc.benchmark import add_matching_bins, assign_split_groups, benchmark_join_query, coarsened_exact_match, stable_hash
from polarfunc.provenance import sha256_file, utc_now, write_json


def atomic_parquet(frame:pd.DataFrame,path:Path)->None:
    temporary=path.with_suffix(path.suffix+".tmp"); frame.to_parquet(temporary,index=False,compression="zstd"); temporary.replace(path)


def protein_hashes(path:Path)->pd.DataFrame:
    rows=[]; identifier=None; sequence=[]
    def emit():
        if identifier is not None:
            value="".join(sequence); rows.append((identifier,hashlib.sha256(value.encode()).hexdigest()))
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                emit(); identifier=line[1:].split()[0]; sequence=[]
            else: sequence.append(line.strip())
    emit(); return pd.DataFrame(rows,columns=["CDHit_ID","protein_sha256"])


def quotas(total:int)->dict[str,int]:
    train=int(total*0.70); validation=int(total*0.15); return {"train":train,"validation":validation,"test":total-train-validation}


def derive_extended_per_stratum(counts:pd.DataFrame,cap:int=12_500)->int:
    minimum_total=int(counts.groupby("stratum").n.sum().min())
    for total in range(min(cap,minimum_total),6_249,-1):
        target=quotas(total)
        if all(int(row.n)>=target[row.split] for row in counts.itertuples()): return total
    raise RuntimeError("Strict group split cannot support the 6,250-per-stratum core")


def sample(frame:pd.DataFrame,per_stratum:int)->pd.DataFrame:
    target=quotas(per_stratum); selected=[]
    for (stratum,split),group in frame.groupby(["stratum","split"],sort=True):
        ordered=group.assign(sample_hash=group.CDHit_ID.astype(str).map(lambda x:stable_hash(x,2026))).sort_values("sample_hash")
        selected.append(ordered.head(target[split]))
    return pd.concat(selected,ignore_index=True).drop(columns="sample_hash")


def leakage(frame:pd.DataFrame)->dict[str,int|bool]:
    def crossings(column:str)->int: return int(frame.groupby(column).split.nunique().gt(1).sum())
    return {"CDHit_overlap":crossings("CDHit_ID"),"AGC_overlap":crossings("AGC_ID"),"MMseqs_group_overlap":crossings("split_group"),"nucleotide_hash_overlap":crossings("sequence_sha256"),"protein_hash_overlap":crossings("protein_sha256"),"target_missing":int(frame[["fraction","function_label","ecology_label","R2_caret"]].isna().any(axis=1).sum()),"all_splits":bool(set(frame.split)=={"train","validation","test"}),"all_eight_strata":bool(frame.stratum.nunique()==8)}


def main()->int:
    started=utc_now(); manifest_dir=ROOT/"manifests"; report_dir=ROOT/"reports/corrective"; corrective=ROOT/"artifacts/corrective"
    strict=manifest_dir/"strict_matched_pool_v1.parquet"; core=manifest_dir/"benchmark_v1_core_50k.parquet"; split_path=manifest_dir/"split_manifest_v1.parquet"; protein_index=corrective/"candidate_protein_hashes.parquet"; policy=report_dir/"benchmark_v1_policy.md"; balance=report_dir/"benchmark_v1_balance.md"; audit_path=report_dir/"benchmark_v1_leakage.json"; retention_path=report_dir/"matching_retention_v1.tsv"; manifest_path=ROOT/"manifests/corrective/W2C-06/manifest.json"
    fixed=[strict,core,split_path,protein_index,policy,balance,audit_path,retention_path,manifest_path]
    for path in fixed:
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    paths={"groups":ROOT/"artifacts/homology/candidate_split_groups.parquet","qc":ROOT/"artifacts/sequences/candidate_translation_qc.parquet","candidates":ROOT/"manifests/candidate_pool_v1.parquet","abundance":ROOT/"artifacts/wave2/agc_abundance_covariates_combined.parquet","taxonomy":ROOT/"artifacts/wave2/agc_taxonomy.parquet"}
    con=duckdb.connect(); con.execute("SET threads=48"); con.execute("SET memory_limit='256GB'")
    frame=con.execute(benchmark_join_query(paths)).fetchdf(); binned,bins=add_matching_bins(frame,quantiles=2,top_phyla=10); matched=coarsened_exact_match(binned,bins,seed=20260830)
    if len(matched)!=139_316: raise RuntimeError(f"Strict matched pool changed unexpectedly: {len(matched):,}")
    groups=assign_split_groups(matched,seed=2026); lookup=groups.set_index("split_group").split; matched["split"]=matched.split_group.map(lookup)
    hashes=protein_hashes(ROOT/"artifacts/sequences/candidate_pool_v1.faa"); atomic_parquet(hashes,protein_index); matched=matched.merge(hashes,on="CDHit_ID",how="left",validate="one_to_one")
    thresholds=json.loads((ROOT/"config/benchmark_v1_low_thresholds.json").read_text())["thresholds"]; matched["low_threshold"]=[thresholds[f"{f}::{l}"] for f,l in zip(matched.fraction,matched.function_label)]
    matched["strict_known"]=matched.function_label.eq("strict_known"); matched["strict_unknown"]=matched.function_label.eq("strict_unknown"); matched["function_subset"]=matched.function_label; matched["benchmark_version"]="v1_strict_joint_2026-08-30"
    counts=matched.groupby(["stratum","split"]).size().rename("n").reset_index(); extended_per=derive_extended_per_stratum(counts); extended_total=8*extended_per; extended=sample(matched,extended_per); core_frame=sample(matched,6_250)
    if not set(core_frame.CDHit_ID).issubset(set(extended.CDHit_ID)): raise RuntimeError("Core is not nested in extended")
    extended_path=manifest_dir/f"benchmark_v1_extended_{extended_total}.parquet"
    if extended_path.exists(): raise FileExistsError(extended_path)
    keep=["CDHit_ID","AGC_ID","fraction","strict_known","strict_unknown","function_subset","R2_caret","ecology_label","low_threshold","split_group","split","sequence_sha256","protein_sha256","length_nt","GC_fraction","protein_length","prevalence","mean_abundance","Domain","Phylum","domain_conflict","benchmark_version","function_label","stratum","mmseqs_representative","cluster_size"]
    keep=[column for column in keep if column in matched]
    atomic_parquet(matched[keep].sort_values("CDHit_ID"),strict); atomic_parquet(core_frame[keep].sort_values("CDHit_ID"),core); atomic_parquet(extended[keep].sort_values("CDHit_ID"),extended_path); atomic_parquet(matched[keep].sort_values("CDHit_ID"),split_path)
    audits={"strict_matched_pool":leakage(matched),"core_50k":leakage(core_frame),f"extended_{extended_total}":leakage(extended),"core_nested_in_extended":bool(set(core_frame.CDHit_ID).issubset(set(extended.CDHit_ID))),"sizes":{"strict":len(matched),"core":len(core_frame),"extended":len(extended),"extended_per_stratum":extended_per},"split_quotas":{"core":quotas(6250),"extended":quotas(extended_per)}}
    zero_keys=("CDHit_overlap","AGC_overlap","MMseqs_group_overlap","nucleotide_hash_overlap","protein_hash_overlap","target_missing")
    checks={"strict_exact":bool(len(matched)==139316),"core_exact":bool(len(core_frame)==50000),"extended_derived":bool(len(extended)==extended_total and extended_total<=100000),"nested":bool(audits["core_nested_in_extended"]),"all_leakage_zero":bool(all(audits[level][key]==0 for level in ("strict_matched_pool","core_50k",f"extended_{extended_total}") for key in zero_keys)),"all_strata_and_splits":bool(all(audits[level]["all_splits"] and audits[level]["all_eight_strata"] for level in ("core_50k",f"extended_{extended_total}"))),"protein_hash_complete":bool(matched.protein_sha256.notna().all())}
    if not all(checks.values()): raise RuntimeError(f"W2C-06 failed: {checks}; audits={audits}")
    audits["validation"]={"passed":True,"checks":checks}; write_json(audit_path,audits)
    retention=[]
    for stage,data in [("protein_valid_input",frame),("strict_matched_pool",matched),("core_50k",core_frame),(f"extended_{extended_total}",extended)]:
        for stratum,n in data.groupby("stratum").size().items(): retention.append({"stage":stage,"stratum":stratum,"n":int(n)})
    pd.DataFrame(retention).to_csv(retention_path,sep="\t",index=False)
    numeric=["length_nt","GC_fraction","protein_length","prevalence","mean_abundance"]
    smd=[]
    for column in numeric:
        a=extended.loc[extended.function_label.eq("strict_known"),column].astype(float); b=extended.loc[extended.function_label.eq("strict_unknown"),column].astype(float); pooled=np.sqrt((a.var()+b.var())/2); smd.append({"covariate":column,"SMD_known_minus_unknown":float((a.mean()-b.mean())/pooled) if pooled else 0})
    balance.write_text("# Benchmark v1 balance\n\n"+pd.DataFrame(smd).to_csv(sep="\t",index=False)+"\n\n"+counts.to_csv(sep="\t",index=False))
    policy.write_text("\n".join(["# Conservative benchmark v1 policy","",f"- Strict joint matched pool: `{len(matched):,}`",f"- Core: `{len(core_frame):,}` (6,250 per stratum)",f"- Extended: `{len(extended):,}` ({extended_per:,} per stratum), derived from minimum split-aware capacity",f"- Split quotas per extended stratum: `{quotas(extended_per)}`","- Matching was not relaxed; benchmark_v2 200k is deferred.","- Split assignment precedes sampling and is frozen at MMseqs group level.",""]))
    def out(path:Path): return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    outputs=[strict,core,extended_path,split_path,protein_index,policy,balance,audit_path,retention_path]
    write_json(manifest_path,{"task_id":"W2C-06","status":"PASS","started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv),"inputs":[{"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":None} for path in paths.values()],"parameters":{"matching":"strict_joint_coarsened","seed_matching":20260830,"seed_split_sampling":2026,"extended_per_stratum":extended_per},"outputs":[out(path) for path in outputs],"validation":{"passed":True,"checks":checks},"scientific_decision":"FREEZE_CORE_50K_AND_LARGEST_STRICT_BALANCED_EXTENSION","affected_previous_tasks":["W2-12 resolved by corrective policy"],"notes":["Original blocked W2-12 manifest preserved."]})
    print(json.dumps({"status":"PASS","strict":len(matched),"core":len(core_frame),"extended":len(extended),"checks":checks},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
