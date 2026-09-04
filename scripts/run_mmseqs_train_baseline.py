#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from polarfunc.provenance import sha256_file, utc_now, write_json


def fasta_records(path:Path,selected:set[str]):
    identifier=None; lines=[]
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if identifier in selected: yield identifier,"".join(lines)
                identifier=line[1:].split()[0]; lines=[]
            elif identifier in selected: lines.append(line.strip())
        if identifier in selected: yield identifier,"".join(lines)


def metrics(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    return {"AUROC":float(roc_auc_score(y,p)) if len(set(y))>1 else float("nan"),"AUPRC":float(average_precision_score(y,p)),"MCC":float(matthews_corrcoef(y,p>=.5)),"balanced_accuracy":float(balanced_accuracy_score(y,p>=.5)),"F1":float(f1_score(y,p>=.5,zero_division=0))}


def main()->int:
    started=utc_now(); mmseqs=shutil.which("mmseqs") or str(Path(sys.executable).resolve().parent/"mmseqs")
    if not Path(mmseqs).is_file() or not os.access(mmseqs,os.X_OK): raise RuntimeError("mmseqs unavailable")
    matches=sorted((ROOT/"manifests").glob("benchmark_v1_extended_*.parquet"));
    if len(matches)!=1: raise RuntimeError(matches)
    run_token=os.environ.get("POLARFUNC_RUN_ID",f"{os.environ.get('OAR_JOB_ID','none')}_{os.getpid()}")
    benchmark_path=matches[0]; benchmark=pd.read_parquet(benchmark_path); benchmark_sha=sha256_file(benchmark_path); base=ROOT/"artifacts/homology"; fasta_dir=base/"benchmark_v1_fastas"; db_dir=base/"mmseqs_train_db"; val_hits=base/"val_to_train_hits.parquet"; test_hits=base/"test_to_train_hits.parquet"; results_path=ROOT/"artifacts/results/mmseqs_baseline.parquet"; report=ROOT/"reports/corrective/mmseqs_baseline.md"; manifest_path=ROOT/"manifests/corrective/W2C-08/manifest.json"; scratch=Path(f"/tmp/lvalenzuela/w2c08_{run_token}")
    protected=[fasta_dir,db_dir,val_hits,test_hits,results_path,report,manifest_path,scratch]
    for path in protected:
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
    fasta_dir.mkdir(parents=True); db_dir.mkdir(parents=True); scratch.mkdir(parents=True); results_path.parent.mkdir(parents=True,exist_ok=True); report.parent.mkdir(parents=True,exist_ok=True); manifest_path.parent.mkdir(parents=True,exist_ok=True)
    selected=set(benchmark.CDHit_ID.astype(str)); records=dict(fasta_records(ROOT/"artifacts/sequences/candidate_pool_v1.faa",selected))
    if set(records)!=selected: raise RuntimeError("Protein FASTA IDs do not match benchmark")
    fasta_paths={}
    for split in ("train","validation","test"):
        path=fasta_dir/f"{split}.faa"; fasta_paths[split]=path
        with path.open("w") as handle:
            for identifier in sorted(benchmark.loc[benchmark.split.eq(split),"CDHit_ID"].astype(str)): handle.write(f">{identifier}\n{records[identifier]}\n")
    subprocess.run([mmseqs,"createdb",str(fasta_paths["train"]),str(db_dir/"train")],check=True)
    columns=["query","target","fident","qcov","tcov","alnlen","evalue","bitscore"]
    mmseqs_columns=["query","target","fident","qcov","tcov","alnlen","evalue","bits"]
    hit_outputs={"validation":val_hits,"test":test_hits}
    result_rows=[]
    train=benchmark.loc[benchmark.split.eq("train"),["CDHit_ID","ecology_label","R2_caret"]].rename(columns={"CDHit_ID":"target","ecology_label":"target_ecology","R2_caret":"target_R2"})
    train_prevalence=float(benchmark.loc[benchmark.split.eq("train"),"ecology_label"].eq("high").mean())
    for split in ("validation","test"):
        raw=scratch/f"{split}.tsv"; tmp=scratch/f"tmp_{split}"
        subprocess.run([mmseqs,"easy-search",str(fasta_paths[split]),str(db_dir/"train"),str(raw),str(tmp),"--threads","64","--max-seqs","10","--format-output",",".join(mmseqs_columns)],check=True)
        hits=pd.read_csv(raw,sep="\t",names=columns) if raw.stat().st_size else pd.DataFrame(columns=columns)
        if not hits.empty:
            hits=hits.sort_values(["query","bitscore","evalue","target"],ascending=[True,False,True,True]).drop_duplicates("query",keep="first")
        query=benchmark.loc[benchmark.split.eq(split),["CDHit_ID","ecology_label","R2_caret"]].rename(columns={"CDHit_ID":"query"}); joined=query.merge(hits,on="query",how="left",validate="one_to_one").merge(train,on="target",how="left",validate="many_to_one")
        joined["has_hit"]=joined.target.notna(); joined["prediction"]=np.where(joined.has_hit,joined.target_ecology.eq("high").astype(float),train_prevalence)
        finite_identity=joined.loc[joined.has_hit,"fident"].dropna(); identity_scale=100.0 if finite_identity.empty or float(finite_identity.max())<=1.0 else 1.0
        joined["fident_percent"]=joined.fident*identity_scale
        joined["identity_bin"]=pd.cut(joined.fident_percent.fillna(-1),bins=[-2,-.1,30,50,70,101],labels=["no_hit","<30%","30-50%","50-70%",">70%"],right=False)
        joined.to_parquet(hit_outputs[split],index=False,compression="zstd")
        for identity_bin,group in [("overall",joined),*((str(k),g) for k,g in joined.groupby("identity_bin",observed=True))]:
            if len(group)==0: continue
            values=metrics(group.ecology_label.eq("high").to_numpy(int),group.prediction.to_numpy(float)); result_rows.append({"split":split,"identity_bin":identity_bin,"n":len(group),"hit_fraction":float(group.has_hit.mean()),**values})
    results=pd.DataFrame(result_rows); results.to_parquet(results_path,index=False,compression="zstd")
    freeze_manifest=json.loads((ROOT/"manifests/corrective/W2C-06/manifest.json").read_text())
    frozen_sha={item["path"]:item["sha256"] for item in freeze_manifest["outputs"]}.get(str(benchmark_path.relative_to(ROOT)))
    nontrain_ids=set(benchmark.loc[benchmark.split.ne("train"),"CDHit_ID"].astype(str)); train_ids=set(benchmark.loc[benchmark.split.eq("train"),"CDHit_ID"].astype(str))
    observed_targets=set(pd.concat([pd.read_parquet(val_hits,columns=["target"]),pd.read_parquet(test_hits,columns=["target"])],ignore_index=True).target.dropna().astype(str))
    checks={"train_db_excludes_queries":nontrain_ids.isdisjoint(train_ids),"all_observed_targets_are_train":observed_targets.issubset(train_ids),"train_db_created":any(db_dir.glob("train*")),"every_validation_query":pq.ParquetFile(val_hits).metadata.num_rows==int(benchmark.split.eq("validation").sum()),"every_test_query":pq.ParquetFile(test_hits).metadata.num_rows==int(benchmark.split.eq("test").sum()),"split_checksum_matches":frozen_sha==benchmark_sha,"finite_overall_metrics":bool(np.isfinite(results.loc[results.identity_bin.eq("overall"),["AUROC","AUPRC","MCC","balanced_accuracy","F1"]]).all().all())}
    if not all(checks.values()): raise RuntimeError(f"W2C-08 failed: {checks}")
    report.write_text("# Train-only MMseqs nearest-neighbour baseline\n\nReference database contains TRAIN proteins only. Validation/test no-hit queries are retained and assigned the train prevalence.\n\n"+results.to_csv(sep="\t",index=False))
    outputs=[*fasta_paths.values(),val_hits,test_hits,results_path,report]
    def out(path:Path): return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    write_json(manifest_path,{"task_id":"W2C-08","status":"PASS","started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv),"inputs":[{"path":str(benchmark_path.relative_to(ROOT)),"sha256":benchmark_sha,"rows":len(benchmark)}],"parameters":{"reference_split":"train","max_seqs":10,"top_hit_tie_break":"bitscore desc, evalue asc, target ID asc","identity_bins_unit":"percent; fident retained in native MMseqs units","no_hit_prediction":"train prevalence"},"outputs":[out(path) for path in outputs],"validation":{"passed":True,"checks":checks},"scientific_decision":"TRAIN_ONLY_HOMOLOGY_BASELINE_READY","affected_previous_tasks":["W2-14"],"notes":[f"mmseqs={subprocess.check_output([mmseqs,'version'],text=True).strip()}",f"train_db_files={len(list(db_dir.glob('train*')))}"]})
    print(json.dumps({"status":"PASS","checks":checks,"results":result_rows},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
