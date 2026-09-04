#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from polarfunc.provenance import sha256_file, utc_now, write_json

SEEDS=(13,42,73,101,2026); C_GRID=(0.01,0.1,1.0,10.0,100.0)


def scores(y:np.ndarray,p:np.ndarray)->dict[str,float]:
    hard=p>=.5
    return {"AUROC":float(roc_auc_score(y,p)),"AUPRC":float(average_precision_score(y,p)),"MCC":float(matthews_corrcoef(y,hard)),"balanced_accuracy":float(balanced_accuracy_score(y,hard)),"F1":float(f1_score(y,hard,zero_division=0)),"Brier":float(brier_score_loss(y,p))}


def model(C:float,seed:int)->LogisticRegression:
    return LogisticRegression(C=C,solver="liblinear",max_iter=2000,random_state=seed,class_weight=None)


def main()->int:
    started=utc_now(); benchmark_path=ROOT/"manifests/benchmark_v1_core_50k.parquet"; benchmark_sha=sha256_file(benchmark_path); benchmark=pd.read_parquet(benchmark_path).sort_values("CDHit_ID").reset_index(drop=True); ids=benchmark.CDHit_ID.astype(str).tolist()
    result_path=ROOT/"artifacts/results/results_long.parquet"; null_path=ROOT/"artifacts/results/null_controls.parquet"; report=ROOT/"reports/corrective/cpu_evaluation.md"; transfer_report=ROOT/"reports/corrective/known_to_unknown_cpu.md"; manifest_path=ROOT/"manifests/corrective/W2C-09/manifest.json"
    for path in (result_path,null_path,report,transfer_report,manifest_path):
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    def aligned_parquet(path:Path,columns:list[str])->np.ndarray:
        f=pd.read_parquet(path).set_index("CDHit_ID").loc[ids,columns]; return f.to_numpy(np.float32)
    length=aligned_parquet(ROOT/"artifacts/features/length_gc.parquet",["length_nt"]); gc=aligned_parquet(ROOT/"artifacts/features/length_gc.parquet",["GC_fraction"]); length_gc=np.hstack([length,gc]); codon_cols=[f"codon_{''.join(x)}" for x in __import__('itertools').product("ACGT",repeat=3)]+["GC1","GC2","GC3"]; codon=aligned_parquet(ROOT/"artifacts/features/codon_features.parquet",codon_cols); aa=aligned_parquet(ROOT/"artifacts/features/aa_composition.parquet",[f"aa_{x}" for x in "ACDEFGHIKLMNPQRSTVWY"])
    kmer_all=sparse.load_npz(ROOT/"artifacts/features/kmer_1_6.npz"); kmer_ids=pd.read_parquet(ROOT/"artifacts/features/kmer_ids.parquet").CDHit_ID.astype(str).tolist(); kpos={x:i for i,x in enumerate(kmer_ids)}; kmer=kmer_all[[kpos[x] for x in ids]]
    representations={"length_nt":length,"GC":gc,"length_GC":length_gc,"codon_GCpos":codon,"AA_composition":aa,"kmer_1_6":kmer}
    tasks={
        "known":(benchmark.function_label.eq("strict_known"),benchmark.function_label.eq("strict_known"),benchmark.function_label.eq("strict_known")),
        "unknown":(benchmark.function_label.eq("strict_unknown"),benchmark.function_label.eq("strict_unknown"),benchmark.function_label.eq("strict_unknown")),
        "combined":(pd.Series(True,index=benchmark.index),pd.Series(True,index=benchmark.index),pd.Series(True,index=benchmark.index)),
        "known_to_unknown":(benchmark.function_label.eq("strict_known"),benchmark.function_label.eq("strict_known"),benchmark.function_label.eq("strict_unknown")),
    }
    y=benchmark.ecology_label.eq("high").to_numpy(int); split=benchmark.split.astype(str)
    def evaluate(job):
        rep_name,X,task,train_mask,val_mask,test_mask=job
        sparse_input=sparse.issparse(X)
        train=np.where(split.eq("train")&train_mask)[0]; val=np.where(split.eq("validation")&val_mask)[0]; test=np.where(split.eq("test")&test_mask)[0]
        scaler=StandardScaler(with_mean=not sparse_input); x_train=scaler.fit_transform(X[train]); x_val=scaler.transform(X[val]); x_test=scaler.transform(X[test])
        best_C=None; best_score=-1.0
        for C in C_GRID:
            estimator=model(C,2026).fit(x_train,y[train]); value=average_precision_score(y[val],estimator.predict_proba(x_val)[:,1])
            if value>best_score: best_score=float(value); best_C=C
        choice={"C":best_C,"validation_AUPRC":best_score,"train_n":len(train),"validation_n":len(val),"test_n":len(test)}
        local=[]
        for seed in SEEDS:
            estimator=model(best_C,seed).fit(x_train,y[train]); p=estimator.predict_proba(x_test)[:,1]; local.append({"representation":rep_name,"task":task,"seed":seed,"C":best_C,"split_hash":benchmark_sha,"train_n":len(train),"test_n":len(test),**scores(y[test],p)})
        return f"{rep_name}::{task}",choice,local
    jobs=[(rep_name,X,task,*masks) for rep_name,X in representations.items() for task,masks in tasks.items()]
    rows=[]; chosen={}
    with ThreadPoolExecutor(max_workers=12) as executor:
        for key,choice,local in executor.map(evaluate,jobs): chosen[key]=choice; rows.extend(local)
    results=pd.DataFrame(rows); results.to_parquet(result_path,index=False,compression="zstd")
    train=np.where(split.eq("train"))[0]; test=np.where(split.eq("test"))[0]; scaler=StandardScaler(with_mean=False); x_train=scaler.fit_transform(kmer[train]); x_test=scaler.transform(kmer[test]); best_C=chosen["kmer_1_6::combined"]["C"]; prevalence=float(y[test].mean())
    def evaluate_null(seed:int):
        local=[]
        rng=np.random.default_rng(seed); shuffled=y[train].copy(); rng.shuffle(shuffled); estimator=model(best_C,seed).fit(x_train,shuffled); p=estimator.predict_proba(x_test)[:,1]; local.append({"null":"permuted_labels","seed":seed,"dimension":x_train.shape[1],"prevalence":prevalence,**scores(y[test],p)})
        density=min(0.05,float(kmer.nnz)/(kmer.shape[0]*kmer.shape[1])); random_train=sparse.random(len(train),kmer.shape[1],density=density,random_state=seed,data_rvs=lambda n:rng.normal(size=n).astype(np.float32),format="csr"); random_test=sparse.random(len(test),kmer.shape[1],density=density,random_state=seed+1,data_rvs=lambda n:rng.normal(size=n).astype(np.float32),format="csr"); random_scaler=StandardScaler(with_mean=False); random_train=random_scaler.fit_transform(random_train); random_test=random_scaler.transform(random_test); estimator=model(best_C,seed).fit(random_train,y[train]); p=estimator.predict_proba(random_test)[:,1]; local.append({"null":"gaussian_sparse_features","seed":seed,"dimension":kmer.shape[1],"prevalence":prevalence,**scores(y[test],p)})
        return local
    null_rows=[]
    with ThreadPoolExecutor(max_workers=5) as executor:
        for local in executor.map(evaluate_null,SEEDS): null_rows.extend(local)
    nulls=pd.DataFrame(null_rows); nulls.to_parquet(null_path,index=False,compression="zstd")
    means=nulls.groupby("null").agg(AUROC=("AUROC","mean"),AUPRC=("AUPRC","mean"),prevalence=("prevalence","mean"))
    split_sets={name:set(benchmark.loc[split.eq(name),"CDHit_ID"]) for name in ("train","validation","test")}
    checks={"five_seed_records":bool(results.groupby(["representation","task"]).seed.nunique().eq(5).all()),"known_transfer_train_has_zero_unknown":not bool((benchmark.loc[split.eq("train")&tasks["known_to_unknown"][0],"function_label"]=="strict_unknown").any()),"train_val_test_disjoint":bool(split_sets["train"].isdisjoint(split_sets["validation"]|split_sets["test"]) and split_sets["validation"].isdisjoint(split_sets["test"])),"feature_ids_cover_core":bool(set(ids).issubset(set(kmer_ids))),"train_only_scaling":True,"all_metrics_finite":bool(np.isfinite(results[["AUROC","AUPRC","MCC","balanced_accuracy","F1","Brier"]]).all().all()),"permuted_null_near_chance":bool(abs(means.loc["permuted_labels","AUROC"]-.5)<=.1 and abs(means.loc["permuted_labels","AUPRC"]-means.loc["permuted_labels","prevalence"])<=.1),"random_features_near_chance":bool(abs(means.loc["gaussian_sparse_features","AUROC"]-.5)<=.1 and abs(means.loc["gaussian_sparse_features","AUPRC"]-means.loc["gaussian_sparse_features","prevalence"])<=.1)}
    if not all(checks.values()): raise RuntimeError(f"W2C-09 null/leakage gate failed: {checks}; means={means}")
    summary=results.groupby(["representation","task"]).agg(AUROC_mean=("AUROC","mean"),AUROC_sd=("AUROC","std"),AUPRC_mean=("AUPRC","mean"),MCC_mean=("MCC","mean"),balanced_accuracy_mean=("balanced_accuracy","mean")).reset_index(); report.write_text("# Unified CPU evaluation\n\nLogistic probes use train-only scaling, validation-only C selection, five retained seeds, and the frozen group-aware split.\n\n"+summary.to_csv(sep="\t",index=False)+"\n## Null controls\n\n"+means.reset_index().to_csv(sep="\t",index=False)); transfer_report.write_text("# Known-to-unknown CPU transfer\n\nTrain and validation contain strict-known proteins only; evaluation contains strict-unknown test proteins only.\n\n"+summary.loc[summary.task.eq("known_to_unknown")].to_csv(sep="\t",index=False))
    def out(path:Path): return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    outputs=[result_path,null_path,report,transfer_report]; write_json(manifest_path,{"task_id":"W2C-09","status":"PASS","started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv),"inputs":[{"path":str(benchmark_path.relative_to(ROOT)),"sha256":benchmark_sha,"rows":len(benchmark)}],"parameters":{"seeds":SEEDS,"C_grid":C_GRID,"selection_metric":"validation AUPRC","parallel_representation_tasks":12,"chosen":chosen},"outputs":[out(path) for path in outputs],"validation":{"passed":True,"checks":checks},"scientific_decision":"CPU_BASELINES_AND_NULLS_PASS","affected_previous_tasks":["W2-15"],"notes":["Continuous R2 ridge sensitivity deferred until headline classification and null gates pass."]})
    print(json.dumps({"status":"PASS","checks":checks,"summary":summary.to_dict('records')},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
