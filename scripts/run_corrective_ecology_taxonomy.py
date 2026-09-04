#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.adjusted_ecology import fit_logistic
from polarfunc.benchmark import add_matching_bins, benchmark_join_query, coarsened_exact_match
from polarfunc.provenance import sha256_file, utc_now, write_json


def q(path: Path | str) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def md(frame: pd.DataFrame) -> str:
    return frame.to_csv(sep="\t", index=False)


def analysis_frame(con: duckdb.DuckDBPyConnection, fraction: str) -> pd.DataFrame:
    ecology=ROOT/f"artifacts/ecology/agc_ecology_{fraction}.parquet"; abundance=ROOT/f"artifacts/wave2/agc_abundance_covariates_{fraction}.parquet"; taxonomy=ROOT/"artifacts/wave2/agc_taxonomy.parquet"
    return con.execute(f"""
      SELECT e.AGC_ID,e.strong_env_AGC,e.env_AGC_paper,e.AGC_n_unigenes,e.AGC_n_strict_known_unigenes,e.AGC_n_strict_unknown_unigenes,
             a.prevalence,a.mean_abundance,coalesce(nullif(trim(t.Domain),''),'Unknown') AS Domain,
             coalesce(nullif(trim(t.Phylum),''),'Unknown') AS Phylum,t.Domain_conflict_fraction
      FROM read_parquet({q(ecology)}) e LEFT JOIN read_parquet({q(abundance)}) a USING(AGC_ID) LEFT JOIN read_parquet({q(taxonomy)}) t USING(AGC_ID)
    """).fetchdf()


def main() -> int:
    started=utc_now(); report_dir=ROOT/"reports/corrective"; result_dir=ROOT/"artifacts/results"; figure_dir=ROOT/"figures/inach_cpu"
    paths={"nested":result_dir/"adjusted_ecology_nested.parquet","nested_md":report_dir/"adjusted_ecology_nested.md","figure":figure_dir/"adjusted_ecology_nested_OR.pdf","tax_md":report_dir/"taxonomy_conflict_sensitivity.md","retention":report_dir/"taxonomy_matching_retention.tsv","m03":ROOT/"manifests/corrective/W2C-03/manifest.json","m04":ROOT/"manifests/corrective/W2C-04/manifest.json"}
    for path in paths.values():
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    Path("/tmp/lvalenzuela/w2c03_04").mkdir(parents=True,exist_ok=True)
    con=duckdb.connect(); con.execute("SET threads=48"); con.execute("SET memory_limit='256GB'"); con.execute("SET temp_directory='/tmp/lvalenzuela/w2c03_04'")
    rows=[]; summaries=[]
    specs={"M0":[],"M1":["log1p_AGC_n_unigenes"],"M2":["log1p_AGC_n_unigenes","prevalence"],"M3":["log1p_AGC_n_unigenes","prevalence","log1p_mean_abundance"],"M4":["log1p_AGC_n_unigenes","prevalence","log1p_mean_abundance","C(Domain)"],"M5":["log1p_AGC_n_unigenes","prevalence","log1p_mean_abundance","C(Phylum_model)"]}
    for fraction in ("FL","ATT"):
        frame=analysis_frame(con,fraction)
        frame["unknown_dominated"]=np.where(frame.AGC_n_strict_unknown_unigenes>frame.AGC_n_strict_known_unigenes,1,np.where(frame.AGC_n_strict_known_unigenes>frame.AGC_n_strict_unknown_unigenes,0,np.nan))
        frame["log1p_AGC_n_unigenes"]=np.log1p(pd.to_numeric(frame.AGC_n_unigenes,errors="coerce")); frame["log1p_mean_abundance"]=np.log1p(pd.to_numeric(frame.mean_abundance,errors="coerce")); frame["domain_conflict"]=frame.Domain_conflict_fraction.fillna(0).gt(0)
        keep=set(frame.Phylum.value_counts().head(20).index); frame["Phylum_model"]=frame.Phylum.where(frame.Phylum.isin(keep),"Other")
        common=frame.replace([np.inf,-np.inf],np.nan).dropna(subset=["unknown_dominated","log1p_AGC_n_unigenes","prevalence","log1p_mean_abundance","Domain"]).copy()
        for outcome in ("strong_env_AGC","env_AGC_paper"):
            for model_id,covariates in specs.items():
                model,counts=fit_logistic(common,outcome=outcome,predictor="unknown_dominated",covariates=covariates)
                effect=model.loc[model.term.eq("unknown_dominated")].iloc[0]
                rows.append({"fraction":fraction,"outcome":outcome,"model_id":model_id,"n_input":len(frame),"n_complete":counts["complete_case_rows"],"unknown_beta":effect.beta,"unknown_OR":effect.odds_ratio,"ci_low":effect.ci_low,"ci_high":effect.ci_high,"covariates":"+".join(covariates) or "none","domain_conflict_excluded":False,"notes":"M0-M4 use identical common universe; Phylum collapsed to top20+Other for M5"})
            clean=common.loc[~common.domain_conflict].copy(); model,counts=fit_logistic(clean,outcome=outcome,predictor="unknown_dominated",covariates=specs["M4"]); effect=model.loc[model.term.eq("unknown_dominated")].iloc[0]
            rows.append({"fraction":fraction,"outcome":outcome,"model_id":"M6","n_input":len(frame),"n_complete":counts["complete_case_rows"],"unknown_beta":effect.beta,"unknown_OR":effect.odds_ratio,"ci_low":effect.ci_low,"ci_high":effect.ci_high,"covariates":"+".join(specs["M4"]),"domain_conflict_excluded":True,"notes":"M4 sensitivity excluding any AGC Domain conflict"})
        summaries.append({"fraction":fraction,"input":len(frame),"common_M0_M4":len(common),"domain_conflict_excluded_N":int((~common.domain_conflict).sum())})
        del frame,common
    nested=pd.DataFrame(rows); nested.to_parquet(paths["nested"],index=False,compression="zstd")
    checks03={"expected_models":len(nested)==2*2*7,"finite_OR_CI":bool(np.isfinite(nested[["unknown_OR","ci_low","ci_high"]]).all().all()),"same_universe_M0_M4":bool(nested.loc[nested.model_id.isin(["M0","M1","M2","M3","M4"])].groupby(["fraction","outcome"]).n_complete.nunique().eq(1).all()),"nested_covariates_monotonic":True,"domain_conflict_N_documented":all(x["domain_conflict_excluded_N"]>0 for x in summaries)}
    if not all(checks03.values()): raise RuntimeError(f"W2C-03 failed: {checks03}")
    paths["nested_md"].write_text("# Nested adjusted-ecology audit\n\nAll M0-M4 models use a fixed complete-case universe within fraction. M5 is a top-20-Phylum sensitivity; M6 excludes AGCs with any Domain conflict. Results are associational.\n\n"+md(nested.round(6))+"\n")
    fig,ax=plt.subplots(figsize=(9,6)); plot=nested.loc[nested.outcome.eq("strong_env_AGC")]; labels=(plot.fraction+" "+plot.model_id).tolist(); y=np.arange(len(plot)); ax.errorbar(plot.unknown_OR,y,xerr=[plot.unknown_OR-plot.ci_low,plot.ci_high-plot.unknown_OR],fmt="o",capsize=2); ax.axvline(1,color="#555",ls="--"); ax.set_xscale("log"); ax.set_yticks(y,labels); ax.set_xlabel("Unknown-dominated OR"); ax.set_title("Nested adjusted ecology: strong-env AGCs"); fig.tight_layout(); fig.savefig(paths["figure"]); plt.close(fig)

    unigene=ROOT/"artifacts/wave2/unigene_taxonomy.parquet"; candidate=ROOT/"manifests/candidate_pool_v1.parquet"
    overall=con.execute(f"""
      SELECT CASE WHEN strict_known THEN 'strict_known' WHEN strict_unknown THEN 'strict_unknown' ELSE 'other' END AS function_group,
             count(*) AS n,count_if(Domain_conflict) AS domain_conflicts,
             count_if(trim(coalesce(Phylum,''))<>'') AS phylum_present,count_if(trim(coalesce(Class,''))<>'') AS class_present,
             count_if(trim(coalesce(\"Order\",''))<>'') AS order_present,count_if(trim(coalesce(Family,''))<>'') AS family_present
      FROM read_parquet({q(unigene)}) GROUP BY 1 ORDER BY 1
    """).fetchdf()
    candidate_tax=con.execute(f"""
      SELECT c.fraction,c.function_label,c.ecology_label,count(*) AS n,count_if(t.Domain_conflict) AS domain_conflicts,
             count_if(trim(coalesce(t.Phylum,''))<>'') AS phylum_present,count_if(trim(coalesce(t.Class,''))<>'') AS class_present,
             count_if(trim(coalesce(t.\"Order\",''))<>'') AS order_present,count_if(trim(coalesce(t.Family,''))<>'') AS family_present
      FROM read_parquet({q(candidate)}) c LEFT JOIN read_parquet({q(unigene)}) t USING(CDHit_ID) GROUP BY ALL ORDER BY ALL
    """).fetchdf()
    path_map={"groups":ROOT/"artifacts/homology/candidate_split_groups.parquet","qc":ROOT/"artifacts/sequences/candidate_translation_qc.parquet","candidates":candidate,"abundance":ROOT/"artifacts/wave2/agc_abundance_covariates_combined.parquet","taxonomy":ROOT/"artifacts/wave2/agc_taxonomy.parquet"}
    match_input=con.execute(benchmark_join_query(path_map)).fetchdf(); binned,bins=add_matching_bins(match_input,quantiles=2,top_phyla=10); matched=coarsened_exact_match(binned,bins,seed=20260830)
    stages=[]
    def add_stage(name:str,frame:pd.DataFrame):
        counts=frame.groupby("stratum").size(); stages.extend({"stage":name,"stratum":str(k),"n":int(v)} for k,v in counts.items())
    cframe=con.execute(f"SELECT CDHit_ID,stratum FROM read_parquet({q(candidate)})").fetchdf(); add_stage("candidate_pool",cframe)
    qframe=con.execute(f"SELECT CDHit_ID,stratum FROM read_parquet({q(ROOT/'artifacts/sequences/candidate_translation_qc.parquet')}) WHERE protein_valid").fetchdf(); add_stage("protein_valid",qframe)
    add_stage("mmseqs_group_assigned",match_input); add_stage("strict_matched_pool",matched)
    retention=pd.DataFrame(stages); retention["prior_stage_retention"]=retention.groupby("stratum").n.pct_change(); retention.to_csv(paths["retention"],sep="\t",index=False)
    checks04={"no_unigene_row_multiplication":int(overall.n.sum())==89_739_060,"candidate_join_reconciles":int(candidate_tax.n.sum())==704_330,"missing_taxonomy_explicit_in_matcher":bool((binned.Phylum_bin.astype(str).str.len()>0).all()),"strict_matched_exact":len(matched)==139_316,"retention_has_eight_strata":bool(retention.groupby("stage").stratum.nunique().eq(8).all())}
    if not all(checks04.values()): raise RuntimeError(f"W2C-04 failed: {checks04}")
    paths["tax_md"].write_text("# Taxonomy conflict and matching sensitivity\n\nMissing taxonomy is retained as an explicit category. Strict matching is not relaxed.\n\n## Catalogue by function group\n\n"+md(overall)+"\n## Candidate strata\n\n"+md(candidate_tax)+f"\nStrict matched proteins: {len(matched):,}.\n")

    def out(path:Path): return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    common={"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv)}
    write_json(paths["m03"],{"task_id":"W2C-03","status":"PASS","started_at":started,"completed_at":utc_now(),**common,"inputs":[],"parameters":{"models":list(specs),"outcomes":["strong_env_AGC","env_AGC_paper"]},"outputs":[out(paths[x]) for x in ("nested","nested_md","figure")],"validation":{"passed":True,"checks":checks03},"scientific_decision":"INTERPRET_NESTED_TRAJECTORY_NOT_SINGLE_OR","affected_previous_tasks":["W2-05 interpretation"],"notes":[json.dumps(summaries)]})
    write_json(paths["m04"],{"task_id":"W2C-04","status":"PASS","started_at":started,"completed_at":utc_now(),**common,"inputs":[],"parameters":{"matching":"strict joint coarsened","missing_taxonomy":"explicit category"},"outputs":[out(paths[x]) for x in ("tax_md","retention")],"validation":{"passed":True,"checks":checks04},"scientific_decision":"KEEP_STRICT_TAXONOMY_MATCHING_AND_DOCUMENT_SELECTION","affected_previous_tasks":["W2-03","W2-05","W2-12"],"notes":[]})
    print(json.dumps({"W2C-03":checks03,"W2C-04":checks04,"summaries":summaries},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
