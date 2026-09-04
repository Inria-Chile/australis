#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from polarfunc.provenance import sha256_file, utc_now, write_json


def save(fig,path:Path)->None:
    tmp=path.with_name("."+path.name); fig.savefig(tmp,bbox_inches="tight"); tmp.replace(path); plt.close(fig)


def main()->int:
    started=utc_now(); numbers_path=ROOT/"reports/INACH_CPU_NUMBERS_v2.json"; results_path=ROOT/"reports/INACH_CPU_RESULTS_v2.md"; claims=ROOT/"reports/corrective/INACH_claims_allowed.md"; figure_dir=ROOT/"figures/inach_cpu_v2"; manifest_path=ROOT/"manifests/corrective/W2C-13/manifest.json"
    figures=[figure_dir/name for name in ("catalogue_funnel.pdf","nested_adjusted_OR.pdf","cag_macro_specificity.pdf","cpu_baselines.pdf")]
    for path in [numbers_path,results_path,claims,manifest_path,*figures]:
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    input_paths=[ROOT/"reports/corrective/function_label_audit.json",ROOT/"artifacts/results/adjusted_ecology_nested.parquet",ROOT/"reports/wave2/cag_macro_watermass.json",ROOT/"artifacts/results/results_long.parquet",ROOT/"artifacts/results/null_controls.parquet",ROOT/"artifacts/results/mmseqs_baseline.parquet"]
    labels=json.loads(input_paths[0].read_text()); nested=pd.read_parquet(input_paths[1]); cag=json.loads(input_paths[2].read_text()); cpu=pd.read_parquet(input_paths[3]); nulls=pd.read_parquet(input_paths[4]); mmseqs=pd.read_parquet(input_paths[5])
    counts=labels["counts"]; cpu_summary=cpu.groupby(["representation","task"]).agg(AUROC_mean=("AUROC","mean"),AUROC_sd=("AUROC","std"),AUPRC_mean=("AUPRC","mean"),MCC_mean=("MCC","mean")).reset_index(); nested_head=nested.loc[nested.model_id.isin(["M0","M4","M6"])].to_dict("records")
    provenance=[{"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path)} for path in input_paths]
    payload={"catalogue_funnel":{"all_unigenes":counts["total_unigenes"],"broad_unknown":counts["broad_unknown_recomputed"],"strict_unknown":counts["strict_unknown_recomputed"],"strict_known":counts["strict_known_recomputed"],"definition":"representative-row explicit functional fields"},"nested_adjusted_ecology":nested_head,"cag_macro":{"scope":"env-AGC-derived CAG universe","fractions":cag["fractions"]},"cpu_sequence_baselines":cpu_summary.to_dict("records"),"mmseqs_nearest_train":mmseqs.loc[mmseqs.identity_bin.eq("overall")].to_dict("records"),"null_controls":nulls.groupby("null")[["AUROC","AUPRC"]].mean().reset_index().to_dict("records"),"scope":"CPU-only associational and predictive analyses; not causal","foundation_model_results_included":False,"provenance":provenance}; write_json(numbers_path,payload)
    funnel=payload["catalogue_funnel"]; fig,ax=plt.subplots(figsize=(7,4)); keys=["all_unigenes","broad_unknown","strict_unknown","strict_known"]; values=[funnel[k]/1e6 for k in keys]; ax.bar(keys,values,color=["#264653","#E76F51","#D1495B","#2A9D8F"]); ax.set_ylabel("Unigenes (millions)"); ax.set_title("ACE annotation funnel, corrected"); ax.tick_params(axis="x",rotation=20); save(fig,figures[0])
    plot=nested.loc[nested.model_id.isin(["M0","M1","M2","M3","M4","M6"]) & nested.outcome.eq("strong_env_AGC")]; fig,ax=plt.subplots(figsize=(8,5)); labels_plot=(plot.fraction+" "+plot.model_id).tolist(); y=np.arange(len(plot)); ax.errorbar(plot.unknown_OR,y,xerr=[plot.unknown_OR-plot.ci_low,plot.ci_high-plot.unknown_OR],fmt="o",capsize=2); ax.axvline(1,color="#555",ls="--"); ax.set_xscale("log"); ax.set_yticks(y,labels_plot); ax.set_xlabel("Unknown-dominated OR"); ax.set_title("Nested associational ecology"); save(fig,figures[1])
    fractions=["FL","ATT"]; coef=[cag["fractions"][f]["unknown_fraction_coefficient"]["estimate"] for f in fractions]; fig,ax=plt.subplots(figsize=(5,4)); ax.bar(fractions,coef,color=["#007C91","#D1495B"]); ax.set_ylabel("Unknown-fraction coefficient"); ax.set_title("CAG water-mass specificity"); save(fig,figures[2])
    combined=cpu_summary.loc[cpu_summary.task.eq("combined"),["representation","AUROC_mean"]].copy(); homology=mmseqs.loc[(mmseqs.split.eq("test"))&mmseqs.identity_bin.eq("overall"),["AUROC"]].copy(); homology["representation"]="MMseqs_nearest_train"; homology=homology.rename(columns={"AUROC":"AUROC_mean"}); combined=pd.concat([combined,homology],ignore_index=True).sort_values("AUROC_mean"); fig,ax=plt.subplots(figsize=(7,4)); ax.barh(combined.representation,combined.AUROC_mean,color="#2A9D8F"); ax.axvline(.5,color="#555",ls="--"); ax.set_xlabel("Test AUROC, mean across seeds"); ax.set_xlim(.45,1); ax.set_title("CPU sequence baselines"); save(fig,figures[3])
    results_path.write_text("\n".join(["# Corrected INACH CPU-only results","",f"The ACE catalogue contains {funnel['all_unigenes']:,} canonical unigenes, including {funnel['broad_unknown']:,} broad-unknown under the representative-row explicit-field definition, {funnel['strict_unknown']:,} strict-unknown, and {funnel['strict_known']:,} strict-known.","","Nested ecology models separate common-universe selection from covariate adjustment. Interpret trajectories and sensitivities, not a single preferred odds ratio.","","The CAG-macro analysis is explicitly restricted to the env-AGC-derived CAG universe.","","Classical CPU sequence and train-only MMseqs baselines plus null controls use the frozen leakage-resistant benchmark. No foundation-model biological performance result is included.",""]))
    claims.write_text("""# INACH claim matrix

## Allowed
- Exact corrected annotation counts with the representative-row definition.
- Translation-valid subset size and explicit selection caveat.
- MMseqs remote-homology group statistics.
- Nested adjusted ecology as associational, with common-universe and conflict sensitivities.
- CAG-macro association within the env-AGC-derived CAG scope.
- CPU composition/homology predictive baselines when accompanied by leakage and null gates.

## Forbidden
- Causal environmental or mechanistic interpretation.
- ESM2 or GenomeOcean biological performance.
- Biochemical function claims for strict-unknown genes.
- A 200k benchmark-v1 claim.
- Claim that CAG results independently validate random-forest selection.
""")
    checks={"correct_broad_unknown":funnel["broad_unknown"]==44_596_593,"not_complement_shortcut":funnel["broad_unknown"]!=funnel["all_unigenes"]-funnel["strict_known"],"all_numbers_traceable":bool(len(provenance)==len(input_paths) and all(item["sha256"]==sha256_file(ROOT/item["path"]) for item in provenance)),"cag_scope_explicit":payload["cag_macro"]["scope"]=="env-AGC-derived CAG universe","mmseqs_included":len(payload["mmseqs_nearest_train"])==2,"foundation_model_absent":payload["foundation_model_results_included"] is False,"figures_nonempty":all(path.stat().st_size>5000 for path in figures)}
    if not all(checks.values()): raise RuntimeError(f"W2C-13 failed: {checks}")
    outputs=[numbers_path,results_path,claims,*figures]; write_json(manifest_path,{"task_id":"W2C-13","status":"PASS","started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv),"inputs":[{"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":None} for path in input_paths],"parameters":{"foundation_models":False,"cag_scope":"env-AGC-derived CAG universe"},"outputs":[{"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":None} for path in outputs],"validation":{"passed":True,"checks":checks},"scientific_decision":"CORRECTED_INACH_CPU_REPORT_READY","affected_previous_tasks":["W2-19 report corrected"],"notes":[]}); print(json.dumps({"status":"PASS","checks":checks},indent=2)); return 0


if __name__=="__main__":raise SystemExit(main())
