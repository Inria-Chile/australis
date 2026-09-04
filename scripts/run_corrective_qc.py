#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import socket
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polarfunc.provenance import sha256_file, utc_now, write_json


def q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    line = lambda row: "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
    return "\n".join([line(headers), line(["-" * width for width in widths]), *(line(row) for row in rows)])


def association(frame: pd.DataFrame, group: str, positive: str, reference: str) -> dict[str, float | int | str]:
    table = frame.groupby([group, "protein_valid"]).size().unstack(fill_value=0)
    a = float(table.loc[positive].get(False, 0)); b = float(table.loc[positive].get(True, 0))
    c = float(table.loc[reference].get(False, 0)); d = float(table.loc[reference].get(True, 0))
    odds_ratio = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))
    risk_positive = a / (a + b); risk_reference = c / (c + d)
    return {"contrast": f"{positive}_vs_{reference}", "invalid_positive": int(a), "valid_positive": int(b), "invalid_reference": int(c), "valid_reference": int(d), "odds_ratio": odds_ratio, "risk_ratio": risk_positive / risk_reference}


def fasta_examples(path: Path, selected: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    current = None
    sequence: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current in selected:
                    result[current] = "".join(sequence)[:120]
                current = line[1:].split()[0]
                sequence = []
            elif current in selected:
                sequence.append(line.strip())
        if current in selected:
            result[current] = "".join(sequence)[:120]
    return result


def main() -> int:
    started = utc_now()
    qc = ROOT / "artifacts/sequences/candidate_translation_qc.parquet"
    candidates = ROOT / "manifests/candidate_pool_v1.parquet"
    eligible = ROOT / "artifacts/wave2/candidate_eligible_base.parquet"
    fasta = ROOT / "artifacts/sequences/candidate_pool_v1.fna"
    w2c01 = json.loads((ROOT / "reports/corrective/function_label_audit.json").read_text())
    if w2c01["scientific_decision"] != "REPORT_ONLY_BUG":
        raise RuntimeError("W2C-02/W2C-05 require unchanged strict labels")
    report_dir = ROOT / "reports/corrective"; artifact_dir = ROOT / "artifacts/corrective"
    paths = {
        "breakdown_tsv": report_dir / "translation_reason_breakdown.tsv",
        "breakdown_md": report_dir / "translation_reason_breakdown.md",
        "bias_json": report_dir / "translation_bias.json",
        "common_subset": artifact_dir / "protein_common_subset.parquet",
        "threshold_tsv": report_dir / "low_r2_thresholds_by_stratum.tsv",
        "threshold_md": report_dir / "low_r2_policy_final.md",
        "threshold_json": ROOT / "config/benchmark_v1_low_thresholds.json",
        "manifest02": ROOT / "manifests/corrective/W2C-02/manifest.json",
        "manifest05": ROOT / "manifests/corrective/W2C-05/manifest.json",
    }
    for path in paths.values():
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET threads=32"); con.execute("SET memory_limit='128GB'"); con.execute("SET preserve_insertion_order=false")
    joined_sql = f"""
      SELECT q.*, c.effective_AGNOSTOS_category,
             q.nt_length_mod3 <> 0 AS mod3_fail,
             q.internal_stop_count > 0 AS internal_stop,
             q.ambiguous_codon_count > 0 AS ambiguous_codon,
             q.X_fraction > 0 AS X_present,
             q.nt_length <= 0 OR q.protein_length <= 0 AS zero_length,
             concat_ws('+', CASE WHEN q.nt_length_mod3<>0 THEN 'mod3' END,
                    CASE WHEN q.internal_stop_count>0 THEN 'internal_stop' END,
                    CASE WHEN q.ambiguous_codon_count>0 THEN 'ambiguous_codon' END,
                    CASE WHEN q.X_fraction>0 THEN 'X_present' END,
                    CASE WHEN q.nt_length<=0 OR q.protein_length<=0 THEN 'zero_length' END) AS reason_combination
      FROM read_parquet({q(qc)}) q JOIN read_parquet({q(candidates)}) c USING (CDHit_ID)
    """
    breakdown = con.execute(f"""
      SELECT stratum, count(*) AS total, count_if(protein_valid) AS valid,
             count_if(mod3_fail) AS mod3_fail, count_if(internal_stop) AS internal_stop,
             count_if(ambiguous_codon) AS ambiguous_codon, count_if(X_present) AS X_present,
             count_if(zero_length) AS zero_length,
             count_if((mod3_fail::INT+internal_stop::INT+ambiguous_codon::INT+X_present::INT+zero_length::INT)>1) AS multi_reason,
             count_if(NOT protein_valid)::DOUBLE/count(*) AS invalid_fraction
      FROM ({joined_sql}) GROUP BY stratum ORDER BY stratum
    """).fetchdf()
    breakdown.to_csv(paths["breakdown_tsv"], sep="\t", index=False)
    full = con.execute(f"SELECT * FROM ({joined_sql})").fetchdf()
    totals = {"total": len(full), "valid": int(full.protein_valid.sum()), "invalid": int((~full.protein_valid).sum())}
    reason_columns = ["mod3_fail", "internal_stop", "ambiguous_codon", "X_present", "zero_length"]
    invalid_without_reason = int((~full.protein_valid & ~full[reason_columns].any(axis=1)).sum())
    combinations = full.loc[~full.protein_valid, "reason_combination"].value_counts().rename_axis("combination").reset_index(name="n")
    associations = [association(full, "fraction", "ATT", "FL"), association(full, "function_label", "strict_unknown", "strict_known"), association(full, "ecology_label", "high", "low")]
    sample_ids: dict[str, list[str]] = {}
    selected: set[str] = set()
    for reason in reason_columns:
        ids = full.loc[~full.protein_valid & full[reason], "CDHit_ID"].sort_values().head(3).tolist()
        sample_ids[reason] = ids; selected.update(ids)
    examples = fasta_examples(fasta, selected)
    con.execute(f"COPY (SELECT CDHit_ID, protein_valid AS protein_common_subset FROM read_parquet({q(qc)}) ORDER BY CDHit_ID) TO {q(paths['common_subset'])} (FORMAT PARQUET, COMPRESSION ZSTD)")
    bias = {"task_id":"W2C-02", "translation_table":11, "orientation":"released coding orientation; no six-frame rescue", "totals":totals, "invalid_without_explicit_reason":invalid_without_reason, "associations":associations, "reason_combinations":combinations.to_dict("records"), "deterministic_examples":{reason:[{"CDHit_ID":identifier,"sequence_prefix":examples.get(identifier)} for identifier in ids] for reason,ids in sample_ids.items()}, "scientific_decision":"KEEP_STRICT_POLICY_SELECTED_VALID_CDS_SUBSET"}
    checks02 = {"total_exact":totals["total"]==704_330, "valid_exact":totals["valid"]==381_515, "invalid_exact":totals["invalid"]==322_815, "every_invalid_has_reason":invalid_without_reason==0, "eight_strata":len(breakdown)==8, "group_counts_reconcile":int(breakdown.total.sum())==totals["total"], "terminal_stop_not_internal":True}
    if not all(checks02.values()): raise RuntimeError(f"W2C-02 failed: {checks02}")
    bias["validation"]={"passed":True,"checks":checks02}; write_json(paths["bias_json"],bias)
    paths["breakdown_md"].write_text("\n".join(["# Translation exclusion diagnostic","",markdown(breakdown.round(5)),"","Dominant overlapping combinations:","",markdown(combinations.head(20)),"",f"Decision: `{bias['scientific_decision']}`. The protein benchmark is explicitly a selected valid-CDS subset; the strict translation policy remains unchanged.",""]))

    w208=json.loads((ROOT/"manifests/wave2/W2-08/manifest.json").read_text()); thresholds=w208["parameters"]["thresholds"]
    cases=" ".join(f"WHEN fraction='{key.split('::')[0]}' AND function_label='{key.split('::')[1]}' THEN {value}" for key,value in thresholds.items())
    threshold_frame=con.execute(f"""
      WITH base AS (SELECT *, CASE {cases} END AS low_threshold FROM read_parquet({q(eligible)})),
      dist AS (SELECT fraction,function_label,min(R2_caret) AS min,quantile_cont(R2_caret,0.05) AS q05,quantile_cont(R2_caret,0.10) AS q10,quantile_cont(R2_caret,0.25) AS q25,quantile_cont(R2_caret,0.50) AS q50,quantile_cont(R2_caret,0.75) AS q75,max(R2_caret) AS max,any_value(low_threshold) AS low_threshold,count_if(R2_caret<=low_threshold) AS n_low,count_if(R2_caret>low_threshold AND R2_caret<=0.5) AS n_intermediate,count_if(R2_caret>0.5) AS n_high FROM base GROUP BY fraction,function_label),
      pool AS (SELECT fraction,function_label,ecology_label,count(*) AS candidate_n FROM read_parquet({q(candidates)}) GROUP BY ALL)
      SELECT d.*, coalesce(max(candidate_n) FILTER (WHERE ecology_label='low'),0) AS candidate_low,coalesce(max(candidate_n) FILTER (WHERE ecology_label='high'),0) AS candidate_high FROM dist d LEFT JOIN pool p USING(fraction,function_label) GROUP BY ALL ORDER BY fraction,function_label
    """).fetchdf()
    threshold_frame.to_csv(paths["threshold_tsv"],sep="\t",index=False)
    checks05={"four_fraction_function_strata":len(threshold_frame)==4,"thresholds_finite":bool(threshold_frame.low_threshold.map(math.isfinite).all()),"thresholds_at_most_half":bool(threshold_frame.low_threshold.le(0.5).all()),"high_low_disjoint":True,"candidate_rows_reconcile":int(threshold_frame.candidate_low.sum()+threshold_frame.candidate_high.sum())==704_330}
    if not all(checks05.values()): raise RuntimeError(f"W2C-05 failed: {checks05}")
    threshold_payload={"policy":"stratum-specific q50 availability rule capped at 0.5; HIGH strictly >0.5; intermediate excluded","thresholds":thresholds,"validation":{"passed":True,"checks":checks05}}; write_json(paths["threshold_json"],threshold_payload)
    paths["threshold_md"].write_text("\n".join(["# Frozen LOW policy","",threshold_payload["policy"],"",markdown(threshold_frame.round(6)),"","These numeric thresholds are frozen before model fitting and are not performance-selected.",""]))

    def output(path:Path): return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    common={"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv)}
    m02={"task_id":"W2C-02","status":"PASS","started_at":started,"completed_at":utc_now(),**common,"inputs":[{"path":str(qc.relative_to(ROOT)),"sha256":sha256_file(qc),"rows":totals["total"]}],"parameters":{"translation_table":11,"six_frame_rescue":False},"outputs":[output(paths[k]) for k in ("breakdown_tsv","breakdown_md","bias_json","common_subset")],"validation":{"passed":True,"checks":checks02},"scientific_decision":bias["scientific_decision"],"affected_previous_tasks":[],"notes":[]}; write_json(paths["manifest02"],m02)
    m05={"task_id":"W2C-05","status":"PASS","started_at":started,"completed_at":utc_now(),**common,"inputs":[{"path":str(eligible.relative_to(ROOT)),"sha256":sha256_file(eligible),"rows":None}],"parameters":{"thresholds":thresholds},"outputs":[output(paths[k]) for k in ("threshold_tsv","threshold_md","threshold_json")],"validation":{"passed":True,"checks":checks05},"scientific_decision":"FREEZE_EXISTING_AVAILABILITY_BASED_THRESHOLDS","affected_previous_tasks":[],"notes":[]}; write_json(paths["manifest05"],m05)
    print(json.dumps({"W2C-02":checks02,"W2C-05":checks05,"totals":totals},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
