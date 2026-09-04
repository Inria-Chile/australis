#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from polarfunc.provenance import sha256_file, utc_now, write_json

DNA="ACGT"; AA="ACDEFGHIKLMNPQRSTVWY"; CODONS=["".join(x) for x in itertools.product(DNA,repeat=3)]; KMERS=["".join(x) for k in range(1,7) for x in itertools.product(DNA,repeat=k)]


def selected_fasta(path:Path,selected:set[str])->dict[str,str]:
    result={}; identifier=None; sequence=[]
    def emit():
        if identifier in selected: result[identifier]="".join(sequence).upper()
    with path.open() as handle:
        for line in handle:
            if line.startswith(">"):
                emit(); identifier=line[1:].split()[0]; sequence=[]
            elif identifier in selected: sequence.append(line.strip())
        emit()
    return result


def relative_counts(sequence:str,tokens:list[str],step:int=1)->np.ndarray:
    index={token:i for i,token in enumerate(tokens)}; values=np.zeros(len(tokens),dtype=np.float32); total=0
    width=len(tokens[0])
    for start in range(0,len(sequence)-width+1,step):
        token=sequence[start:start+width]
        if token in index: values[index[token]]+=1; total+=1
    if total: values/=total
    return values


def main()->int:
    started=utc_now(); clock=time.monotonic(); matches=sorted((ROOT/"manifests").glob("benchmark_v1_extended_*.parquet"))
    if len(matches)!=1: raise RuntimeError(f"Expected one extended benchmark, found {matches}")
    benchmark_path=matches[0]; benchmark=pd.read_parquet(benchmark_path).sort_values("CDHit_ID").reset_index(drop=True); ids=benchmark.CDHit_ID.astype(str).tolist(); selected=set(ids)
    feature_dir=ROOT/"artifacts/features"; report=ROOT/"reports/corrective/cpu_feature_generation.md"; manifest_path=ROOT/"manifests/corrective/W2C-07/manifest.json"
    outputs={"length_gc":feature_dir/"length_gc.parquet","codon":feature_dir/"codon_features.parquet","aa":feature_dir/"aa_composition.parquet","kmer":feature_dir/"kmer_1_6.npz","vocab":feature_dir/"kmer_vocab.json","ids":feature_dir/"kmer_ids.parquet","report":report,"manifest":manifest_path}
    for path in outputs.values():
        if path.exists(): raise FileExistsError(f"Refusing to overwrite: {path}")
        path.parent.mkdir(parents=True,exist_ok=True)
    nt=selected_fasta(ROOT/"artifacts/sequences/candidate_pool_v1.fna",selected); proteins=selected_fasta(ROOT/"artifacts/sequences/candidate_pool_v1.faa",selected)
    if set(nt)!=selected or set(proteins)!=selected: raise RuntimeError(f"FASTA selection mismatch nt={len(nt)} aa={len(proteins)} expected={len(selected)}")
    nt_sequences=[nt[x] for x in ids]; aa_sequences=[proteins[x] for x in ids]
    length_gc=benchmark[["CDHit_ID","length_nt","GC_fraction","protein_length"]].copy(); length_gc.to_parquet(outputs["length_gc"],index=False,compression="zstd")
    codon=np.vstack([relative_counts(seq,CODONS,3) for seq in nt_sequences]); gcpos=np.zeros((len(ids),3),dtype=np.float32)
    for i,seq in enumerate(nt_sequences):
        for position in range(3):
            bases=seq[position::3]; valid=sum(base in DNA for base in bases); gcpos[i,position]=sum(base in "GC" for base in bases)/valid if valid else 0
    codon_frame=pd.DataFrame(codon,columns=[f"codon_{x}" for x in CODONS]); codon_frame.insert(0,"CDHit_ID",ids)
    for position in range(3): codon_frame[f"GC{position+1}"]=gcpos[:,position]
    codon_frame.to_parquet(outputs["codon"],index=False,compression="zstd")
    aa=np.vstack([relative_counts(seq,list(AA),1) for seq in aa_sequences]); aa_frame=pd.DataFrame(aa,columns=[f"aa_{x}" for x in AA]); aa_frame.insert(0,"CDHit_ID",ids); aa_frame.to_parquet(outputs["aa"],index=False,compression="zstd")
    vocabulary={token:index for index,token in enumerate(KMERS)}; vectorizer=CountVectorizer(analyzer="char",ngram_range=(1,6),lowercase=False,vocabulary=vocabulary,dtype=np.float32); raw=vectorizer.transform(nt_sequences)
    blocks=[]; start=0
    for k in range(1,7):
        width=4**k; blocks.append(normalize(raw[:,start:start+width],norm="l1",axis=1,copy=True)); start+=width
    matrix=sparse.hstack(blocks,format="csr",dtype=np.float32); sparse.save_npz(outputs["kmer"],matrix,compressed=True); pd.DataFrame({"CDHit_ID":ids}).to_parquet(outputs["ids"],index=False,compression="zstd")
    write_json(outputs["vocab"],{"ordering":"lexicographic A,C,G,T within cumulative k=1..6","dimensions":len(KMERS),"tokens":KMERS})
    checks={"feature_rows_match_extended":len(ids)==len(benchmark)==matrix.shape[0],"kmer_dimension":matrix.shape[1]==5460,"codon_dimension":codon.shape[1]+3==67,"aa_dimension":aa.shape[1]==20,"no_nan_inf_dense":bool(np.isfinite(codon).all() and np.isfinite(gcpos).all() and np.isfinite(aa).all()),"no_nan_inf_sparse":bool(np.isfinite(matrix.data).all()),"core_ids_subset":set(pd.read_parquet(ROOT/"manifests/benchmark_v1_core_50k.parquet",columns=["CDHit_ID"]).CDHit_ID).issubset(selected),"vocabulary_deterministic":KMERS[0]=="A" and KMERS[-1]=="TTTTTT" and len(set(KMERS))==5460}
    if not all(checks.values()): raise RuntimeError(f"W2C-07 failed: {checks}")
    elapsed=time.monotonic()-clock; report.write_text("\n".join(["# CPU feature generation","",f"- Extended benchmark: `{len(ids):,}`",f"- Walltime: `{elapsed:.1f}` seconds","- Length/GC: 4 columns including ID","- Codons + GC1/2/3: 67 features","- AA composition: 20 features","- Cumulative nucleotide k=1..6: 5,460 sparse float32 features","","Each k block is independently L1-normalized by its valid A/C/G/T k-mer count. Core IDs are subset without recomputation.",""]))
    def out(path:Path): return {"path":str(path.relative_to(ROOT)),"sha256":sha256_file(path),"rows":pq.ParquetFile(path).metadata.num_rows if path.suffix==".parquet" else None}
    produced=[outputs[x] for x in ("length_gc","codon","aa","kmer","vocab","ids","report")]
    write_json(manifest_path,{"task_id":"W2C-07","status":"PASS","started_at":started,"completed_at":utc_now(),"host":socket.getfqdn(),"oar_job_id":os.environ.get("OAR_JOB_ID"),"git_commit":os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),"command":" ".join(sys.argv),"inputs":[{"path":str(benchmark_path.relative_to(ROOT)),"sha256":sha256_file(benchmark_path),"rows":len(ids)}],"parameters":{"dtype":"float32","kmer_range":[1,6],"kmer_normalization":"per-k L1"},"outputs":[out(path) for path in produced],"validation":{"passed":True,"checks":checks},"scientific_decision":"CPU_REPRESENTATIONS_READY","affected_previous_tasks":["W2-13"],"notes":[]})
    print(json.dumps({"status":"PASS","rows":len(ids),"shape":matrix.shape,"nnz":matrix.nnz,"elapsed_seconds":elapsed},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
