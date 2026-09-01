import gzip

import pandas as pd

from polarfunc_repro.annotation import build_unigene_master, derive_function_flags

FUNCTIONAL = ["KEGG_KO", "PFAMs"]


def test_function_flags_separate_broad_and_agnostos_unknown():
    frame = pd.DataFrame(
        {
            "AGC_Cat": ["K", "GU", "SINGL"],
            "AGC_Singl_Cat": ["NA", "NA", "EU"],
            "KEGG_KO": ["K00001", "-", ""],
            "PFAMs": ["-", "-", "NA"],
        }
    )

    result = derive_function_flags(frame, functional_columns=FUNCTIONAL)

    assert result["strict_known"].tolist() == [True, False, False]
    assert result["strict_unknown"].tolist() == [False, True, True]


def test_unigene_master_keeps_only_representatives(tmp_path):
    source = tmp_path / "annotation.tsv.gz"
    columns = [
        "GENE_ID",
        "AGC_ID",
        "AGC_Cat",
        "AGC_Singl_Cat",
        "CDHit_ID",
        "KEGG_KO",
        "PFAMs",
    ]
    rows = [
        ["gene_a", "1", "K", "NA", "gene_a", "K1", "-"],
        ["gene_b", "1", "K", "NA", "gene_a", "K1", "-"],
        ["gene_c", "2", "GU", "NA", "gene_c", "-", "-"],
    ]
    with gzip.open(source, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")

    result = build_unigene_master(
        source, tmp_path / "master", functional_columns=FUNCTIONAL, chunksize=2
    )

    assert result["input_rows"] == 3
    assert result["representative_rows"] == 2
    output = pd.concat(pd.read_parquet(part["path"]) for part in result["parts"])
    assert output["GENE_ID"].tolist() == ["gene_a", "gene_c"]
