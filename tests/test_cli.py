from polarfunc_repro.cli import build_parser


def test_bucket_index_accepts_recursive_patterns():
    args = build_parser().parse_args(
        [
            "bucket-index",
            "--root",
            "structures",
            "--pattern",
            "*.pdb",
            "--output",
            "index.jsonl",
        ]
    )
    assert args.paths == []
    assert args.pattern == ["*.pdb"]
