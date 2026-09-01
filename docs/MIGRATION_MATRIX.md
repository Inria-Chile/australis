# Historical migration matrix

| Public stage | Internal source families | Required migration evidence |
|---|---|---|
| ingest | `00_preflight.py`, `01_inventory_data.py`, `02_capture_faure_reference.py` | source fixture, checksum test, schema report |
| preprocess | `05_build_unigene_master.py`, translation and candidate scripts | identifier invariants, cardinality tests, exclusion report |
| ecology | environment, water-mass, CAG and adjusted-ecology scripts | measured/imputed flags, sample-group tests, confounder specification |
| split | MMseqs clustering and benchmark-freezing scripts | immutable split manifest, leakage tests |
| classical_features | CPU feature generation | deterministic fixture and feature schema |
| embeddings | GenomeOcean, ESM2 and ESM3 runners | pinned model revisions, pooling contract, shard validator |
| models | CPU evaluation and fusion scripts | train-only transforms, seed matrix, model contract |
| evaluation | comparison, bootstrap, null and robustness scripts | paired group bootstrap fixture and calibration tests |
| discovery | ecology alignment, retrieval and candidate evidence | masked-known fixture, evidence-tier rules, no-annotation claim test |
| reporting | finalizers and claim matrices | implemented in initial scaffold |

Migration never consists of copying scripts blindly. Each stage must be converted into importable functions, configured through Hydra, tested on a small public fixture and compared against a frozen internal checksum or summary statistic.

