# Clean-room reproduction

## Scope

The public repository contains code, configurations, schemas, tests and small fixtures. ACE catalogs, model weights, embeddings, structures and checkpoints remain external and are addressed by immutable manifests.

## CPU reproduction

1. Install `uv` and clone the repository into an empty directory.
2. Run `uv sync --locked --extra data --extra ml --extra gpu --extra dev`.
3. Run `uv run ruff check .` and `uv run pytest -q`.
4. Export `POLARFUNC_DATA_ROOT`, `POLARFUNC_ARTIFACT_ROOT` and `POLARFUNC_OUTPUT_ROOT` to writable locations outside the checkout.
5. Create `manifests/ace_external_artifacts.json` with `polarfunc zenodo-manifest`.
6. Download or verify every source artifact before preprocessing.
7. Run each stage from its Hydra configuration and retain its resolved configuration, input manifest, output checksums and software provenance.

For a frozen representation, run `validate_embedding_manifest.py` before classification. For
packaged structural artifacts, run `validate_package_index.py` before deleting, moving or
publishing any source tree. These validators check scientific identity and storage integrity;
file existence alone is never treated as completion.

The full ACE download is large. A release must also provide a fixture profile that exercises every stage on small synthetic or redistributable records.

## Scientific invariants

- Function classes held out for OOD evaluation cannot appear in ID training.
- MMseqs groups cannot cross train, calibration and test partitions.
- Imputers, scalers, dimensionality reduction and calibration fit on TRAIN or the designated calibration set only.
- Sequence translation is frozen before labels or unknown-candidate status are consulted.
- Unknown-gene outputs are ranked hypotheses, not functional annotations.
- Structural representations remain interpretation-only until the preregistered quality gate passes.

## GPU handoff

CPU preprocessing produces immutable manifests and hash-sharded worklists. A GPU job consumes only one frozen manifest, writes atomic shards, and exports a completion index. CPU consolidation verifies IDs, dimensions, checksums and task coverage before any downstream evaluation. See `docs/GPU_HANDOFF.md`.

The handoff report must state whether an embedding was generated from sequence alone or from a
structure-conditioned payload. Comparisons between those representations use the exact
intersection of validated IDs and never impute a missing modality.

## Release evidence

A release candidate is complete only when a clean checkout can rebuild the fixture results, all public tables map to commands and resolved configurations, and external artifacts can be independently retrieved and verified. Both the source distribution and wheel must pass the release archive policy; AppleDouble files and Python caches are prohibited.
