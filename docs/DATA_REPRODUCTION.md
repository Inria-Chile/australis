# Data Reproduction Contract

This repository separates portable code from large scientific artifacts. The
source data, model weights, embeddings, structures and benchmark outputs are
not committed to GitHub. They are retrieved or mounted through immutable
manifests and verified before use.

## Current release scope

This release provides executable implementations for every stage in
`configs/pipeline/full.yaml`, including preprocessing, ecology, homology-aware
splits, sequence features, model embeddings, classifiers, calibration, OOD
evaluation and discovery. A clean-clone rerun still requires the external ACE
catalogs, model access and derived-artifact manifests described below; large or
access-controlled files are never committed to GitHub.

## Primary ACE source

The primary source is the published Zenodo record for the Southern Ocean
Reference Gene Catalogs:

- Record: https://zenodo.org/records/14181291
- API record: https://zenodo.org/api/records/14181291
- DOI: https://doi.org/10.5281/zenodo.14181291
- Local metadata snapshot: `manifests/ace_zenodo_record_14181291.json`

The metadata snapshot contains the 14 ACE files, direct download URLs, file
sizes and upstream MD5 checksums. Generate a fresh normalized retrieval
manifest from the live record before downloading:

```bash
uv run polarfunc zenodo-manifest 14181291 \
  --output manifests/ace_external_artifacts.json
uv run polarfunc download manifests/ace_external_artifacts.json \
  --destination "$POLARFUNC_DATA_ROOT"
uv run polarfunc verify-manifest manifests/ace_external_artifacts.json \
  --root "$POLARFUNC_DATA_ROOT" --output "$POLARFUNC_OUTPUT_ROOT/ace_verify.json"
```

The largest files are multi-gigabyte compressed catalogs. They require
substantial storage and should be downloaded atomically. The Zenodo license
and the institutional redistribution decision must be checked again before
any mirror or archival deposit is created.

## Models and derived artifacts

Exact model sources and revisions are recorded in
`manifests/model_sources.json`. Weights are retrieved from their authoritative
model cards and are not mirrored by this repository. Historical embeddings,
PDB structures and benchmark outputs are derived artifacts; they require a
separate checksum manifest and must not be silently treated as reproducible
from source data alone.

ESM3 and GenomeOcean may require an approved Hugging Face account or gated-model
access. Access is granted by the upstream model card, not by this repository. For
a production rerun, request access on the model-card page if prompted, then
authenticate only on the compute node or local machine that will download the
weights:

```bash
hf auth login
hf auth whoami
hf download biohub/esm3-sm-open-v1 --revision 47f0545b2b6daf26a93439a3cd610f4f7f3d5478
hf download DOEJGI/GenomeOcean-500M --revision 0d9f453925aca9c278505cf0103b6b4311092052
```

ESM2 is public, but its revision is also pinned in `manifests/model_sources.json`.
The `HF_TOKEN` environment variable may be used in an ephemeral job environment
instead of `hf auth login`; never place the token in Git, a README, a manifest,
a command file, a log, or a Space variable with public visibility. The repository
never stores credentials. After downloading, record the approved revision, local
cache location outside the repository, and SHA256 checksums in the external
artifact manifest before launching embedding generation.

The complete clean-room boundary is therefore:

1. retrieve the ACE source files from Zenodo;
2. retrieve the exact model revision permitted by its license;
3. regenerate preprocessing, splits and representations using the pinned
   configuration and environment;
4. validate every output by IDs, dimensions, checksums and frozen task
   manifests; and
5. publish only metadata, code and outputs whose redistribution is authorized.

The fixture in `manifests/example_external_artifacts.json` is the only fully
redistributable test input committed to this repository. It validates the
software contract without pretending to reproduce production-scale biology.
