# Release checklist

## Engineering

- [x] `uv sync --locked` and the Linux release build succeed.
- [x] Ruff and 41 CPU contract tests pass on Grid'5000.
- [x] `scripts/check_release_readiness.py` reports `PASS` for wheel and sdist.
- [x] No personal paths, credentials, data, embeddings, PDBs or generated runs enter the release.
- [x] CI tests Python 3.11 and 3.12 and performs a clean wheel-install smoke test.

## Scientific reproducibility

- Zenodo and Figshare manifests contain size and cryptographic checksums.
- Frozen registries and split manifests have exact cardinality reports.
- Model revisions, pooling, precision, truncation and seeds are explicit.
- Published tables point to resolved configurations and checksummed outputs.
- Leakage, null-control, calibration and domain-shift audits are archived.

## Governance

- [ ] Inria Chile approves the repository name and maintainer list.
- [ ] Data and model redistribution decisions are complete.
- [ ] Citation metadata includes the final authors and archived code DOI.
- [ ] The private review repository is created under the Inria Chile organization.

The release candidate passes the engineering audit. Governance, external-license review and the
institutional GitHub/Zenodo publication remain explicit release gates.
