# GitHub and Zenodo publication

This repository is prepared as a release candidate for publication under the
Inria-Chile organization. Scientific datasets and model artifacts are not
committed to Git: they remain external and are identified by checksummed
manifests.

## Institutional gates

Before making the repository public:

1. Confirm the final repository name and maintainer permissions with Inria Chile.
2. Complete the redistribution review for ACE, supplementary data, model weights,
   derived embeddings, predicted structures, and benchmark outputs.
3. Confirm the final author list, affiliations, citation metadata, and Zenodo
   community or collection.
4. Create a private GitHub repository and run the CI workflow from a clean clone
   before changing its visibility. CI validates the portable code layer and
   fixtures; production-scale reruns remain a separate gated workflow requiring
   external data and model access.

## First push

Create an empty repository in the Inria-Chile organization, without generating
README, license, or gitignore files. Then run:

```bash
git remote add origin git@github.com:Inria-Chile/australis.git
git push -u origin main
git push origin v0.1.0-rc1
```

If the approved repository name differs, replace only the remote URL. Do not
rewrite paths or package metadata merely to match the GitHub slug.

## Release verification

The GitHub Actions workflow must pass on Python 3.11 and 3.12. It performs:

- locked dependency synchronization with `uv`;
- Ruff linting;
- the complete contract test suite;
- wheel and source-distribution construction;
- release-content auditing;
- isolated wheel installation and CLI smoke testing; and
- upload of the wheel, source distribution, and release audit as CI artifacts.

Create the GitHub prerelease from tag `v0.1.0-rc1` only after CI passes. Attach
the wheel, source distribution, `release_readiness.json`, and `SHA256SUMS` from
the validated Grid'5000 run.

## Zenodo linkage

After institutional approval:

1. Enable Zenodo for the GitHub repository.
2. Publish the first stable GitHub release, preferably `v0.1.0`.
3. Record the resulting DOI in `CITATION.cff`, the README, and proposal material.
4. Publish large scientific artifacts as separate versioned Zenodo deposits with
   licenses, checksums, and manifest links; do not add them to the source repository.
5. Re-run the release gate and tag the metadata-only DOI update.

## Reproducibility boundary

The repository provides portable code, configuration, manifests, tests, and
provenance contracts. Access-controlled or redistribution-restricted inputs must
be obtained from their authoritative sources. Historical production artifacts
are evidence for the release candidate, not substitutes for a clean-clone rerun.
