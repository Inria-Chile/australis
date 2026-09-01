# Reproducibility evidence for the INACH proposal

POLAR-FUNC has a validated public-code release candidate rather than only a prospective software
plan. The repository uses Hydra configuration, a locked `uv` environment, machine-independent
runtime paths, immutable manifests, provenance capture, bounded-memory catalog processing,
deterministic sharding and automated release audits. On 2026-09-01, its Linux/Grid'5000 gate passed
Ruff, 41 contract tests, wheel and source-distribution construction, archive-content policy and
independent SHA-256 verification.

The same gate verified 14/14 ACE source artifacts, a 89,739,060-row canonical unigene registry,
91,663 matched GenomeOcean embeddings, 91,663 matched ESM3 sequence and structure embeddings, and
366,652 complete package members. A separate index contains 91,663 unique PDB paths across all 256
deterministic buckets. Large scientific artifacts remain outside Git and are referenced by frozen
indexes and checksums.

This evidence demonstrates engineering feasibility and production-scale data control. It is not a
claim that every historical scientific stage has already been migrated or independently reproduced.
The public release path remains staged: preserve validated artifacts, migrate one scientific
contract at a time, compare against frozen internal outputs, and publish permitted material only
after institutional and license review.
