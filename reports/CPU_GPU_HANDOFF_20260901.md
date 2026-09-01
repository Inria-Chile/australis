# CPU to GPU handoff, 2026-09-01

## Frozen CPU evidence

| Contract | Run | Result |
| --- | --- | --- |
| ACE source artifacts | `cpu_data_validation_20260901T1730` | 14/14 Zenodo artifacts passed MD5 verification |
| Unigene master | `cpu_data_validation_20260901T1730` | 89,739,060 rows across 36 Parquet parts; schema and checksums passed |
| GenomeOcean sequence index | `cpu_genomeocean_index_corrected_20260901T1801` | 91,663 unique IDs, 2 shards, 1,536-dimensional float32 vectors |
| ESM3 sequence-only index | `cpu_esm3_sequence_validation_20260901T1820` | 91,663 unique IDs, 128 shards, 1,536-dimensional finite float32 vectors |
| ESM3 structure-conditioned index | `cpu_esm3_structure_validation_20260901T1826` | 91,663 unique IDs, 128 shards, 1,536-dimensional finite float32 vectors |
| Structural package | `cpu_structure_package_validation_20260901T1832` | 366,652 members, 91,663 complete IDs, 256 archives, all archive hashes passed |
| Stable PDB bucket index | `cpu_data_validation_20260901T1730` | 91,663 unique IDs and paths across all 256 buckets; 299-414 PDBs per bucket |
| Publication release | `cpu_repo_release_verified_20260901T1632Z` | Ruff passed, 41/41 tests passed, wheel/sdist policy passed, all release hashes passed |

The two ESM3 indexes cover exactly the same frozen `structure_common_final` universe. The
structural package contains one PDB, sequence embedding, structure embedding and metadata member
for every accepted ID. GenomeOcean uses model revision
`0d9f453925aca9c278505cf0103b6b4311092052`; ESM3 uses `esm3_sm_open_v1`.

## Musa-2 acceptance sequence

1. Record the OAR job, host, two assigned H100 devices, driver and CUDA runtime.
2. Create or synchronize the locked environment with the `gpu` and `ood` dependency groups.
3. Run `scripts/preflight_ood_gpu.py`; exactly two visible GPUs with at least 80 GiB each are
   required.
4. Resolve and record the FAISS-GPU build only after confirming the node CUDA runtime.
5. Run a small fixed probe for PyTorch-OOD, PUNCC and FAISS-GPU, recording peak VRAM, host RAM,
   throughput and numerical agreement with the CPU reference.
6. Consume the frozen open-set and homology partitions without redefining labels or split roles.
7. Assign disjoint shards to both H100 devices and use atomic outputs plus resumable completion
   indexes.
8. Return checksummed outputs to CPU validation before scientific comparison or reporting.

## Work reserved for GPU

- production PyTorch-OOD detector evaluation over the frozen representations;
- FAISS-GPU nearest-neighbour scoring and CPU/GPU equivalence checks;
- PUNCC calibration only after its exchangeability assumptions are audited;
- expanded representation experiments that genuinely require accelerator inference or training.

No ESM3 or GenomeOcean regeneration is required for the current 91,663-gene benchmark. The
remaining GPU work is experimental evaluation and future model expansion, not input repair.
