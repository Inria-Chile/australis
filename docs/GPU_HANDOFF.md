# GPU production handoff

GPU work begins only after CPU contracts are frozen.

## Required inputs

- immutable gene IDs and sequence hashes;
- train, calibration and test assignments with leakage audit;
- one hash-sharded work manifest per representation;
- pinned model revision, pooling, truncation and precision;
- expected output dimension and dtype;
- destination with sufficient space and atomic shard semantics.

## Musa validation sequence

1. Confirm the OAR allocation and both assigned H100 GPUs.
2. Run `scripts/preflight_ood_gpu.py` in the locked GPU environment.
3. Execute one small representative probe and record peak VRAM, RAM and throughput.
4. Compare CPU and GPU k-nearest-neighbor scores on a fixed fixture when FAISS-GPU is used.
5. Launch the frozen shard universe with two GPUs and disjoint ownership.
6. Validate every output shard before marking its task complete.
7. Return only checksummed indexes and validated artifacts to CPU consolidation.

## CPU acceptance gate

After GPU production, CPU consolidation must establish all of the following before evaluation:

- every manifest ID is unique and belongs to the frozen reference universe;
- every shard row maps to the same ID recorded inside its payload;
- dimensions, dtypes, model identity and pooling policy are uniform and declared;
- every embedding value is finite;
- sequence-only and structure-conditioned controls cover the same evaluable IDs;
- package archives contain exactly the indexed members and match their recorded SHA-256 values.

Failed validation creates a new corrective run. It never edits a historical index or overwrites
the only artifact copy.

PyTorch-OOD and PUNCC may supply algorithms, but neither library is permitted to redefine function-class or homology partitions.
