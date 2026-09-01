# Storage and sharding contract

Large artifacts are never stored as one file per gene in Git. Runtime roots are supplied through `POLARFUNC_*` environment variables.

## Tabular data

Canonical registries use Parquet with explicit schemas. Large registries are written as immutable parts. Partition keys must have bounded cardinality; scientific labels and train/test assignments are not used as physical partition keys unless the split is already frozen.

## Embeddings

Dense matrices use Zarr, HDF5 or contiguous NumPy shards with a companion Parquet index. Every index records the biological identifier, shard, row offset, dimension, dtype, sequence hash, model revision and pooling rule.

## Structures and numerous files

PDB and related per-gene files receive a deterministic bucket from the first 64 bits of `SHA256(relative_path) modulo bucket_count`. Production archives should be named `bucket_0000.tar.zst`, accompanied by an index that records artifact ID, archive member, byte size, checksum and structural QC. Bucket count and compression level are frozen after an I/O probe; changing either creates a new storage version.

The `polarfunc bucket-index` command creates the deterministic index. Archive packing remains a separate stage so interrupted compression cannot invalidate the canonical file index.

## Atomicity

Downloads use hidden `.part` files and an atomic rename after size and checksum validation. Scientific stages write into a new run directory and publish a completion manifest only after validation. Existing valid outputs are never overwritten in place.
