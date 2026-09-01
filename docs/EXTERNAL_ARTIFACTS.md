# External artifacts

Data catalogs, embeddings, model weights, checkpoints and PDB files are intentionally excluded from Git. Each release manifest must provide a stable artifact identifier, scientific role, URI or accession, byte size, cryptographic checksum, license and producing configuration.

Machine-local paths are runtime concerns supplied through `POLARFUNC_*` environment variables. They must never appear in committed manifests. A future DOI snapshot should contain only artifacts whose redistribution terms permit deposition; otherwise it should contain a checksum index and retrieval instructions.

