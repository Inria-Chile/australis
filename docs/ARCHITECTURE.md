# Architecture

The workflow is a directed sequence of immutable contracts:

1. public source inventory and checksum validation;
2. canonical ORF, unigene, AGC, sample and environment registries;
3. explicit measured/reconstructed/imputed environmental metadata;
4. frozen homology groups and train/validation/test assignments;
5. classical, genomic, protein and gated structural representations;
6. task-specific classifiers and context-aware fusion;
7. grouped uncertainty, calibration, null controls and domain-shift tests;
8. masked-known validation and evidence-tiered unknown-gene prioritization;
9. reports whose claims are machine-checkable against result tables.

All large edges in this graph are external artifacts. Git stores code, configuration, schemas, small fixtures and manifests. A run stores its resolved Hydra configuration, code revision, dependency lock, seeds, input/output hashes and resource metadata.

