# Data and model licenses

Large scientific artifacts are not committed. The public code release contains retrieval metadata,
schemas and checksums only. This review was last updated on 2026-09-01; live upstream terms must be
checked again for every tagged release.

| Resource | Frozen identifier | Upstream terms | Repository policy |
|---|---|---|---|
| ACE Southern Ocean Reference Gene Catalogs | Zenodo `14181291` | Zenodo record metadata states CC-BY-4.0; verify current terms before mirroring | External-only; retrieve from Zenodo and never commit catalogs |
| ESM2 | `facebook/esm2_t33_650M_UR50D` | MIT on the official model card | External model cache; pin the exact Hub revision |
| ESM3 | `esm3_sm_open_v1`, historical snapshot `47f0545b2b6daf26a93439a3cd610f4f7f3d5478` | The live Biohub model card and code repository state MIT; the production snapshot predates that transition | Code and manifests may be public; embeddings, structures and weights remain external pending institutional confirmation |
| GenomeOcean | `DOEJGI/GenomeOcean-500M`, revision `0d9f453925aca9c278505cf0103b6b4311092052` | BSD-like license published with the official model | External model cache; preserve copyright and disclaimer requirements |

## Authoritative sources

- ACE: <https://zenodo.org/records/14181291>
- ESM2: <https://huggingface.co/facebook/esm2_t33_650M_UR50D>
- ESM3 weights: <https://huggingface.co/biohub/esm3-sm-open-v1>
- ESM code: <https://github.com/Biohub/esm/blob/main/LICENSE.md>
- GenomeOcean: <https://huggingface.co/DOEJGI/GenomeOcean-500M/blob/main/LICENSE>

The code repository may be published because it redistributes none of these scientific artifacts.
Redistribution of ACE data or historical ESM3-derived embeddings and structures requires a separate
institutional decision and is outside the software license.

ESM3 and GenomeOcean access may be gated by Hugging Face. Users must obtain
their own approval and authenticate locally; credentials must never be placed
in Git, manifests or publicly visible Space variables.
