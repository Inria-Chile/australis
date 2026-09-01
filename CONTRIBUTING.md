# Contributing

Use a focused branch and include tests for every behavioral change. Run:

```bash
uv sync --locked --extra data --extra ml --extra dev
uv run ruff check .
uv run pytest -q
```

Do not commit private paths, credentials, restricted datasets, model caches, embeddings, structures or generated runs. Scientific changes must state the affected data split, fitting scope, random seeds and expected impact on claims. New external resources require a license and checksum entry.
