# Contributing

```bash
pip install -e .[dev,docs]
pytest
ruff check .
mypy src
mkdocs serve
```

Keep docs in sync when CLI behavior changes.

## Merge-ready gate (mandatory)

Do not mark a PR merge-ready unless the Hobbit 7-layer acceptance gate is PASS.

- CI check required: `hobbit-7layer-acceptance-gate`
- Local checker: `python scripts/check_hobbit_7layer_gate.py`
- Reference checklist: `docs/wiki/merge-ready-checklist.md`
