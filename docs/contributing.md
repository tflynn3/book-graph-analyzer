# Contributing

```bash
pip install -e .[dev,docs]
pytest
ruff check .
mypy src
mkdocs serve
```

Keep docs in sync when CLI behavior changes.
