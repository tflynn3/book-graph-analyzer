# Book Graph Analyzer

Transform novels into knowledge graphs, style fingerprints, world bibles, and generation-ready context.

## Docs wiki (GitHub Pages)

This repo now uses MkDocs Material for wiki/docs.

- Docs source: `docs/`
- Site config: `mkdocs.yml`
- Deploy workflow: `.github/workflows/docs-pages.yml`

### Local preview

```bash
pip install -e .[docs]
mkdocs serve
```

### Build

```bash
mkdocs build --strict
```
