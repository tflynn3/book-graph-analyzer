# Book Graph Analyzer

Transform novels into knowledge graphs, dense proposition layers, style fingerprints, world bibles, and generation-ready context.

## Extraction Paths

Use the seeded graph path when you want to rebuild the Tolkien lore graph:

```bash
bga pipeline full data/texts/the_hobbit.txt --title "The Hobbit" --author "J.R.R. Tolkien"
bga corpus process tolkien_works
```

- `bga pipeline full` runs the shared seeded extractor for one book and, when Neo4j is available, writes passages, entity mentions, canon relationships, dense proposition nodes, unresolved references, style, and voice artifacts.
- `bga corpus process` runs the same seeded extraction path across every book in a corpus and also refreshes the cross-book entity index.
- `bga analyze` remains the zero-seed exploratory path. It is useful for ad hoc novels and JSON inspection, but it is not the primary Tolkien lore-graph rebuild workflow.
- The splitter now prefers real body chapter markers like `_Chapter 1_` over contents-page entries like `Chapter 1 A Long-expected Party`, and trims obvious foreword/appendix scaffolding from the public LOTR text dumps before graph extraction.

## Graph Layers

- The canon graph stays relatively strict: grounded entity-to-entity relationships that resolve cleanly.
- The proposition layer is intentionally denser: sentence-level action, movement, speech, possession, attribute, and state propositions with per-argument grounding.
- Unresolved proposition arguments are preserved as `UnresolvedReference` nodes so shadow-graph generation can use the semantic scaffold without pretending every mention is canonically resolved.
- Canon-edge projection is type-gated, so invalid pairs like `Character -> TRAVELED_TO -> Character` are dropped instead of polluting the strict graph.
- Voice profiles only attach to already grounded character nodes; unmatched quote-attribution guesses are skipped rather than creating floating `Character` nodes.
- Voice extraction now handles both single-quoted and double-quoted dialogue, filters scare quotes before profiling, merges speaker aliases that map to the same canonical entity during graph writes, and preserves cumulative corpus-level voice profiles instead of overwriting them book by book.

## Model Evaluation

Use the local MLX benchmark when you want to compare quantized models that run directly on Apple Silicon:

```bash
python scripts/evaluate_hf_unresolved.py --output data/evals/hf_unresolved_results.json
```

Use the routed Hugging Face provider benchmark when you want to spend HF inference credits on larger hosted models:

```bash
python scripts/evaluate_hf_provider_unresolved.py \
  --model Qwen/Qwen2.5-72B-Instruct \
  --candidate-shortlist \
  --output data/evals/hf_provider_qwen72b_results.json
```

- `evaluate_hf_provider_unresolved.py` uses the current HF auth token and does not require Neo4j, because it ships with the benchmark and a frozen Tolkien character inventory.
- `--candidate-shortlist` swaps the full inventory prompt for a small retrieved candidate set and improves alias-heavy grounding with hosted models like `Qwen2.5-72B-Instruct`.
- Thinking-capable models often need model-specific prompt control like `/no_think` or larger token budgets before they emit the final structured answer line.

Use the live unresolved-repair pass when you want to enrich the current Neo4j graph without rebuilding the whole corpus:

```bash
bga lore resolve-unresolved \
  --model Qwen/Qwen2.5-72B-Instruct \
  --candidate-limit 6 \
  --limit 50 \
  --json-out data/evals/live_unresolved_resolution_report.json
```

- `bga lore resolve-unresolved` reads the current `:UnresolvedReference` queue, filters to character-like mentions, and runs a staged hosted-model pass (`reject` vs `character`, then `existing` vs `new_entity`).
- `--apply-existing` is on by default and only auto-applies safe `existing` matches that map to a current `:Character` inventory node; `new_entity` results remain reviewable suggestions on the unresolved node.
- Suggestions are written back to Neo4j as `llm_resolution_*` properties so you can audit what the hosted model proposed before deciding whether a full graph rebuild is warranted.

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
