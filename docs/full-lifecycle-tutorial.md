# Full Lifecycle Tutorial (Ingest -> Analyze -> Generate)

This walkthrough covers the complete practical path from raw text to generated chapters.

## 1) Ingest source text

```bash
python -m book_graph_analyzer.cli ingest data/texts/the_hobbit.txt --title "The Hobbit"
```

## 2) Run full analysis pipeline

```bash
python -m book_graph_analyzer.cli pipeline full data/texts/the_hobbit.txt --title "The Hobbit"
```

What this gives you:

- extracted entities/relationships
- style analysis artifacts
- voice analysis artifacts
- graph-grounded context for generation

## 3) Build an outline

```bash
python -m book_graph_analyzer.cli generate outline \
  --character Bilbo \
  --from "Unexpected Party" \
  --to "Return to the Shire" \
  --chapters 5 \
  --output data/output/lifecycle_outline.json
```

## 4) Generate chaptered draft from outline

```bash
python -m book_graph_analyzer.cli generate novel \
  --outline data/output/lifecycle_outline.json \
  --checkpoint data/checkpoints \
  --resume \
  --output data/output/lifecycle_novel.md
```

## 5) Validate output quality

Inspect in `data/output/lifecycle_novel.md`:

- chapter-level continuity
- scene score blocks (`lore_score`, `style_score`, `narrative_score`)
- critique notes and obvious consistency issues

## 6) Iterate

- adjust anchor points (`--from`, `--to`)
- adjust chapter count (`--chapters`)
- re-run outline + novel
- keep `--checkpoint` and `--resume` for reliability on longer runs

## Known caveat

If `generate outline` fails when loading a world bible, run without `--world-bible` for now (schema mismatch issue in current model shape).
