# Full End-to-End Example (Real Run)

This is a real run that generated a multi-chapter draft in this repo.

## Step 1: Generate outline

```bash
python -m book_graph_analyzer.cli generate outline \
  --character Bilbo \
  --from "Unexpected Party" \
  --to "Return to the Shire" \
  --chapters 5 \
  --output data/output/example_outline.json
```

Expected output:

- `OK Outline saved to data\\output\\example_outline.json`
- `Chapters generated: 5`

## Step 2: Generate novel draft

```bash
python -m book_graph_analyzer.cli generate novel \
  --outline data/output/example_outline.json \
  --checkpoint data/checkpoints \
  --resume \
  --output data/output/example_novel.md
```

Expected output:

- `Generating novel from outline...`
- `Chapter 1 / 5 ... Chapter 5 / 5 ...`
- `OK Story saved to data\\output\\example_novel.md`
- `Chapters: 5 | Scenes: 5 | Words: 3,254`

## Output files

- `data/output/example_outline.json`
- `data/output/example_novel.md`
