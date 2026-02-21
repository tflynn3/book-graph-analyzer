# Story Generation

## Single scene

```bash
bga generate scene -g "Bilbo and Gandalf argue about risk" -c Bilbo -c Gandalf -p "Bag End" -w data/worldbible/hobbit_ch1_bible.json
```

## Outline

```bash
bga generate outline --help
```

## Full novel

```bash
bga generate novel --outline data/output/outline.json --checkpoint data/checkpoints --resume --world-bible data/worldbible/hobbit_ch1_bible.json --output data/output/novel_draft.md
```
