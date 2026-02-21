# Pipeline Walkthrough

```bash
bga ingest data/texts/the_hobbit.txt --title "The Hobbit"
bga extract entities data/texts/the_hobbit.txt --title "The Hobbit" --show-new
bga pipeline full data/texts/the_hobbit.txt --title "The Hobbit"
```

Then use `bga worldbible --help` and `bga generate --help` for generation workflows.
