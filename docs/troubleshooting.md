# Troubleshooting

- `bga` missing: `pip install -e .`
- Neo4j connection errors: check `docker compose ps` and `.env`
- spaCy model missing: `python -m spacy download en_core_web_sm`
- For long runs, use `--checkpoint` + `--resume`
