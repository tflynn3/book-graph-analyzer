# Installation

```bash
git clone https://github.com/tflynn3/book-graph-analyzer.git
cd book-graph-analyzer
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -e .
pip install -e .[docs]
python -m spacy download en_core_web_sm
docker compose up -d neo4j
```

Set environment values in `.env` if needed:

```env
BGA_NEO4J_URI=bolt://localhost:7687
BGA_NEO4J_USER=neo4j
BGA_NEO4J_PASSWORD=bookgraph123
```

LLM provider defaults to Ollama. To use OpenAI for lore/corpus event extraction:

```env
# Choose provider: ollama | huggingface | openai
BGA_LLM_PROVIDER=openai

# Required for OpenAI provider
BGA_OPENAI_API_KEY=sk-...

# Optional (defaults shown)
BGA_OPENAI_MODEL=gpt-4o-mini
BGA_OPENAI_BASE_URL=https://api.openai.com/v1
```
