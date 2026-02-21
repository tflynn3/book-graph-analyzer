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
