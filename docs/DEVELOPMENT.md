# Development Guide

How to set up and work on Book Graph Analyzer.

---

## Prerequisites

- **Python 3.11+** — We use modern Python features
- **Docker** — For running Neo4j locally
- **Git** — Obviously

Optional but recommended:
- **NVIDIA GPU + CUDA** — For local LLM inference
- **Ollama** — For running local language models

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/tflynn3/book-graph-analyzer.git
cd book-graph-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Core dependencies
pip install -e .

# Development dependencies
pip install -e ".[dev]"

# Local LLM support (optional)
pip install -e ".[local-llm]"
```

### 4. Start Neo4j

```bash
docker compose up -d
```

Wait ~30 seconds for Neo4j to start, then verify:
- Open http://localhost:7474 in your browser
- Login with `neo4j` / `bookgraph123`

### 5. Verify installation

```bash
bga status
```

You should see:
```
Book Graph Analyzer Status

Neo4j URI: bolt://localhost:7687
✓ Neo4j connected
```

---

## Project Structure

```
book-graph-analyzer/
├── src/book_graph_analyzer/    # Main package
│   ├── cli.py                  # Command-line interface
│   ├── config.py               # Configuration management
│   ├── graph/                  # Neo4j interface
│   ├── ingest/                 # Text loading and splitting
│   ├── extract/                # Entity and relationship extraction (TODO)
│   ├── analyze/                # Style analysis (TODO)
│   └── models/                 # Pydantic data models
├── data/
│   ├── seeds/                  # Pre-built entity lists
│   ├── texts/                  # Your book files (gitignored)
│   └── exports/                # Exported data
├── tests/                      # Test suite
├── docs/                       # Documentation
└── docker-compose.yml          # Neo4j container
```

---

## Common Commands

### CLI

```bash
# Check system status
bga status

# Ingest a book
bga ingest data/texts/the_hobbit.txt --title "The Hobbit"

# Search passages
bga search "ring"

# Graph stats
bga graph stats
```

### Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=book_graph_analyzer

# Type checking
mypy src/

# Linting
ruff check src/

# Format code
ruff format src/
```

### Docker

```bash
# Start Neo4j
docker compose up -d

# Stop Neo4j
docker compose down

# Stop and remove data (fresh start)
docker compose down -v

# View logs
docker compose logs -f neo4j
```

---

## Configuration

Settings can be configured via environment variables or a `.env` file:

```bash
# .env file
BGA_NEO4J_URI=bolt://localhost:7687
BGA_NEO4J_USER=neo4j
BGA_NEO4J_PASSWORD=bookgraph123
BGA_OLLAMA_MODEL=llama3.1:8b
```

See `config.py` for all available settings.

---

## Adding Your Books

Place your book files in `data/texts/`:

```
data/texts/
├── the_hobbit.txt
├── fellowship_of_the_ring.epub
└── ...
```

Supported formats:
- `.txt` — Plain text
- `.epub` — EPUB ebooks

**Note**: Book files are gitignored. You must provide your own copies.

---

## Working with Neo4j

### Browser

Open http://localhost:7474 for the Neo4j Browser. You can:
- Run Cypher queries
- Visualize the graph
- Explore relationships

### Example Queries

```cypher
-- All nodes
MATCH (n) RETURN n LIMIT 25

-- All characters
MATCH (c:Character) RETURN c

-- Character relationships
MATCH (c1:Character)-[r]->(c2:Character)
RETURN c1.canonical_name, type(r), c2.canonical_name
LIMIT 50

-- Passages mentioning a character
MATCH (c:Character {canonical_name: "Gandalf"})<-[:MENTIONED_IN]-(p:Passage)
RETURN p.text, p.book, p.chapter
LIMIT 10
```

### Reset Database

To clear all data and start fresh:

```cypher
MATCH (n) DETACH DELETE n
```

Or restart with fresh volumes:

```bash
docker compose down -v
docker compose up -d
```

---

## Local LLM Setup (Optional)

For entity/relationship extraction without API costs:

### 1. Install Ollama

Download from https://ollama.ai

### 2. Pull a model

```bash
# Recommended: good balance of quality and speed
ollama pull llama3.1:8b

# Alternative: smaller, faster
ollama pull mistral:7b

# Alternative: larger, better quality
ollama pull llama3.1:70b  # Requires lots of VRAM
```

### 3. Verify

```bash
ollama list
```

The application will auto-detect Ollama at `http://localhost:11434`.

---

## Testing

### Run all tests

```bash
pytest
```

### Run specific test file

```bash
pytest tests/test_splitter.py
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage

```bash
pytest --cov=book_graph_analyzer --cov-report=html
# Open htmlcov/index.html
```

---

## Code Style

We use:
- **Ruff** for linting and formatting
- **MyPy** for type checking
- **Pydantic** for data validation

### Pre-commit hooks (optional)

```bash
pre-commit install
```

This will run linting/formatting on every commit.

---

## Troubleshooting

### Neo4j won't start

Check Docker logs:
```bash
docker compose logs neo4j
```

Common issues:
- Port already in use (7474 or 7687)
- Docker not running
- Insufficient memory

### Can't connect to Neo4j

- Make sure Docker is running: `docker ps`
- Check the container is healthy: `docker compose ps`
- Verify credentials match `.env` file

### Import errors

Make sure you installed in editable mode:
```bash
pip install -e .
```

### spaCy model not found

Download the English model:
```bash
python -m spacy download en_core_web_sm
```
