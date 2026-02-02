# Architecture

Technical design for the Book Graph Analyzer system.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           BOOK GRAPH ANALYZER                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │
│  │   INGEST    │───▶│   EXTRACT   │───▶│   RESOLVE   │───▶│  STORE   │ │
│  │             │    │             │    │             │    │          │ │
│  │ Text files  │    │ Entities    │    │ Canonical   │    │ Neo4j    │ │
│  │ Splitting   │    │ Relations   │    │ Matching    │    │ Graph    │ │
│  │ Metadata    │    │ Attributes  │    │ Aliases     │    │          │ │
│  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │
│                                                                   │     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐           │     │
│  │   ANALYZE   │    │   PROFILE   │    │  SYNTHESIZE │           │     │
│  │             │    │             │    │             │◀──────────┘     │
│  │ Style       │    │ Character   │    │ World Bible │                 │
│  │ Metrics     │    │ Voices      │    │ Rules       │                 │
│  │ Patterns    │    │ Patterns    │    │ Themes      │                 │
│  └─────────────┘    └─────────────┘    └─────────────┘                 │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                            │                                            │
│                            ▼                                            │
│                    ┌─────────────┐                                      │
│                    │   GENERATE  │                                      │
│                    │             │                                      │
│                    │ Context     │                                      │
│                    │ Assembly    │                                      │
│                    │ Prompting   │                                      │
│                    └─────────────┘                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Ingest Layer

**Responsibility**: Load text files and split them into processable units.

**Input**: EPUB, TXT, or other text formats
**Output**: Structured passages with metadata

**Splitting Strategy**:
- Books → Chapters → Paragraphs → Sentences
- Each unit retains full provenance (book, chapter, paragraph index, sentence index)
- Sentence splitting must handle:
  - Abbreviations (Mr., Dr., etc.)
  - Dialogue punctuation
  - Poetry and songs (different splitting rules)

**Metadata Captured**:
- Source file path
- Book title (from filename or frontmatter)
- Chapter title/number
- Position indices at each level
- Character offset for precise linking

**Design Considerations**:
- Lazy loading for large files
- Streaming processing where possible
- Checkpoint/resume for long-running ingestion

---

### Extract Layer

**Responsibility**: Identify entities and relationships in text.

**Sub-components**:

#### Entity Extractor
- Named Entity Recognition for characters, places, objects
- Custom handling for Tolkien-specific names
- Pronoun resolution (coreference) where feasible
- Confidence scoring for uncertain extractions

#### Relationship Extractor  
- Subject-verb-object pattern detection
- Relationship type classification
- Implicit relationship inference
- Multi-entity relationship handling

#### Attribute Extractor
- Character descriptions (physical, behavioral)
- Place descriptions (geography, atmosphere)
- Temporal markers (when events occur)
- Sentiment/tone of passages

**Model Options** (local-first):

| Task | Primary Option | Fallback |
|------|----------------|----------|
| NER | Fine-tuned spaCy or local LLM | Cloud LLM API |
| Coreference | Local LLM | neuralcoref (limited) |
| Relationships | Local LLM (structured output) | Rule-based patterns |
| Attributes | Local LLM | Keyword extraction |

**Local LLM Strategy**:
- Target 7B-13B parameter models that fit in 12GB VRAM
- Quantized models (GGUF format) for efficiency
- Ollama or llama.cpp as runtime
- Batched inference for throughput

---

### Resolve Layer

**Responsibility**: Match extracted entities to canonical records; handle aliases.

**Entity Registry**:
- Central database of canonical entities
- Pre-seeded with known entities (from wikis, manually curated)
- Grows incrementally as new entities are discovered

**Resolution Process**:
1. Exact match against canonical names
2. Exact match against known aliases
3. Fuzzy match (edit distance, phonetic) for typos/variants
4. LLM-assisted resolution for ambiguous cases
5. Create new entity if no match (flag for review)

**Alias Management**:
- Aliases stored as list on canonical entity
- Track first appearance of each alias
- Handle context-dependent aliases (Strider only used before identity revealed)

**Conflict Resolution**:
- Same name, different entities (multiple Elronds? No, but multiple Finwës children named similarly)
- LLM disambiguation using surrounding context
- Human review queue for uncertain cases

---

### Store Layer

**Responsibility**: Persist graph data and enable querying.

**Primary Storage**: Neo4j
- Mature graph database with excellent query language
- Built-in visualization
- Full-text search capabilities
- ACID transactions

**Why Neo4j**:
- Cypher query language is readable and powerful
- Native graph storage (not relational with joins)
- Active community and tooling
- Free Community Edition sufficient for local use

**Alternative Considered**: 
- NetworkX (Python) — good for analysis, not persistent storage
- ArangoDB — multi-model but more complex
- Amazon Neptune — cloud-only, violates local-first

**Graph Schema**: See DATA_MODEL.md

**Indexing Strategy**:
- Index on canonical_name for all entity types
- Full-text index on passage text
- Composite index on (book, chapter, sentence) for provenance queries

---

### Analyze Layer

**Responsibility**: Extract style metrics and patterns from processed text.

**Sentence-Level Metrics**:
- Token count, word count, character count
- Clause depth (nested subordinate clauses)
- Voice (active vs. passive)
- Tense
- Part-of-speech distribution

**Vocabulary Analysis**:
- Word frequency distributions
- Hapax legomena (words appearing only once)
- Archaism detection (against curated word list)
- Invented word detection (not in dictionary)
- Latinate vs. Germanic ratio
- Average word length

**Passage-Level Patterns**:
- Scene type classification (battle, travel, council, feast, etc.)
- Dialogue density (percentage of passage that is dialogue)
- Description focus (what gets described — landscape, characters, objects?)
- Pacing (events per word, estimated "story time" per word)

**Aggregate Statistics**:
- Per-book metrics
- Per-scene-type metrics
- Comparative analysis across works
- Outlier detection (unusually long/short sentences, etc.)

**Output Format**:
- JSON documents per analysis type
- Queryable from main application
- Exportable for external analysis tools

---

### Profile Layer

**Responsibility**: Build character-specific voice profiles.

**Dialogue Extraction**:
- Identify quoted speech in text
- Attribute dialogue to speaker
- Handle complex attribution ("said Gandalf, turning to Frodo")
- Track conversation context (who else is present, topic)

**Per-Character Analysis**:
- Vocabulary fingerprint (distinctive words)
- Sentence length distribution in dialogue
- Formality metrics
- Question frequency
- Exclamation/imperative frequency
- Contraction usage
- Archaic term usage

**Profile Document Structure**:
- Character identifier (links to graph entity)
- Speech statistics (aggregate metrics)
- Distinctive patterns (what makes them sound unique)
- Sample dialogue (indexed, retrievable)
- Dialogue context distribution (who they talk to, about what)

**Validation**:
- Cross-reference with graph (character actually exists)
- Minimum dialogue threshold (need enough samples)
- Distinctiveness score (how different from other characters)

---

### Synthesize Layer

**Responsibility**: Derive world rules and thematic patterns.

**This Is Different**: Unlike extraction (pulling facts from text), synthesis involves interpretation across many passages.

**Process**:
1. Gather all passages related to a topic (e.g., "magic", "Elves", "death")
2. Feed to LLM with synthesis prompt
3. LLM proposes rules/patterns
4. Human reviews and refines
5. Store with source citations

**World Bible Categories**:
- Magic/Power systems
- Cosmology and metaphysics
- Cultural norms per people/race
- Geography and travel
- Technology and artifacts
- Recurring themes

**Rule Format**:
- Statement of the rule/pattern
- Confidence level (certain, probable, interpretive)
- Supporting passages (citations)
- Contradicting passages if any
- Notes/caveats

**Theme Detection**:
- Identify recurring motifs
- Track how themes are expressed
- Note thematic evolution across works

---

### Generate Layer

**Responsibility**: Assemble context and support writing.

**Not a Writing Engine**: This layer helps writers, it doesn't replace them.

**Context Assembly**:
Given a writing intent, gather:
- Relevant entities from knowledge graph
- Timeline/continuity constraints
- Applicable world rules
- Style examples (similar scenes from corpus)
- Character voice profiles for involved characters

**Prompt Construction**:
- Template-based prompt building
- Include style constraints (sentence length targets, vocabulary guidance)
- Include lore context (what's true about this place/time/character)
- Include character voice guidance (how this character speaks)

**Validation**:
- Check generated content against world rules
- Flag potential continuity errors
- Identify out-of-vocabulary words (invented words not in corpus)
- Style deviation detection (does this sound like the author?)

**Output**:
- Assembled context (for manual prompt use)
- Generated draft (if using built-in LLM)
- Validation report (issues found)

---

## Data Flow

### Ingestion Flow
```
Text File → Parse → Split → Passages → [Queue for Extraction]
```

### Extraction Flow  
```
Passage → Entity Extraction → Relationship Extraction → Attribute Extraction
    │
    ▼
[Extracted triples + passage metadata]
    │
    ▼
Entity Resolution → Canonical Matching
    │
    ▼
Graph Write (Neo4j)
```

### Analysis Flow
```
Passages (from DB) → Style Metrics → Aggregate → Store Profiles
                  → Dialogue Extract → Character Profiles → Store
```

### Query Flow
```
User Query → Query Parser → Cypher Generation → Neo4j → Format Results
```

---

## Local Execution Model

### Hardware Requirements
- **CPU**: Modern multi-core (parallelizable extraction)
- **RAM**: 16GB minimum (Neo4j + models in memory)
- **GPU**: NVIDIA with 8GB+ VRAM for local LLM inference
- **Storage**: SSD recommended, ~10GB for models + database

### Process Model
- Single-machine execution
- Parallel processing where safe (sentence extraction)
- Sequential for stateful operations (entity resolution)
- Background workers for long-running tasks

### Scaling Path
If local resources insufficient:
1. Reduce model size (use smaller/quantized models)
2. Batch processing (process overnight)
3. Cloud LLM API fallback (costs money but offloads compute)
4. Hugging Face Spaces for hosted inference (future)

---

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | NLP ecosystem, Neo4j drivers, LLM tooling |
| Graph DB | Neo4j Community | Mature, visualizable, free, Cypher |
| NLP | spaCy | Fast, local, customizable |
| Local LLM | Ollama + open models | Easy setup, good model variety |
| Embeddings | sentence-transformers | Local embedding generation |
| CLI | Click or Typer | Clean interface building |
| Config | YAML + Pydantic | Readable config, validated models |

---

## Security and Privacy

**Local-First Benefits**:
- Your books never leave your machine
- No cloud API required for core functionality
- No telemetry or data collection

**Cloud LLM Option**:
- Opt-in only
- Sends passage text to LLM API
- User responsible for API terms compliance

**Copyright Considerations**:
- This tool is for analysis and learning
- Don't redistribute extracted content as substitute for source texts
- Fan fiction has complex legal status; know your jurisdiction
