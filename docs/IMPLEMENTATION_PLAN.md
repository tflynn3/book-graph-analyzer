# Implementation Plan

A phased approach to building the Book Graph Analyzer. Each phase delivers usable value and validates assumptions before moving to the next.

---

## Phase 0: Foundation

**Goal**: Set up the development environment and validate core technical choices.

### Deliverables
- Repository structure with dependency management
- Neo4j local instance running (Docker or native)
- Basic text ingestion pipeline (load a book, split into sentences)
- Proof-of-concept: manually create a few nodes and relationships, visualize in Neo4j browser

### Validation
- Can we efficiently store and query a graph with thousands of nodes?
- Is sentence-level splitting reliable for narrative prose?

### Key Decisions
- Python as primary language (ecosystem strength for NLP)
- Neo4j as graph database (mature, great visualization, Cypher is readable)
- Local-first with Docker for reproducibility

---

## Phase 1: Entity Extraction Pipeline

**Goal**: Automatically identify characters, places, and objects in text.

### Deliverables
- Named Entity Recognition pipeline using local models
- Tolkien-specific entity seed database (scraped from fan wikis or manually curated)
- Entity extraction on a single book (The Hobbit as test case)
- Basic entity resolution (matching extracted names to canonical entities)

### Challenges to Solve
- Tolkien's invented names don't appear in standard NER training data
- Pronoun resolution ("He walked to the mountain" — who is "he"?)
- Partial references ("The wizard" → Gandalf, but which wizard in Silmarillion?)

### Approaches to Explore
- Fine-tune a small NER model on Tolkien-specific data
- Use local LLM for entity extraction as alternative/complement to NER
- Build alias mapping table (Strider → Aragorn → Elessar → Estel)

### Validation
- Precision/recall on a manually annotated chapter
- Can we correctly identify 90%+ of named entities in The Hobbit?

---

## Phase 2: Relationship Extraction

**Goal**: Understand how entities relate to each other within each sentence/passage.

### Deliverables
- Relationship extraction pipeline (who did what to whom, where)
- Relationship type taxonomy (TRAVELED_TO, SPOKE_WITH, FOUGHT, GAVE, etc.)
- Graph populated with entity-relationship-entity triples
- Source linking (every relationship links to its source passage)

### Challenges to Solve
- Implicit relationships ("Frodo and Sam continued east" — implies TRAVELED_WITH)
- Temporal relationships (what order did events happen?)
- Nested relationships ("Gandalf told Frodo that Bilbo had found the Ring")

### Approaches to Explore
- LLM-based extraction with structured output (JSON relationship triples)
- Dependency parsing to identify subject-verb-object patterns
- Hybrid: rule-based for common patterns, LLM for complex sentences

### Validation
- Sample 50 sentences, manually verify extracted relationships
- Graph should be navigable: "Show me Gandalf's relationships" returns sensible results

---

## Phase 3: The Hobbit Complete

**Goal**: Fully process The Hobbit as end-to-end proof of concept.

### Deliverables
- Complete knowledge graph for The Hobbit
- All major characters with profiles (appearances, relationships, locations)
- Timeline of events extractable from graph
- Basic query interface (CLI or notebook)

### Queries That Should Work
- "List all locations Bilbo visited in order"
- "Who did Bilbo speak with?"
- "Show me all scenes involving Gollum"
- "What objects did Bilbo acquire?"

### Validation
- Domain expert review (someone who knows The Hobbit well)
- No major characters or events missing
- Relationships make sense, minimal false positives

---

## Phase 4: Style Analysis

**Goal**: Extract quantifiable patterns in the author's writing style.

### Deliverables
- Sentence-level metrics (length, clause depth, voice)
- Vocabulary analysis (word frequency, archaism detection, invented words)
- Passage classification (scene types: battle, travel, dialogue, description)
- Author voice profile aggregate (statistical summary)

### Metrics to Capture
- Sentence length distribution (mean, variance, by scene type)
- Passive vs. active voice ratio
- Dialogue-to-narration ratio
- Vocabulary richness (unique words per N words)
- Archaism frequency (curated word list)
- Adverb usage patterns

### Approaches to Explore
- spaCy for POS tagging and dependency parsing
- Custom classifiers for scene type detection
- Comparative analysis: Hobbit vs. LOTR vs. Silmarillion

### Validation
- Metrics should reveal known differences (Hobbit is lighter/simpler than Silmarillion)
- Style profile should be distinct from other authors (control test)

---

## Phase 5: Character Voice Profiles

**Goal**: Capture how each character speaks distinctly.

### Deliverables
- Dialogue extraction and attribution (who said what)
- Per-character speech analysis (vocabulary, formality, length)
- Character voice profile documents
- Sample dialogue retrievable by character

### Challenges to Solve
- Dialogue attribution in complex scenes (multiple speakers)
- Indirect speech and reported dialogue
- Characters with limited dialogue (enough signal?)

### Metrics Per Character
- Vocabulary fingerprint (distinctive words)
- Average utterance length
- Formality score (based on word choice, contractions, etc.)
- Question vs. statement ratio
- Signature phrases (if any)

### Validation
- Blind test: given an unmarked quote, can we identify the speaker?
- Profiles should feel accurate to readers ("Yes, that's how Gandalf talks")

---

## Phase 6: Expand to Full Corpus

**Goal**: Process LOTR, Silmarillion, Unfinished Tales.

### Deliverables
- Complete knowledge graph spanning all processed texts
- Entity resolution across books (characters who appear in multiple works)
- Timeline spanning Ages (First Age through Fourth Age)
- Cross-reference capabilities ("Show me everything about Galadriel across all texts")

### Challenges to Solve
- Scale: Silmarillion has hundreds of characters in dense prose
- Conflicting information (Tolkien revised his legendarium constantly)
- Genealogies and family trees (complex relationship types)

### Approach
- Process incrementally: one book at a time, merge into existing graph
- Build genealogy-specific extraction (son of, daughter of, married)
- Track textual source for conflicts ("In Silmarillion X, but in UT Y")

### Validation
- Major characters fully connected across appearances
- Timeline is navigable by Age
- No duplicate entities (same character as two nodes)

---

## Phase 7: World Bible Extraction

**Goal**: Capture the implicit rules and patterns of the fictional world.

### Deliverables
- Magic/power rules documentation (how it works, constraints)
- Cultural profiles per race/people
- Cosmology summary (gods, creation, metaphysics)
- Thematic pattern documentation

### This Is Different
World Bible content is interpretive, not just extractive. We're synthesizing patterns across many passages, not pulling from individual sentences.

### Approach
- LLM-assisted synthesis: "Based on these 50 passages about Elven magic, what rules can we infer?"
- Human-in-the-loop validation (AI proposes rules, human confirms/refines)
- Source-linked (every rule cites supporting passages)

### Categories to Document
- Magic: How does it work? Who can use it? What are the costs?
- Technology: What exists? What's anachronistic?
- Geography: How does travel work? Distances? Climate?
- Culture: Values, customs, taboos per people
- Cosmology: Gods, afterlife, creation, metaphysics
- Themes: Recurring patterns and their expressions

### Validation
- Domain expert review
- Rules should be consistent with text (no contradictions)
- Should feel "right" to knowledgeable readers

---

## Phase 8: Generation Support

**Goal**: Use everything we've built to assist writing new content.

### Deliverables
- Query interface for writers ("What do I need to know to write a scene in Gondolin?")
- Context builder (assembles relevant lore, style examples, character voices for a prompt)
- Style-constrained generation (prompt templates that include style fingerprint)
- Validation checker (does generated content violate world rules?)

### Workflow
1. Writer describes scene intent
2. System queries knowledge graph for relevant entities, locations, timeline
3. System pulls style examples (similar scene types from corpus)
4. System retrieves character voice profiles for involved characters
5. System checks world bible for relevant rules/constraints
6. All context assembled into LLM prompt
7. LLM generates draft
8. System validates against world bible (flags potential issues)
9. Writer refines

### Validation
- Generated content should pass lore-accuracy check
- Style should be measurably similar to source author
- Writers should find the tool genuinely useful

---

## Phase 9: Polish and Extend

**Goal**: Production-ready tool with documentation and extensibility.

### Deliverables
- Clean CLI interface
- Optional web UI for graph exploration
- Documentation for adding new authors/corpora
- Performance optimization for large graphs
- Export formats (JSON, GraphML, etc.)

### Future Possibilities (Not In Scope Yet)
- Hugging Face hosting for larger-scale processing
- Multi-author comparison ("How does Tolkien's style differ from Le Guin's?")
- Community-contributed entity databases
- Integration with writing tools (Scrivener, Obsidian, etc.)

---

## Timeline Estimate

| Phase | Description | Estimated Effort |
|-------|-------------|------------------|
| 0 | Foundation | 1 week |
| 1 | Entity Extraction | 2-3 weeks |
| 2 | Relationship Extraction | 2-3 weeks |
| 3 | The Hobbit Complete | 1-2 weeks |
| 4 | Style Analysis | 2 weeks |
| 5 | Character Voice Profiles | 2 weeks |
| 6 | Full Corpus | 3-4 weeks |
| 7 | World Bible | 2-3 weeks |
| 8 | Generation Support | 3-4 weeks |
| 9 | Polish | Ongoing |

**Total estimated: 4-6 months for full system**

Phases can overlap. Value is delivered incrementally — a working Hobbit graph is useful even without style analysis.

---

## Risk Factors

### Technical Risks
- **Entity resolution at scale**: Hundreds of characters with overlapping names
- **LLM costs**: Heavy LLM usage could get expensive; need efficient prompting
- **Silmarillion density**: May require different extraction strategies than narrative prose

### Mitigation
- Start simple, add complexity as needed
- Build local LLM option early to control costs
- Silmarillion gets its own extraction tuning

### Scope Risks
- Feature creep: "What if we also..."
- Perfectionism: Waiting for 100% accuracy before moving on

### Mitigation
- Strict phase gates: ship value, then iterate
- Good enough beats perfect: 90% accuracy is useful, 100% may be impossible
