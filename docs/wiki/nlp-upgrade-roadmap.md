# NLP Upgrade Roadmap

This roadmap turns the current proposition-layer work into a research-backed extraction stack for lore graphs and shadow-graph generation.

## Thesis

The repo should use a hybrid pipeline:

- deterministic first for clause splitting, proposition generation, temporal tagging, and closed-world candidate generation
- learned models second for the ambiguity that syntax alone cannot settle
- LLM fallback last for the small unresolved slice that remains after both passes

This is the right fit for narrative fiction. Clause structure is usually recoverable with rules, but dialogue pronouns, discourse deixis, bridging references, and long-distance coreference are not reliably recoverable from syntax alone.

Primary sources:

- [ClausIE](https://resources.mpi-inf.mpg.de/d5/clausie/clausie-www13.pdf)
- [Stanford OpenIE](https://nlp.stanford.edu/pubs/2015angeli-openie.pdf)
- [MinIE](https://aclanthology.org/D17-1278/)
- [Stanford Coreference Systems](https://stanfordnlp.github.io/CoreNLP/coref.html)
- [End-to-End Neural Coreference Resolution](https://aclanthology.org/D17-1018/)
- [BERT for Coreference Resolution](https://aclanthology.org/D19-1588/)
- [Deep Semantic Role Labeling](https://aclanthology.org/P17-1044/)
- [A Transition-Based Directed Acyclic Graph Parser for UCCA](https://aclanthology.org/S19-2001/)
- [UDepLambda](https://aclanthology.org/D17-1009/)
- [HeidelTime](https://aclanthology.org/L12-1219/)
- [Unrestricted Bridging Resolution](https://aclanthology.org/J18-2002/)

## Current Repo State

The current stack already has the right outer shape:

- sentence-level proposition extraction in `extract/propositions.py`
- book-level orchestration in `extract/book_pipeline.py`
- dense proposition persistence in `graph/writer.py`
- corpus metadata updates in `corpus/manager.py`

The main weakness is not proposition density anymore. It is reference grounding quality. The unresolved layer is still mixing:

- pronouns
- discourse-deictic mentions such as `this` and `that`
- bridging references such as body parts and generic possessions
- canon entities that should have linked cleanly

Those need different algorithms and different review queues.

## Target Architecture

| Layer | Primary job | Deterministic baseline | Learned upgrade |
| --- | --- | --- | --- |
| Segmentation | sentence, quote, clause boundaries | dependency + quote heuristics | none needed initially |
| Proposition extraction | emit dense typed propositions | ClausIE / OpenIE-style clause splitting + hand-built role typing | SRL or semantic dependency parser |
| Entity candidate generation | propose canon nodes | alias tables, epithets, type gates, chapter recency, genealogy | learned reranker |
| Reference resolution | resolve names, pronouns, bridging refs | multi-pass sieve coref + quote-local speaker rules | neural coref + bridging resolver |
| Temporal grounding | normalize dates and relative time | HeidelTime + explicit timeline rules | temporal relation model on hard cases |
| Canon projection | promote dense layer to strict graph edges | hand-built projection rules and confidence thresholds | optional learned confidence calibration |
| LLM fallback | adjudicate hard unresolved cases | not used | constrained JSON output over closed candidate lists |

## Phase Plan

### Phase 1: Unresolved Taxonomy And Instrumentation

Goal: stop treating every unresolved argument as the same problem.

Add new unresolved classes:

- `pronoun`
- `discourse_deictic`
- `bridging`
- `generic_np`
- `body_part`
- `canon_candidate`
- `unknown`

Implementation:

- extend `PropositionArgument` and `BrokenReference` with `reference_class`
- classify unresolveds inside `extract/propositions.py`
- persist the class in `graph/writer.py`
- add corpus-level summary counts by unresolved class

Files:

- `src/book_graph_analyzer/models/propositions.py`
- `src/book_graph_analyzer/models/lore_depth.py`
- `src/book_graph_analyzer/extract/propositions.py`
- `src/book_graph_analyzer/graph/writer.py`
- `src/book_graph_analyzer/cli.py`

Success metrics:

- pronouns are counted separately from canon-candidate misses
- top unresolved reports become interpretable

### Phase 2: Deterministic Proposition Hardening

Goal: make the dense layer linguistically cleaner before adding heavier models.

Algorithms to adopt:

- ClausIE-style clause inventory
- Stanford OpenIE-style clause shortening
- MinIE-style compaction for modality and polarity
- explicit treatment of copular, appositive, possessive, and prepositional predicates

Implementation:

- split the current extractor into clause detection, proposition generation, and proposition compaction stages
- represent adjunct propositions separately from core predicates
- normalize predicate lemmas and keep modality/polarity on the proposition rather than buried in surface text

Files:

- `src/book_graph_analyzer/extract/propositions.py`
- `src/book_graph_analyzer/models/propositions.py`

Success metrics:

- proposition density stays in the current `4-8` per passage range
- duplicate or vacuous propositions decline
- top predicates become less dominated by generic `be` / `have`

### Phase 3: Closed-World Entity Linking

Goal: make Tolkien linking a first-class subsystem instead of a side effect of NER overlap.

Algorithms to adopt:

- deterministic candidate generation from aliases, epithets, titles, family names, place variants, and seeded canon lists
- global coherence scoring across passage, chapter, and book context
- type constraints from genealogy, geography, and object inventories

Implementation:

- create a dedicated linker module
- separate candidate generation from candidate scoring
- use passage recency and co-occurring canon entities as features
- let unresolveds retain ranked candidates instead of only a single expected type

Files:

- `src/book_graph_analyzer/extract/dynamic_resolver.py`
- `src/book_graph_analyzer/extract/resolver.py`
- new module: `src/book_graph_analyzer/extract/linking.py`

Success metrics:

- grounded proposition arguments increase materially
- canon-candidate unresolveds drop
- cross-book entity conflicts decline

### Phase 4: Coreference And Fiction-Specific Reference Resolution

Goal: clean up the unresolved queue that is currently dominated by narrative reference phenomena.

Algorithms to adopt:

- deterministic multi-pass sieve coref for exact aliases, appositives, titles, and nearby pronouns
- quote-local first/second-person resolution using speaker attribution
- separate handling for discourse deixis and bridging references

Implementation:

- expand pronoun inventories beyond third person
- add speaker-aware `I/you/we` resolution in dialogue passages
- classify `this/that/it` mentions that target events or propositions instead of entities
- route body parts and generic possessed nouns to bridging logic instead of canon entity linking

Files:

- `src/book_graph_analyzer/extract/coref.py`
- `src/book_graph_analyzer/extract/propositions.py`
- `src/book_graph_analyzer/voice/`

Success metrics:

- unresolved pronoun volume drops sharply
- unresolved queue shifts toward genuinely hard canon disambiguation

### Phase 5: Learned Models For Ambiguous Cases

Goal: spend model capacity only where deterministic methods are weakest.

Recommended learned components:

- SRL model for difficult argument spans
- neural coref for long-distance and dialogue-heavy passages
- optional semantic dependency or UCCA/AMR-style parser for selected hard passages

Implementation:

- run these only on sentences or passages that fail deterministic confidence gates
- store which extractor produced each proposition or resolution
- compare deterministic vs learned confidence so the audit trail stays legible

Files:

- new modules under `src/book_graph_analyzer/extract/`
- possibly `src/book_graph_analyzer/story_cli.py` for shadow-graph retrieval weights

Success metrics:

- argument grounding improves without exploding false positives
- hard passages improve more than easy passages

### Phase 6: LLM Adjudication, Not LLM First-Pass Extraction

Goal: keep the system auditable.

Rules:

- only invoke an LLM after deterministic and learned passes fail
- pass a closed candidate set, not the whole legendarium
- require structured JSON output with rationale fields
- store adjudication provenance separately from canon projection

This phase is for:

- rare paraphrastic aliases
- event/proposition references that need broad context
- genuinely ambiguous dialogue pronouns

## Projection Strategy

Do not project every proposition into the canon graph.

Keep two layers:

- dense proposition layer for retrieval and shadow-graph generation
- strict canon projection for grounded, durable facts

Projection should require:

- core arguments resolved
- acceptable confidence
- proposition kind allowed by projection rules
- temporal consistency when applicable

## Recommended Immediate Work

The next two engineering tasks should be:

1. Phase 1 unresolved taxonomy
2. Phase 4 pronoun and dialogue-coref cleanup

Reason:

- they directly address the current graph pathology
- they improve auditability before new models are added
- they make later SRL and linking work measurable instead of anecdotal

## Suggested Metrics Dashboard

Track these after every corpus rebuild:

- propositions per passage
- grounded proposition arguments
- unresolved arguments by class
- top unresolved mentions by class
- canon projection rate
- cross-book entity conflict count
- per-book speaker-attributed quote coverage
- temporal normalization coverage

These metrics should be written to corpus artifacts and surfaced in CLI summaries.
