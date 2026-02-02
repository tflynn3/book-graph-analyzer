# Vision & Goals

## The Problem

Writing fan fiction (or original work inspired by an author) is hard because:

1. **Lore is vast and scattered** — Who was alive when? What happened where? Keeping track of thousands of characters across millennia is brutal.

2. **Voice is subtle** — You can *feel* when something doesn't sound like Tolkien, but articulating why is difficult. Is it word choice? Sentence length? Pacing?

3. **World rules are implicit** — Tolkien never wrote "magic in Middle-earth works like X." You absorb the rules through exposure, but can you articulate them?

4. **Research is slow** — Want to write a scene in Gondolin? Hope you remember which book, which chapter, which paragraph described it.

## The Solution

A system that reads an author's complete works and extracts structured, queryable knowledge:

### Knowledge Graph
Every entity (character, place, object, event) as a node. Every relationship as an edge. Every mention linked back to source text.

*"Show me everyone Fëanor interacted with, and when."*
*"What locations existed in both the First and Third Ages?"*
*"Trace the Silmarils through every owner."*

### Style Fingerprint
Quantified patterns in how the author writes:
- Sentence structure distributions
- Vocabulary fingerprints (archaic terms, invented words, word-length patterns)
- Pacing metrics (words per scene type, dialogue density)
- Descriptive tendencies (what does the author notice about landscapes? faces? battles?)

*"Tolkien averages 23 words per sentence in battle scenes but 31 in landscape descriptions."*
*"He uses passive voice 40% more in Silmarillion than in Hobbit."*

### Character Voice Profiles
Per-character speech patterns:
- Formality level and variance
- Signature phrases and verbal tics
- Vocabulary constraints (Sam doesn't use the same words as Gandalf)
- Dialogue length distributions

*"Generate dialogue for Gandalf speaking to a hobbit vs. speaking to a king."*

### World Bible
The extracted rules of the fictional world:
- How does magic/technology work?
- What are the cultural norms of each people/race?
- What's cosmologically possible vs. impossible?
- What themes recur and how are they expressed?

*"In Middle-earth, magic corrupts when used for domination. Power is always a burden."*

## What Success Looks Like

### For Analysis
- Query: "Show me Galadriel's complete timeline across all texts"
- Result: Chronological list of every appearance, with links to source passages

### For Writing
- Prompt: "Write a scene where a young Númenórean sailor first glimpses Tol Eressëa"
- Result: Prose that feels like it could be from Unfinished Tales—correct lore, appropriate voice, no rule violations

### For Understanding
- Question: "How does Tolkien's prose style differ between The Hobbit and The Silmarillion?"
- Answer: Quantified comparisons with examples

## Design Principles

### Local-First
This runs on your machine. No cloud dependency for core functionality. Your books, your database, your analysis.

### Corpus-Agnostic
Built for Tolkien first, but designed to work with any author. Swap the texts, retrain the entity recognizer, and analyze Ursula K. Le Guin or Frank Herbert.

### Source-Linked
Every extracted fact links back to the original passage. The graph is an index into the texts, not a replacement for them.

### Incremental
Process one book at a time. Add new texts to an existing graph. Refine entity resolution as you go.

### Composable
Use just the knowledge graph. Or just the style analysis. Or combine everything for generation. Components work independently.

## Non-Goals

- **Not a writing AI**: This is infrastructure for writers, not a replacement for writing. It provides context and constraints; you (or an LLM you direct) do the actual writing.

- **Not a copyright tool**: Don't use this to plagiarize. It's for understanding craft, not copying content.

- **Not real-time**: Processing a corpus takes time. This is for deep analysis, not quick lookups.
