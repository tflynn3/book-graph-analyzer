# Getting Started

## What this project does

Book Graph Analyzer turns books into:

- knowledge graph data (characters, places, objects, events)
- dense proposition-layer data for sentence-level semantics
- style fingerprints
- character voice profiles
- world bible constraints

and uses those assets to generate new story content.

## Core flow

1. Rebuild a single-book graph with `bga pipeline full <text> --title "<Book>"`
2. Rebuild a corpus graph with `bga corpus process <corpus>`
3. Use `bga analyze <text>` only when you want zero-seed exploratory extraction
4. Build world bible context
5. Generate scenes, outline, then novel
6. Review and iterate

The main rebuild path now produces two graph layers:

- a stricter canon layer of grounded entity-to-entity edges
- a denser proposition layer for clause-level narrative scaffolding, including unresolved arguments that still matter for generation
