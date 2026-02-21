# Issue #49 Slice 2 — Genealogy Closeout Enhancements

This slice closes the highest-value gaps left in slice 1 for the genealogy pipeline.

## Delivered

- Improved entity resolution for extraction:
  - Name normalization (honorific/appositive trimming)
  - Passage-local alias resolution for shortened name mentions
- House/clan inference from local textual context:
  - Recognizes `House of X`, `of the House X`, and `clan X` patterns
  - Applies inferred house metadata to extracted relations when explicit `--house` is not provided
- Generational-depth inference via traversal:
  - Direct edge defaults (`PARENT_OF`/`CHILD_OF` => 1, sibling/spouse => 0)
  - Ancestor/descendant depth inferred from shortest parent-chain path when available
- Traversal-based tree helpers:
  - `build_ancestor_chain()` and `build_descendant_tree()` now walk multiple generations (BFS)
- Schema/index improvements for genealogy queries:
  - Character `canonical_id` uniqueness + lookup index
  - Relationship indexes on `GENEALOGY.house`, `GENEALOGY.relation_type`, `GENEALOGY.generation_depth`

## Test coverage added

- House inference from context
- Generation depth inference for direct and traversal-derived relations
- Multi-generation ancestor traversal behavior

## Remaining follow-up (non-blocking)

- Plug extraction-time resolver into global cross-book disambiguation dictionaries
- Add conflict validation for contradictory parentage lines
- Add optional query mode for path-derived depth in Neo4j (not just stored edge property)
