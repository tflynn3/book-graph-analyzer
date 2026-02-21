# Issue #51 — Editorial Meta-Layer (Slice 1)

## Delivered in this slice

- Added source **strata model** (`core_text`, `appendix`, `gloss`, `annotation`) via `SourceStratum`.
- Extended `Passage` with provenance metadata:
  - `source_id`, `source_title`, `source_stratum`, `source_authority_weight`
  - `provenance_tags`, `factual_claims`
- Added MVP divergence detector (`worldbible.editorial.detect_editorial_divergences`) with:
  - factual contradiction signal (same claim key, different values)
  - style drift signal across strata (sentence length / passive ratio deltas)
- Added graph persistence/query support:
  - passage source/stratum properties in `GraphWriter.write_passage`
  - `write_passage_provenance()`
  - `query_layer_report()`
  - schema constraints/indexes for `:Source` and `(Passage.source_id, Passage.source_stratum)`
- CLI integration under existing `corpus sources` command:
  - `--tag-strata`
  - `--report-divergence`

## Notes

This is additive/backward-compatible: all new fields are optional and existing flows continue working without provenance data.

## Remaining scope for Issue #51

- richer rule-level contradiction clustering and weighting by authority
- direct Neo4j-backed divergence querying (not only JSON-based report)
- ingestion-stage automatic stratum extraction from source structure
- end-to-end reporting surfaces in pipeline/worldbuilding outputs
