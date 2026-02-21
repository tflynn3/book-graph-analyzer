# Issue #51 Slice 2 — Editorial Meta-Layer (Authority + Divergence)

## Delivered in this slice

- Authority-weighted contradiction clustering in `CorpusReconciler`
  - Adds `ContradictionCluster` output grouping cross-book conflicts by `(entity_id, conflict_type)`.
  - Computes average source authority from editorial source registry.
  - Emits a suggested resolution policy (`use_later_text`, `use_most_cited`, or `flag_for_human`).

- Ingestion-stage structural stratum enrichment in `ExtractionBridge`
  - `SpatiotemporalEvent` now carries:
    - `structural_stratum`
    - `editorial_status`
    - `source_authority_weight`
  - Values are inferred from known editorial layers (`infer_editorial_layer`).

- Graph-native divergence querying in Neo4j (`GraphWriter`)
  - `query_divergence_hotspots(min_sources, limit)`
  - `query_source_divergence(source_a, source_b, limit)`

- CLI enhancement
  - New command: `bga corpus timeline-divergence`
  - Supports hotspot mode and source-vs-source mode.

## Testing coverage added

- `tests/test_spatiotemporal_slice2.py`
  - structural stratum inference on bridge output.
- `tests/test_spatiotemporal_slice3.py`
  - divergence query Cypher contract tests.
- `tests/test_spatiotemporal_slice4.py`
  - contradiction cluster generation and recommendation checks.
  - CLI command smoke test for `corpus timeline-divergence`.

## Notes

- This is an MVP for authority-weighted clustering; the recommendation strategy is heuristic and intentionally conservative.
- Future slices can add claim-level canonical winner selection and richer cluster merge semantics (semantic claim similarity + provenance graph distance).
