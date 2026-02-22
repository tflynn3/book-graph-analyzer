# Event ID book-namespace migration (Neo4j)

This change updates Event node identity from `(:Event {id})` to `(:Event {id, source_book})`.

## Why migration is needed

Older data may already contain merged/collided Event nodes where the same `id` appeared in multiple books.
Those collisions cannot be losslessly separated in-place after the fact, so the safest path is:

1. reset Event graph data,
2. replace old single-key constraint,
3. reload from source JSON.

## Safe reset + schema update (recommended)

```cypher
// 1) Remove event-only relationships first (keeps non-event graph intact)
MATCH (:Event)-[r]-(:Event) DELETE r;
MATCH (:Event)-[r]-() DELETE r;

// 2) Delete all Event nodes
MATCH (e:Event) DELETE e;

// 3) Replace old uniqueness constraint if present
DROP CONSTRAINT event_id IF EXISTS;
CREATE CONSTRAINT event_id_book IF NOT EXISTS
FOR (e:Event) REQUIRE (e.id, e.source_book) IS UNIQUE;
```

## Reload

Re-run your normal event ingestion per book (example):

```powershell
python -m book_graph_analyzer.cli lore events data/books/the_hobbit.txt --book "The Hobbit" --neo4j
python -m book_graph_analyzer.cli lore events data/books/unfinished_tales.txt --book "Unfinished Tales" --neo4j
```

Or if using pre-extracted JSON, re-run the command/path your pipeline uses to write those JSONs to Neo4j.
