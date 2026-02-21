# Book Graph Analyzer - Neo4j Schema

## Core Nodes

### Book
```cypher
(:Book {
  id: string,
  title: string,
  author: string,
  
  // Style fingerprint (from Phase 4)
  total_words: int,
  total_sentences: int,
  avg_sentence_length: float,
  flesch_reading_ease: float,
  flesch_kincaid_grade: float,
  dialogue_ratio: float,
  archaism_density: float
})
```

### Character
```cypher
(:Character {
  id: string,
  canonical_name: string,
  aliases: [string],
  mention_count: int,
  
  // Voice profile (from Phase 5)
  total_lines: int,
  avg_utterance_length: float,
  question_ratio: float,
  exclamation_ratio: float,
  vocabulary_richness: float,
  distinctive_words: [string],
  sample_quotes: [string]
})
```

### Place
```cypher
(:Place {
  id: string,
  canonical_name: string,
  aliases: [string],
  mention_count: int
})
```

### Object
```cypher
(:Object {
  id: string,
  canonical_name: string,
  aliases: [string],
  mention_count: int
})
```

### Passage
```cypher
(:Passage {
  id: string,
  text: string,
  chapter: int,
  paragraph: int,
  sentence: int,
  
  // Style classification (from Phase 4)
  passage_type: string,  // dialogue, action, description, travel, battle
  word_count: int,
  has_dialogue: bool
})
```

### DialogueLine
```cypher
(:DialogueLine {
  id: string,
  text: string,
  is_question: bool,
  is_exclamation: bool,
  word_count: int
})
```

## Relationships

### Entity Relationships (Phase 2)
```cypher
(:Character)-[:SPOKE_TO]->(:Character)
(:Character)-[:TRAVELED_TO]->(:Place)
(:Character)-[:POSSESSES]->(:Object)
(:Character)-[:GAVE {object: string}]->(:Character)
(:Character)-[:FOUGHT]->(:Character)
// ... 50+ relationship types
```

### Structural Relationships
```cypher
(:Book)-[:CONTAINS]->(:Passage)
(:Passage)-[:MENTIONS]->(:Character|Place|Object)
(:Passage)-[:NEXT]->(:Passage)  // Sequential order
```

### Voice Relationships (Phase 5)
```cypher
(:Character)-[:SPEAKS]->(:DialogueLine)
(:DialogueLine)-[:IN_PASSAGE]->(:Passage)
```

## Spatiotemporal Nodes (Issue #48)

### SpatiotemporalEvent
```cypher
(:SpatiotemporalEvent {
  id: string, entity_id: string, entity_name: string,
  location_id: string, location_name: string,
  description: string, event_type: string,
  time_era: string, time_year_start: int, time_year_end: int,
  time_confidence: float, time_raw_text: string
})
```

### Location
```cypher
(:Location {id: string, name: string, region: string, x: float, y: float, aliases: [string]})
```

### Spatiotemporal Relationships
```cypher
(:Entity)-[:PARTICIPATED_IN]->(:SpatiotemporalEvent)
(:SpatiotemporalEvent)-[:LOCATED_AT]->(:Location)
(:Location)-[:TRAVEL_ROUTE {travel_days: float, mode: string, difficulty: string}]->(:Location)
```

## Example Queries

### "How does Gandalf speak?"
```cypher
MATCH (c:Character {canonical_name: 'Gandalf'})
RETURN c.avg_utterance_length, c.question_ratio, c.distinctive_words, c.sample_quotes
```

### "All dialogue between Bilbo and Gandalf"
```cypher
MATCH (b:Character {canonical_name: 'Bilbo'})-[:SPEAKS]->(d:DialogueLine)-[:IN_PASSAGE]->(p:Passage)
WHERE EXISTS((p)-[:MENTIONS]->(:Character {canonical_name: 'Gandalf'}))
RETURN d.text
```

### "Passages by type"
```cypher
MATCH (p:Passage)
RETURN p.passage_type, count(*) as count
ORDER BY count DESC
```

### "Character interaction network"
```cypher
MATCH (c1:Character)-[r]->(c2:Character)
RETURN c1.canonical_name, type(r), c2.canonical_name, count(*) as interactions
ORDER BY interactions DESC
```

### "Style comparison: find similar books"
```cypher
MATCH (b1:Book {title: 'The Hobbit'}), (b2:Book)
WHERE b1 <> b2
WITH b1, b2, 
     abs(b1.avg_sentence_length - b2.avg_sentence_length) as sent_diff,
     abs(b1.flesch_kincaid_grade - b2.flesch_kincaid_grade) as grade_diff
RETURN b2.title, sent_diff + grade_diff as style_distance
ORDER BY style_distance
```
