# Data Model

The graph schema and document structures used by Book Graph Analyzer.

---

## Graph Schema (Neo4j)

### Node Types

#### Character
Represents a person or sentient being in the narrative.

```
(:Character {
  id: string,              // Unique identifier (e.g., "char_gandalf")
  canonical_name: string,  // Primary name (e.g., "Gandalf")
  aliases: [string],       // Alternative names ["Mithrandir", "Olórin", "Tharkûn"]
  race: string,            // "Maia", "Hobbit", "Elf", "Man", "Dwarf", etc.
  titles: [string],        // ["The Grey", "The White"]
  gender: string,          // If known/applicable
  birth_era: string,       // "Before Time", "First Age", etc.
  death_era: string,       // If applicable
  description: string,     // Aggregated physical/personality description
  first_appearance: {      // Where we first meet them
    book: string,
    chapter: string
  }
})
```

#### Place
A location in the world.

```
(:Place {
  id: string,              // "place_rivendell"
  canonical_name: string,  // "Rivendell"
  aliases: [string],       // ["Imladris", "The Last Homely House"]
  type: string,            // "city", "region", "landmark", "building"
  parent_region: string,   // Reference to containing Place if applicable
  description: string,     // Aggregated description
  exists_in_eras: [string] // ["Second Age", "Third Age"]
})
```

#### Object
A significant item (weapons, artifacts, jewelry, etc.).

```
(:Object {
  id: string,              // "obj_one_ring"
  canonical_name: string,  // "The One Ring"
  aliases: [string],       // ["Isildur's Bane", "The Ruling Ring", "Precious"]
  type: string,            // "ring", "sword", "jewel", "book"
  creator: string,         // Reference to Character id if known
  description: string,
  properties: [string]     // Special attributes ["invisibility", "corruption"]
})
```

#### Event
A significant occurrence in the narrative.

```
(:Event {
  id: string,              // "event_council_of_elrond"
  name: string,            // "Council of Elrond"
  type: string,            // "council", "battle", "journey", "death"
  era: string,             // "Third Age"
  year: string,            // "3018" (if known, relative dating okay)
  description: string,
  significance: string     // Why this event matters
})
```

#### Passage
A unit of source text (typically a sentence).

```
(:Passage {
  id: string,              // "passage_fellowship_ch2_p5_s3"
  text: string,            // The actual sentence/passage
  book: string,            // "The Fellowship of the Ring"
  chapter: string,         // "The Shadow of the Past"
  chapter_num: integer,    // 2
  paragraph_num: integer,  // 5
  sentence_num: integer,   // 3
  char_offset: integer,    // Character position in source file
  sentiment: string,       // "hopeful", "tense", "melancholy"
  scene_type: string       // "dialogue", "action", "description"
})
```

#### Concept
Abstract ideas, peoples, or lore elements.

```
(:Concept {
  id: string,              // "concept_gift_of_men"
  name: string,            // "The Gift of Men"
  type: string,            // "metaphysical", "cultural", "historical"
  description: string,
  related_themes: [string] // ["mortality", "fate"]
})
```

---

### Relationship Types

#### Character Relationships

```
(:Character)-[:INTERACTED_WITH {
  type: string,            // "spoke_with", "fought", "traveled_with"
  sentiment: string,       // "friendly", "hostile", "neutral"
  passage_id: string       // Link to source passage
}]->(:Character)

(:Character)-[:RELATED_TO {
  relation: string,        // "parent_of", "child_of", "sibling_of", "spouse_of"
  passage_id: string
}]->(:Character)

(:Character)-[:MEMBER_OF {
  role: string,            // "king", "member", "founder"
  passage_id: string
}]->(:Concept)  // e.g., membership in a people or organization
```

#### Location Relationships

```
(:Character)-[:LOCATED_AT {
  context: string,         // "lives", "visited", "born"
  era: string,
  passage_id: string
}]->(:Place)

(:Character)-[:TRAVELED_TO {
  from: string,            // Place id
  passage_id: string
}]->(:Place)

(:Place)-[:PART_OF]->(:Place)  // Geographic containment

(:Event)-[:TOOK_PLACE_AT {
  passage_id: string
}]->(:Place)
```

#### Object Relationships

```
(:Character)-[:POSSESSES {
  acquired_how: string,    // "found", "forged", "inherited", "stole"
  passage_id: string
}]->(:Object)

(:Character)-[:CREATED {
  passage_id: string
}]->(:Object)

(:Object)-[:INVOLVED_IN {
  role: string,            // "weapon_used", "prize", "destroyed"
  passage_id: string
}]->(:Event)
```

#### Textual Relationships

```
(:Character)-[:MENTIONED_IN {
  name_used: string,       // Which alias was used in this mention
  role: string             // "subject", "speaker", "referenced"
}]->(:Passage)

(:Place)-[:MENTIONED_IN]->(:Passage)
(:Object)-[:MENTIONED_IN]->(:Passage)
(:Event)-[:MENTIONED_IN]->(:Passage)

(:Passage)-[:FOLLOWS]->(:Passage)  // Sequential ordering
(:Passage)-[:IN_CHAPTER]->(:Chapter)  // If we model chapters as nodes
```

---

## Document Schemas (JSON)

For non-graph data stored as JSON documents.

### Author Style Profile

```json
{
  "author": "J.R.R. Tolkien",
  "corpus": ["The Hobbit", "LOTR", "Silmarillion"],
  "sentence_metrics": {
    "mean_length": 24.3,
    "median_length": 21,
    "std_dev": 12.7,
    "by_scene_type": {
      "battle": { "mean": 19.2 },
      "landscape": { "mean": 31.5 },
      "dialogue": { "mean": 16.8 }
    }
  },
  "vocabulary": {
    "unique_words": 14523,
    "hapax_count": 3201,
    "archaism_frequency": 0.023,
    "invented_words": ["mithril", "mathom", "smial", "..."],
    "top_100_words": ["the", "and", "..."]
  },
  "voice": {
    "passive_ratio": 0.18,
    "adverb_frequency": 0.034,
    "dialogue_ratio": 0.31
  },
  "patterns": {
    "chapter_opening_style": "Often begins with temporal or geographical setting",
    "chapter_closing_style": "Frequently ends on moments of transition or rest",
    "song_usage": "Songs appear in moments of history, mourning, or celebration"
  }
}
```

### Character Voice Profile

```json
{
  "character_id": "char_gandalf",
  "canonical_name": "Gandalf",
  "dialogue_count": 847,
  "speech_metrics": {
    "mean_utterance_length": 18.4,
    "vocabulary_size": 1203,
    "formality_score": 0.72,
    "question_ratio": 0.15,
    "exclamation_ratio": 0.08,
    "contraction_usage": 0.03,
    "archaism_frequency": 0.041
  },
  "distinctive_words": [
    "fool", "indeed", "certainly", "precisely"
  ],
  "signature_phrases": [
    "Fool of a Took",
    "I am a servant of the Secret Fire"
  ],
  "formality_variance": {
    "to_hobbits": 0.58,
    "to_elves": 0.81,
    "to_enemies": 0.89
  },
  "sample_dialogue_ids": [
    "passage_fellowship_ch1_p23_s2",
    "passage_fellowship_ch2_p8_s5"
  ]
}
```

### World Bible Entry

```json
{
  "category": "magic",
  "topic": "The nature of magic in Middle-earth",
  "rules": [
    {
      "statement": "Magic is not flashy or explicit; it is woven into craft, song, and will",
      "confidence": "high",
      "supporting_passages": ["passage_xxx", "passage_yyy"],
      "notes": "Tolkien explicitly disliked 'magician' connotations"
    },
    {
      "statement": "Magic used for domination or control corrupts the user",
      "confidence": "high",
      "supporting_passages": ["passage_zzz"],
      "examples": ["The One Ring", "Saruman's voice", "Morgoth's song"]
    }
  ],
  "related_concepts": ["concept_ring_lore", "concept_istari"],
  "themes": ["power_corrupts", "humility"]
}
```

### Extraction Result (Intermediate)

```json
{
  "passage_id": "passage_fellowship_ch1_p5_s2",
  "text": "Gandalf came to the Shire for the last time on a bright morning in April.",
  "entities_extracted": [
    { "text": "Gandalf", "type": "CHARACTER", "confidence": 0.99 },
    { "text": "the Shire", "type": "PLACE", "confidence": 0.98 },
    { "text": "April", "type": "TIME", "confidence": 0.95 }
  ],
  "entities_resolved": [
    { "text": "Gandalf", "canonical_id": "char_gandalf" },
    { "text": "the Shire", "canonical_id": "place_shire" }
  ],
  "relationships_extracted": [
    {
      "from": "char_gandalf",
      "to": "place_shire",
      "type": "TRAVELED_TO",
      "context": "arrived",
      "confidence": 0.92
    }
  ],
  "attributes": {
    "sentiment": "nostalgic",
    "scene_type": "travel",
    "temporal_marker": "April, unspecified year"
  }
}
```

---

## Entity Seed Format

For pre-populating known entities.

```json
{
  "characters": [
    {
      "canonical_name": "Aragorn",
      "aliases": ["Strider", "Elessar", "Estel", "Thorongil", "The Dúnadan"],
      "race": "Man (Dúnedain)",
      "titles": ["King of Gondor and Arnor", "Chieftain of the Dúnedain"],
      "source": "tolkiengateway.net"
    }
  ],
  "places": [
    {
      "canonical_name": "Rivendell",
      "aliases": ["Imladris", "The Last Homely House East of the Sea"],
      "type": "settlement",
      "parent_region": "Eriador",
      "source": "tolkiengateway.net"
    }
  ],
  "objects": [
    {
      "canonical_name": "The One Ring",
      "aliases": ["Isildur's Bane", "The Ruling Ring", "The Ring of Power"],
      "type": "ring",
      "source": "tolkiengateway.net"
    }
  ]
}
```

---

## Query Examples (Cypher)

### Find all of Gandalf's interactions
```cypher
MATCH (g:Character {canonical_name: "Gandalf"})-[r:INTERACTED_WITH]->(other:Character)
RETURN other.canonical_name, r.type, count(*) as interactions
ORDER BY interactions DESC
```

### Trace the One Ring through all possessors
```cypher
MATCH (c:Character)-[r:POSSESSES]->(ring:Object {canonical_name: "The One Ring"})
MATCH (p:Passage) WHERE p.id = r.passage_id
RETURN c.canonical_name, r.acquired_how, p.book, p.chapter
ORDER BY p.char_offset
```

### Find all passages mentioning a location
```cypher
MATCH (place:Place {canonical_name: "Mordor"})<-[:MENTIONED_IN]-(p:Passage)
RETURN p.text, p.book, p.chapter
ORDER BY p.book, p.chapter_num, p.sentence_num
```

### Character's journey through locations
```cypher
MATCH (frodo:Character {canonical_name: "Frodo Baggins"})-[:TRAVELED_TO|LOCATED_AT]->(place:Place)
MATCH (p:Passage) WHERE p.id IN [rel.passage_id]
RETURN place.canonical_name, p.book, p.chapter
ORDER BY p.char_offset
```

### Events in the Third Age
```cypher
MATCH (e:Event {era: "Third Age"})
OPTIONAL MATCH (e)-[:TOOK_PLACE_AT]->(place:Place)
RETURN e.name, e.year, place.canonical_name
ORDER BY e.year
```

---

## Indexing Strategy

### Required Indexes
```cypher
CREATE INDEX char_name FOR (c:Character) ON (c.canonical_name);
CREATE INDEX place_name FOR (p:Place) ON (p.canonical_name);
CREATE INDEX obj_name FOR (o:Object) ON (o.canonical_name);
CREATE INDEX passage_loc FOR (p:Passage) ON (p.book, p.chapter_num, p.sentence_num);
CREATE INDEX event_era FOR (e:Event) ON (e.era);
```

### Full-Text Search
```cypher
CREATE FULLTEXT INDEX passage_text FOR (p:Passage) ON EACH [p.text];
```

Enables queries like:
```cypher
CALL db.index.fulltext.queryNodes("passage_text", "ring AND shadow") 
YIELD node, score
RETURN node.text, node.book, score
ORDER BY score DESC
LIMIT 10
```

---

## Schema Evolution

The schema will evolve as we discover new requirements. Principles for evolution:

1. **Additive changes preferred**: Add new properties/relationships rather than changing existing ones
2. **Version tracking**: Document schema version in database metadata
3. **Migration scripts**: Provide upgrade paths for existing databases
4. **Backward compatibility**: Old queries should continue to work where possible
