"""
World Bible Categories and Extraction Prompts

Defines the prompts used to extract world-building
information using LLMs.
"""

from enum import Enum


class WorldBibleCategory(Enum):
    """Categories of world-building information."""
    MAGIC = "magic"                 # Magic systems, powers, costs
    TECHNOLOGY = "technology"       # Tech level, what exists
    GEOGRAPHY = "geography"         # World map, travel, climate
    CULTURE = "culture"             # Customs, values, social structures
    COSMOLOGY = "cosmology"         # Gods, creation, metaphysics
    HISTORY = "history"             # Major events, ages, timeline
    RACES = "races"                 # Species/peoples and their traits
    POLITICS = "politics"           # Power structures, kingdoms, factions
    ECONOMY = "economy"             # Trade, currency, resources
    LANGUAGE = "language"           # Languages, naming conventions
    THEMES = "themes"               # Recurring patterns, symbolism


# Keywords used to find relevant passages for each category
CATEGORY_KEYWORDS = {
    WorldBibleCategory.MAGIC: [
        "magic", "spell", "wizard", "sorcerer", "enchant", "curse", "power",
        "ring", "staff", "wand", "conjure", "summon", "charm", "hex",
        "incantation", "magical", "sorcery", "witchcraft", "enchantment",
    ],
    WorldBibleCategory.TECHNOLOGY: [
        "forge", "craft", "build", "weapon", "armor", "tool", "machine",
        "smith", "metalwork", "invention", "device", "mechanism",
    ],
    WorldBibleCategory.GEOGRAPHY: [
        "mountain", "river", "forest", "sea", "land", "kingdom", "realm",
        "north", "south", "east", "west", "journey", "travel", "road", "path",
        "miles", "leagues", "map", "border", "territory", "climate",
    ],
    WorldBibleCategory.CULTURE: [
        "custom", "tradition", "ritual", "ceremony", "feast", "song",
        "honor", "shame", "duty", "law", "marriage", "death", "burial",
        "greeting", "gift", "hospitality", "oath", "promise",
    ],
    WorldBibleCategory.COSMOLOGY: [
        "god", "gods", "divine", "creator", "creation", "heaven", "hell",
        "afterlife", "soul", "spirit", "immortal", "eternal", "sacred",
        "prophecy", "fate", "destiny", "doom",
    ],
    WorldBibleCategory.HISTORY: [
        "age", "era", "war", "battle", "king", "queen", "throne", "crown",
        "ancient", "old", "long ago", "years", "centuries", "fell", "rose",
        "founded", "destroyed", "first", "last",
    ],
    WorldBibleCategory.RACES: [
        "elf", "elves", "dwarf", "dwarves", "hobbit", "man", "men", "human",
        "orc", "goblin", "troll", "dragon", "ent", "wizard", "people",
        "folk", "race", "kind", "kindred",
    ],
    WorldBibleCategory.POLITICS: [
        "king", "queen", "lord", "lady", "council", "throne", "crown",
        "rule", "reign", "power", "alliance", "treaty", "war", "peace",
        "kingdom", "realm", "empire", "steward",
    ],
    WorldBibleCategory.ECONOMY: [
        "gold", "silver", "coin", "trade", "merchant", "market", "wealth",
        "treasure", "rich", "poor", "farm", "mine", "craft", "guild",
    ],
    WorldBibleCategory.LANGUAGE: [
        "tongue", "language", "word", "name", "speak", "speech", "elvish",
        "dwarvish", "common", "ancient", "rune", "script", "write",
    ],
    WorldBibleCategory.THEMES: [
        "good", "evil", "light", "dark", "hope", "despair", "courage",
        "fear", "love", "hate", "friend", "enemy", "hero", "villain",
        "quest", "journey", "home", "exile", "sacrifice",
    ],
}


# LLM prompts for extracting rules from passages
CATEGORY_PROMPTS = {
    WorldBibleCategory.MAGIC: """Analyze these passages about magic/powers in this fictional world.
Extract specific rules about how magic works:
- Who can use magic? (specific individuals, races, or requirements)
- What are the costs or limitations?
- What forms does magic take?
- Are there different types of magic?

For each rule, provide:
1. A short title
2. A description
3. Which passages support this rule (quote briefly)

Format as JSON array of rules.""",

    WorldBibleCategory.TECHNOLOGY: """Analyze these passages about technology and craftsmanship.
Extract information about:
- What technology level exists?
- What can be crafted or built?
- Are there notable artifacts or inventions?
- Who are the skilled craftspeople?

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.GEOGRAPHY: """Analyze these passages about geography and places.
Extract information about:
- Major locations and their characteristics
- Distances and travel times
- Climate and terrain
- Borders and territories

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.CULTURE: """Analyze these passages about customs and culture.
Extract information about:
- Important traditions and rituals
- Social customs and etiquette
- Values and taboos
- Laws and governance

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.COSMOLOGY: """Analyze these passages about gods, creation, and metaphysics.
Extract information about:
- Divine beings and their roles
- Creation stories
- Afterlife beliefs
- Prophecies and fate

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.HISTORY: """Analyze these passages about history and past events.
Extract information about:
- Major historical events
- Ages or eras
- Rise and fall of kingdoms
- Important figures from the past

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.RACES: """Analyze these passages about different races/peoples.
Extract information about:
- Physical characteristics
- Cultural traits
- Abilities or limitations
- Relations with other races

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.POLITICS: """Analyze these passages about political structures.
Extract information about:
- Forms of government
- Power structures
- Alliances and conflicts
- Notable leaders

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.ECONOMY: """Analyze these passages about economy and trade.
Extract information about:
- Currency and wealth
- Trade routes and goods
- Resources and industries
- Economic relationships

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.LANGUAGE: """Analyze these passages about languages.
Extract information about:
- Different languages spoken
- Writing systems
- Naming conventions
- Important words or phrases

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",

    WorldBibleCategory.THEMES: """Analyze these passages for recurring themes.
Extract information about:
- Central conflicts (good vs evil, etc.)
- Symbolic elements
- Moral lessons
- Recurring motifs

For each finding, provide title, description, and supporting passages.
Format as JSON array.""",
}


# Culture extraction prompt
CULTURE_EXTRACTION_PROMPT = """Analyze these passages about the {culture_name}.
Extract a cultural profile including:

1. Core Values: What do they value most?
2. Customs: Important traditions and rituals
3. Taboos: What is forbidden or shameful?
4. Physical Traits: How do they appear?
5. Homeland: Where do they live?
6. Lifespan: How long do they live?
7. Government: How are they governed?
8. Relations: How do they relate to other peoples?
9. Language: What language do they speak?

Cite specific passages that support each point.
Format as JSON with these fields."""
