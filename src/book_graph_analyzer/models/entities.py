"""Entity models for the knowledge graph."""

from pydantic import BaseModel, Field


class EntityBase(BaseModel):
    """Base class for all entities."""

    id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None


class Character(EntityBase):
    """A character (person or sentient being)."""

    race: str | None = None
    titles: list[str] = Field(default_factory=list)
    gender: str | None = None
    birth_era: str | None = None
    death_era: str | None = None
    first_appearance_book: str | None = None
    first_appearance_chapter: str | None = None


class Place(EntityBase):
    """A location in the world."""

    type: str | None = None  # city, region, landmark, building
    parent_region: str | None = None  # ID of containing Place
    exists_in_eras: list[str] = Field(default_factory=list)


class Object(EntityBase):
    """A significant item."""

    type: str | None = None  # ring, sword, jewel, book
    creator_id: str | None = None  # Character ID
    properties: list[str] = Field(default_factory=list)


class Event(BaseModel):
    """A significant occurrence."""

    id: str
    name: str
    type: str | None = None  # battle, council, journey, death
    era: str | None = None
    year: str | None = None
    description: str | None = None
    significance: str | None = None


class Concept(BaseModel):
    """An abstract idea or lore element."""

    id: str
    name: str
    type: str | None = None  # metaphysical, cultural, historical
    description: str | None = None
    related_themes: list[str] = Field(default_factory=list)
