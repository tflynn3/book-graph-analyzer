"""Neo4j connection management."""

from neo4j import GraphDatabase, Driver
from neo4j.exceptions import ServiceUnavailable, AuthError

from book_graph_analyzer.config import get_settings


def get_driver() -> Driver | None:
    """Get a Neo4j driver instance."""
    settings = get_settings()

    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        return driver
    except Exception:
        return None


def check_neo4j_connection() -> bool:
    """Check if Neo4j is reachable and credentials are valid."""
    driver = get_driver()
    if not driver:
        return False

    try:
        with driver.session() as session:
            session.run("RETURN 1")
        return True
    except (ServiceUnavailable, AuthError):
        return False
    finally:
        driver.close()


def init_schema() -> None:
    """Initialize graph schema (indexes and constraints)."""
    driver = get_driver()
    if not driver:
        raise ConnectionError("Cannot connect to Neo4j")

    constraints = [
        # Unique constraints
        "CREATE CONSTRAINT char_id IF NOT EXISTS FOR (c:Character) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT place_id IF NOT EXISTS FOR (p:Place) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT object_id IF NOT EXISTS FOR (o:Object) REQUIRE o.id IS UNIQUE",
        "CREATE CONSTRAINT event_id IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE",
        "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
    ]

    indexes = [
        # Name indexes for lookup
        "CREATE INDEX char_name IF NOT EXISTS FOR (c:Character) ON (c.canonical_name)",
        "CREATE INDEX place_name IF NOT EXISTS FOR (p:Place) ON (p.canonical_name)",
        "CREATE INDEX object_name IF NOT EXISTS FOR (o:Object) ON (o.canonical_name)",
        # Passage location index
        "CREATE INDEX passage_loc IF NOT EXISTS FOR (p:Passage) ON (p.book, p.chapter_num, p.sentence_num)",
        # Full-text search on passages
        # Note: Full-text indexes have different syntax
    ]

    with driver.session() as session:
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception:
                pass  # Constraint may already exist

        for index in indexes:
            try:
                session.run(index)
            except Exception:
                pass  # Index may already exist

    driver.close()
