"""Location graph seeding MVP for Middle-earth.

Loads canonical locations from data/seeds/places.json and converts them
into LocationNode objects compatible with the spatiotemporal engine.
Also provides a minimal set of travel edges between major locations.

TODO(#48): Add real Tolkien map coordinates from canonical sources.
TODO(#48): Load custom location seeds from user-provided files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .models import LocationEdge, LocationNode

logger = logging.getLogger(__name__)

# Approximate relative coordinates for major Middle-earth locations.
# These are rough and schematic, not cartographically precise.
# Origin roughly at the center of the Misty Mountains.
_LOCATION_COORDS: dict[str, tuple[float, float]] = {
    "the_shire": (-4.0, 1.0),
    "bag_end": (-4.0, 1.0),
    "hobbiton": (-4.0, 1.0),
    "bree": (-3.0, 0.8),
    "rivendell": (-1.5, 0.5),
    "moria": (-0.5, -0.5),
    "lothlorien": (0.0, -1.0),
    "isengard": (-1.0, -2.0),
    "edoras": (-0.5, -3.0),
    "helms_deep": (-0.8, -2.8),
    "minas_tirith": (1.5, -3.0),
    "minas_morgul": (2.0, -3.2),
    "mordor": (3.0, -3.5),
    "mount_doom": (3.2, -3.5),
    "barad_dur": (3.5, -3.5),
    "dale": (1.0, 3.0),
    "erebor": (1.2, 3.2),
    "lake_town": (1.0, 2.8),
    "mirkwood": (0.5, 1.5),
    "grey_havens": (-5.5, 0.5),
    "gondolin": (-1.0, 4.0),  # Approximate First Age location
    "nargothrond": (-2.0, 3.5),
    "dol_guldur": (0.5, 0.0),
    "amon_hen": (0.5, -2.0),
    "weathertop": (-2.5, 0.6),
    "fangorn": (-0.5, -1.5),
}

# Major travel routes with approximate travel times (in days on foot).
_SEED_EDGES: list[dict] = [
    {"source_id": "the_shire", "target_id": "bree", "travel_days": 4.0, "mode": "foot"},
    {"source_id": "bree", "target_id": "weathertop", "travel_days": 3.0, "mode": "foot"},
    {"source_id": "weathertop", "target_id": "rivendell", "travel_days": 7.0, "mode": "foot"},
    {"source_id": "rivendell", "target_id": "moria", "travel_days": 14.0, "mode": "foot"},
    {"source_id": "moria", "target_id": "lothlorien", "travel_days": 3.0, "mode": "foot"},
    {"source_id": "lothlorien", "target_id": "amon_hen", "travel_days": 10.0, "mode": "boat"},
    {"source_id": "amon_hen", "target_id": "edoras", "travel_days": 5.0, "mode": "foot"},
    {"source_id": "edoras", "target_id": "helms_deep", "travel_days": 2.0, "mode": "horse"},
    {"source_id": "edoras", "target_id": "minas_tirith", "travel_days": 7.0, "mode": "horse"},
    {"source_id": "minas_tirith", "target_id": "minas_morgul", "travel_days": 2.0, "mode": "foot"},
    {"source_id": "minas_morgul", "target_id": "mount_doom", "travel_days": 5.0, "mode": "foot"},
    {"source_id": "isengard", "target_id": "edoras", "travel_days": 3.0, "mode": "horse"},
    {"source_id": "isengard", "target_id": "fangorn", "travel_days": 1.0, "mode": "foot"},
    {"source_id": "lake_town", "target_id": "erebor", "travel_days": 1.0, "mode": "foot"},
    {"source_id": "lake_town", "target_id": "dale", "travel_days": 0.5, "mode": "foot"},
    {"source_id": "mirkwood", "target_id": "lake_town", "travel_days": 10.0, "mode": "foot"},
    {"source_id": "rivendell", "target_id": "mirkwood", "travel_days": 14.0, "mode": "foot"},
    {"source_id": "the_shire", "target_id": "grey_havens", "travel_days": 4.0, "mode": "foot"},
    {"source_id": "lothlorien", "target_id": "fangorn", "travel_days": 4.0, "mode": "foot"},
    {"source_id": "dol_guldur", "target_id": "lothlorien", "travel_days": 3.0, "mode": "foot"},
]


def load_seed_locations(
    seeds_path: str | Path | None = None,
) -> dict[str, LocationNode]:
    """Load canonical locations from the seeds file and enrich with coordinates.

    Args:
        seeds_path: Path to places.json. Defaults to data/seeds/places.json.

    Returns:
        Dict mapping location id -> LocationNode.
    """
    if seeds_path is None:
        seeds_path = Path(__file__).parent.parent.parent.parent / "data" / "seeds" / "places.json"
    else:
        seeds_path = Path(seeds_path)

    if not seeds_path.exists():
        logger.warning("Location seeds file not found: %s", seeds_path)
        return {}

    with open(seeds_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    locations: dict[str, LocationNode] = {}
    for item in raw:
        loc_id = item.get("id", "")
        if not loc_id:
            continue
        coords = _LOCATION_COORDS.get(loc_id, (0.0, 0.0))
        locations[loc_id] = LocationNode(
            id=loc_id,
            name=item.get("canonical_name", loc_id),
            region=item.get("parent_region"),
            x=coords[0],
            y=coords[1],
            aliases=item.get("aliases", []),
        )

    logger.info("Loaded %d seed locations", len(locations))
    return locations


def load_seed_edges() -> list[LocationEdge]:
    """Load the built-in travel route edges."""
    return [LocationEdge(**e) for e in _SEED_EDGES]


def load_seed_location_graph(
    seeds_path: str | Path | None = None,
) -> tuple[dict[str, LocationNode], list[LocationEdge]]:
    """Convenience: load both locations and edges."""
    locations = load_seed_locations(seeds_path)
    edges = load_seed_edges()
    return locations, edges
