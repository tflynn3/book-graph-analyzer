"""Conflict tracking for Tolkien lore contradictions.

Contains:
  KNOWN_TOLKIEN_CONFLICTS — pre-seeded baseline of major contradictions
  ConflictRegistry        — in-memory + Neo4j registry of LoreConflict objects
  ConflictDetector        — detects new conflicts when extracting facts
  ConflictAwareValidator  — wraps LoreRuleValidator; suppresses known-conflict false positives
  LoreConflictNeo4jWriter — writes (:LoreConflict) nodes and [:CONFLICTS_WITH] edges
"""

from __future__ import annotations

from typing import Optional

from ..models.lore_conflict import (
    LoreConflict,
    ConflictClaim,
    ConflictType,
    ResolutionPolicy,
    AuthorPeriod,
)
from ..models.lore_rule import LoreValidationResult, LoreViolation
from ..models.worldbuilding import infer_editorial_layer


# ---------------------------------------------------------------------------
# Pre-seeded baseline: known major Tolkien contradictions
# ---------------------------------------------------------------------------

KNOWN_TOLKIEN_CONFLICTS: list[LoreConflict] = [

    # ---- Blue Wizards (Alatar/Pallando vs. Morinehtar/Rómestámo) -----------

    LoreConflict(
        id="blue_wizards_names",
        summary=(
            "The Blue Wizards are named 'Alatar and Pallando' in Unfinished Tales "
            "but renamed 'Morinehtar and Rómestámo' in late notes (Peoples of Middle-earth)."
        ),
        conflict_type=ConflictType.RETCON,
        entity_ids=["blue_wizards", "alatar", "pallando", "morinehtar", "romestamo"],
        rule_ids=[],
        claims=[
            ConflictClaim(
                statement="The Blue Wizards are named Alatar and Pallando.",
                source_book="Unfinished Tales",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.95,
            ),
            ConflictClaim(
                statement="The Blue Wizards are named Morinehtar and Rómestámo.",
                source_book="The Peoples of Middle-earth",
                author_period=AuthorPeriod.LATE,
                confidence=0.90,
            ),
        ],
        resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
        resolution_notes=(
            "Both names are accepted in different scholarly traditions. "
            "The late names reflect Tolkien's final intent; UT names are more familiar. "
            "Both can coexist as names in different Elvish traditions."
        ),
        resolved=True,
    ),

    # ---- Glorfindel identity (First Age / Third Age same elf?) --------------

    LoreConflict(
        id="glorfindel_identity",
        summary=(
            "Is Glorfindel of Rivendell (LotR) the same Elf who died killing a Balrog "
            "in the First Age? Tolkien changed his mind multiple times."
        ),
        conflict_type=ConflictType.RETCON,
        entity_ids=["glorfindel"],
        rule_ids=["race_elf_immortal"],  # Glorfindel died but returned — exception to immortality rule
        claims=[
            ConflictClaim(
                statement=(
                    "Glorfindel of Rivendell is a different Elf from the Glorfindel "
                    "who died in the Fall of Gondolin. The name is a coincidence."
                ),
                source_book="The Lord of the Rings (early conception)",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.6,
            ),
            ConflictClaim(
                statement=(
                    "Glorfindel of Rivendell IS the Glorfindel of Gondolin, reincarnated "
                    "and returned from Valinor by the Valar as a reward for his sacrifice."
                ),
                source_book="Unfinished Tales / late notes",
                author_period=AuthorPeriod.LATE,
                confidence=0.9,
            ),
        ],
        resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        resolution_notes=(
            "Tolkien's late writing (circa 1972) firmly identifies them as the same Elf. "
            "This is now the accepted canonical reading. The earlier ambiguity is moot."
        ),
        resolved=True,
    ),

    # ---- Tom Bombadil's nature -----------------------------------------------

    LoreConflict(
        id="bombadil_nature",
        summary=(
            "Tom Bombadil's nature and origin are deliberately left unexplained in LotR. "
            "Tolkien gave contradictory hints in letters: a 'spirit of the (vanishing) Oxford "
            "and Berkshire countryside', or the oldest living thing, unaffected by the Ring."
        ),
        conflict_type=ConflictType.AMBIGUITY,
        entity_ids=["tom_bombadil"],
        rule_ids=["magic_ring_corruption"],  # Bombadil is immune to Ring corruption
        claims=[
            ConflictClaim(
                statement="Tom Bombadil is a nature spirit — the spirit of the Oxford countryside personified.",
                source_book="Letters of J.R.R. Tolkien",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.7,
            ),
            ConflictClaim(
                statement="Tom Bombadil is 'the oldest' and represents a deliberately enigmatic mystery meant to remain unsolved.",
                source_book="Letters of J.R.R. Tolkien",
                author_period=AuthorPeriod.LATE,
                confidence=0.8,
            ),
            ConflictClaim(
                statement="Tom Bombadil may be a Maia of Yavanna (scholarly interpretation).",
                source_book="Various scholarly analyses",
                author_period=AuthorPeriod.LATE,
                confidence=0.4,
            ),
        ],
        resolution_policy=ResolutionPolicy.IRRESOLVABLE,
        resolution_notes=(
            "Tolkien explicitly stated Bombadil was meant to remain enigmatic. "
            "The Ring's inability to corrupt him is consistent regardless of his nature. "
            "Mark as irresolvable — any depiction of Bombadil should preserve the mystery."
        ),
        resolved=False,
    ),

    # ---- Elvish mortality (early vs. late Tolkien) ---------------------------

    LoreConflict(
        id="elvish_mortality",
        summary=(
            "Early Tolkien had Elves who could 'die of grief' or 'fade' into nothing. "
            "Later Tolkien refined this: Elves are bound to Arda until its end; "
            "death means their spirit goes to Mandos, not annihilation."
        ),
        conflict_type=ConflictType.RETCON,
        entity_ids=[],  # Applies to all Elves
        rule_ids=["race_elf_immortal"],
        claims=[
            ConflictClaim(
                statement="Elves can 'die' of grief, fading from the world into nothing — a true death.",
                source_book="The Book of Lost Tales",
                author_period=AuthorPeriod.EARLY,
                confidence=0.7,
            ),
            ConflictClaim(
                statement=(
                    "Elves cannot truly die; their fëa goes to Mandos. 'Death' for Elves means "
                    "temporary separation of body and spirit — they can be reincarnated."
                ),
                source_book="Laws and Customs of the Eldar / Morgoth's Ring",
                author_period=AuthorPeriod.LATE,
                confidence=0.95,
            ),
        ],
        resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        resolution_notes=(
            "The later theology is explicit and detailed (Laws and Customs). "
            "The LoreRule 'race_elf_immortal' reflects the late canonical view. "
            "Early BoLT material should be treated as superseded."
        ),
        resolved=True,
    ),

    # ---- Balrogs: how many? wings? -------------------------------------------

    LoreConflict(
        id="balrog_count_and_wings",
        summary=(
            "Early Tolkien had Balrogs in the hundreds. Late Tolkien reduced them to 'at most seven'. "
            "Additionally, whether Balrogs have physical wings (can fly) is debated from the text."
        ),
        conflict_type=ConflictType.RETCON,
        entity_ids=["balrog"],
        rule_ids=[],
        claims=[
            ConflictClaim(
                statement="Balrogs existed in large numbers — possibly hundreds — in the early legendarium.",
                source_book="The Book of Lost Tales",
                author_period=AuthorPeriod.EARLY,
                confidence=0.8,
            ),
            ConflictClaim(
                statement="There were at most seven Balrogs remaining in Middle-earth in the later Ages.",
                source_book="Letters of J.R.R. Tolkien / late notes",
                author_period=AuthorPeriod.LATE,
                confidence=0.85,
            ),
            ConflictClaim(
                statement="The Balrog's 'wings' in Moria are metaphorical shadow-wings, not physical.",
                source_book="The Lord of the Rings (textual analysis)",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.6,
            ),
        ],
        resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        resolution_notes=(
            "The 'at most seven' figure is Tolkien's considered late position. "
            "The wing question remains genuinely ambiguous — flag depictions either way."
        ),
        resolved=True,
    ),

    # ---- Orcs: origin (Elves vs. Men) ---------------------------------------

    LoreConflict(
        id="orc_origin",
        summary=(
            "Tolkien was troubled by Orc origin throughout his life. "
            "Early: Orcs = corrupted Elves. Later: also possibly Men, or 'soulless' automatons. "
            "The corrupted-Elves theory creates theological problems Tolkien never resolved."
        ),
        conflict_type=ConflictType.RETCON,
        entity_ids=["orc"],
        rule_ids=[],
        claims=[
            ConflictClaim(
                statement="Orcs were created by Morgoth from captured Elves, corrupted and twisted.",
                source_book="The Silmarillion",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.9,
            ),
            ConflictClaim(
                statement="Orcs may be corrupted Men, not Elves — Tolkien considered this in late writings.",
                source_book="Morgoth's Ring / late notes",
                author_period=AuthorPeriod.LATE,
                confidence=0.7,
            ),
            ConflictClaim(
                statement="Orcs may be 'soulless' automata operated by Morgoth's will, with no independent fëa.",
                source_book="Morgoth's Ring / late notes",
                author_period=AuthorPeriod.LATE,
                confidence=0.6,
            ),
        ],
        resolution_policy=ResolutionPolicy.FLAG_FOR_HUMAN,
        resolution_notes=(
            "Tolkien never resolved this. The Silmarillion version is most widely known "
            "and accepted. Generation should default to corrupted-Elves unless specifically "
            "targeting late-period themes."
        ),
        resolved=False,
    ),

    # ---- Finwë's second marriage (Elvish remarriage) ------------------------

    LoreConflict(
        id="elvish_remarriage",
        summary=(
            "Whether Elves can remarry after the death of a spouse was debated by Tolkien. "
            "Finwë's marriage to Indis after Míriel's death creates theological tension."
        ),
        conflict_type=ConflictType.AMBIGUITY,
        entity_ids=["finwe", "miriel", "indis"],
        rule_ids=[],
        claims=[
            ConflictClaim(
                statement="Elves cannot remarry — each fëa is bound to one spouse for eternity.",
                source_book="Laws and Customs of the Eldar (early draft)",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.7,
            ),
            ConflictClaim(
                statement=(
                    "Finwë was granted special dispensation to remarry because Míriel's "
                    "self-willed death was unprecedented. Remarriage is normally forbidden."
                ),
                source_book="Morgoth's Ring (Laws and Customs of the Eldar, final)",
                author_period=AuthorPeriod.LATE,
                confidence=0.85,
            ),
        ],
        resolution_policy=ResolutionPolicy.USE_LATER_TEXT,
        resolution_notes=(
            "The final Laws and Customs text is the most developed. "
            "Finwë's case is an exception, not the rule."
        ),
        resolved=True,
    ),

    # ---- Númenórean lifespan -------------------------------------------------

    LoreConflict(
        id="numenorean_lifespan",
        summary=(
            "The exact lifespan of Númenóreans varies across texts. "
            "Early texts suggest 200-300 years; later texts imply multiple centuries "
            "for kings, with the lifespan decreasing as they approached the Downfall."
        ),
        conflict_type=ConflictType.AMBIGUITY,
        entity_ids=["numenorean"],
        rule_ids=[],
        claims=[
            ConflictClaim(
                statement="Númenóreans lived approximately three times the lifespan of normal Men.",
                source_book="Appendices to The Lord of the Rings",
                author_period=AuthorPeriod.MIDDLE,
                confidence=0.8,
            ),
            ConflictClaim(
                statement="The greatest Númenórean kings lived 400-500 years; the lifespan decreased near the Downfall.",
                source_book="Unfinished Tales / late texts",
                author_period=AuthorPeriod.LATE,
                confidence=0.85,
            ),
        ],
        resolution_policy=ResolutionPolicy.BOTH_VALID_IN_UNIVERSE,
        resolution_notes="Both figures are consistent — the later number is more specific.",
        resolved=True,
    ),
]


# ---------------------------------------------------------------------------
# ConflictRegistry
# ---------------------------------------------------------------------------

class ConflictRegistry:
    """In-memory registry of LoreConflict objects.

    Can be seeded from KNOWN_TOLKIEN_CONFLICTS, loaded from Neo4j, or built manually.
    """

    def __init__(self) -> None:
        self._conflicts: dict[str, LoreConflict] = {}

    def add(self, conflict: LoreConflict) -> None:
        self._conflicts[conflict.id] = conflict

    def add_many(self, conflicts: list[LoreConflict]) -> None:
        for c in conflicts:
            self.add(c)

    def get(self, conflict_id: str) -> Optional[LoreConflict]:
        return self._conflicts.get(conflict_id)

    def all(self) -> list[LoreConflict]:
        return list(self._conflicts.values())

    def resolved(self) -> list[LoreConflict]:
        return [c for c in self._conflicts.values() if c.is_resolved]

    def unresolved(self) -> list[LoreConflict]:
        return [c for c in self._conflicts.values() if not c.is_resolved]

    def needing_human_review(self) -> list[LoreConflict]:
        return [c for c in self._conflicts.values() if c.needs_human_review]

    def by_entity(self, entity_id: str) -> list[LoreConflict]:
        return [c for c in self._conflicts.values() if entity_id in c.entity_ids]

    def by_rule(self, rule_id: str) -> list[LoreConflict]:
        return [c for c in self._conflicts.values() if rule_id in c.rule_ids]

    def by_type(self, conflict_type: str) -> list[LoreConflict]:
        return [c for c in self._conflicts.values() if c.conflict_type == conflict_type]

    def resolve(
        self,
        conflict_id: str,
        policy: str,
        notes: str = "",
    ) -> bool:
        """Apply a resolution policy to an existing conflict. Returns True if found."""
        conflict = self._conflicts.get(conflict_id)
        if not conflict:
            return False
        conflict.resolution_policy = policy
        conflict.resolution_notes = notes or conflict.resolution_notes
        conflict.resolved = policy not in (
            ResolutionPolicy.FLAG_FOR_HUMAN,
            ResolutionPolicy.IRRESOLVABLE,
        )
        return True

    @classmethod
    def from_tolkien_defaults(cls) -> "ConflictRegistry":
        """Create a registry pre-loaded with all known Tolkien conflicts."""
        registry = cls()
        registry.add_many(KNOWN_TOLKIEN_CONFLICTS)
        return registry

    def __len__(self) -> int:
        return len(self._conflicts)

    def suppresses_violation(self, rule_id: str, entity_id: str) -> bool:
        """Check if any conflict suppresses a lore violation for rule + entity."""
        return any(
            c.suppresses_lore_violation(rule_id, entity_id)
            for c in self._conflicts.values()
        )

    def downgrades_to_soft(self, rule_id: str, entity_id: str) -> bool:
        """Check if any conflict downgrades a HARD violation to SOFT."""
        return any(
            c.downgrades_to_soft(rule_id, entity_id)
            for c in self._conflicts.values()
        )


# ---------------------------------------------------------------------------
# ConflictDetector — detects conflicts when new facts are extracted
# ---------------------------------------------------------------------------

class ConflictDetector:
    """Detects potential conflicts when a new fact is being added.

    Compares incoming claims against existing registry entries and
    flags possible contradictions for review.
    """

    def __init__(self, registry: Optional[ConflictRegistry] = None) -> None:
        self._registry = registry or ConflictRegistry.from_tolkien_defaults()

    @property
    def registry(self) -> ConflictRegistry:
        return self._registry

    def check_entity(
        self, entity_id: str, new_statement: str
    ) -> list[LoreConflict]:
        """Return any existing conflicts that involve the given entity."""
        return self._registry.by_entity(entity_id)

    def check_rule(
        self, rule_id: str
    ) -> list[LoreConflict]:
        """Return any existing conflicts involving the given lore rule."""
        return self._registry.by_rule(rule_id)

    def detect_new_conflict(
        self,
        entity_ids: list[str],
        rule_ids: list[str],
        new_statement: str,
        source_book: str,
        author_period: str,
        existing_statement: str,
        existing_source: str,
        existing_period: str,
    ) -> LoreConflict:
        """Create a new LoreConflict from a detected contradiction.

        The conflict is returned (not auto-added to registry) so the caller
        can decide whether to add it.

        Args:
            entity_ids: Entities involved in the contradiction.
            rule_ids: LoreRules affected.
            new_statement: The new incoming claim.
            source_book: Source book for the new claim.
            author_period: AuthorPeriod of the new claim.
            existing_statement: The existing claim being contradicted.
            existing_source: Source of the existing claim.
            existing_period: AuthorPeriod of the existing claim.

        Returns:
            A new LoreConflict (not yet in registry).
        """
        import hashlib
        id_src = f"{','.join(sorted(entity_ids))}__{new_statement[:30]}"
        conflict_id = "auto_" + hashlib.md5(id_src.encode()).hexdigest()[:8]

        # Classify type: if periods differ, it's a retcon; otherwise direct contradiction
        if author_period != existing_period:
            conflict_type = ConflictType.RETCON
        else:
            conflict_type = ConflictType.DIRECT_CONTRADICTION

        new_layer = infer_editorial_layer(source_book)
        existing_layer = infer_editorial_layer(existing_source)

        return LoreConflict(
            id=conflict_id,
            summary=f"Auto-detected conflict for entity/rule: {entity_ids or rule_ids}",
            conflict_type=conflict_type,
            entity_ids=entity_ids,
            rule_ids=rule_ids,
            claims=[
                ConflictClaim(
                    statement=existing_statement,
                    source_book=existing_source,
                    author_period=existing_period,
                    source_id=getattr(existing_layer, "source_id", None),
                    editorial_status=(
                        getattr(getattr(existing_layer, "editorial_status", None), "value", None)
                        if existing_layer
                        else None
                    ),
                    source_authority_weight=(
                        float(getattr(existing_layer, "authority_weight", 1.0))
                        if existing_layer
                        else None
                    ),
                ),
                ConflictClaim(
                    statement=new_statement,
                    source_book=source_book,
                    author_period=author_period,
                    source_id=getattr(new_layer, "source_id", None),
                    editorial_status=(
                        getattr(getattr(new_layer, "editorial_status", None), "value", None)
                        if new_layer
                        else None
                    ),
                    source_authority_weight=(
                        float(getattr(new_layer, "authority_weight", 1.0))
                        if new_layer
                        else None
                    ),
                ),
            ],
            resolution_policy=(
                ResolutionPolicy.USE_LATER_TEXT
                if conflict_type == ConflictType.RETCON
                else ResolutionPolicy.FLAG_FOR_HUMAN
            ),
            resolved=False,
        )


# ---------------------------------------------------------------------------
# ConflictAwareValidator
# ---------------------------------------------------------------------------

class ConflictAwareValidator:
    """Wraps LoreRuleValidator to suppress false positives from known conflicts.

    Per spec:
      - If resolution_policy = 'both_valid_in_universe' → suppress the violation entirely
      - If resolution_policy = 'flag_for_human' → downgrade HARD to SOFT warning
    """

    def __init__(
        self,
        rule_validator=None,
        conflict_registry: Optional[ConflictRegistry] = None,
    ) -> None:
        if rule_validator is None:
            from .rules import LoreRuleValidator, LoreRuleRegistry
            rule_validator = LoreRuleValidator(LoreRuleRegistry.from_tolkien_defaults())
        self._validator = rule_validator
        self._registry = conflict_registry or ConflictRegistry.from_tolkien_defaults()

    def validate_scene_context(self, context) -> LoreValidationResult:
        """Validate a scene context with conflict-aware suppression."""
        from ..models.lore_rule import LoreValidationResult, LoreViolation

        raw_result = self._validator.validate_scene_context(context)

        filtered_hard: list[LoreViolation] = []
        filtered_soft: list[LoreViolation] = list(raw_result.soft_warnings)

        for violation in raw_result.hard_violations:
            # Check all entities in the scene for suppression
            entity_ids = context.character_names or ["*"]
            suppressed = any(
                self._registry.suppresses_violation(violation.rule_id, eid)
                for eid in entity_ids
            )
            downgraded = any(
                self._registry.downgrades_to_soft(violation.rule_id, eid)
                for eid in entity_ids
            )

            if suppressed:
                continue  # Drop the violation entirely
            elif downgraded:
                # Downgrade to soft warning
                filtered_soft.append(LoreViolation(
                    rule_id=violation.rule_id,
                    rule_statement=violation.rule_statement,
                    hardness="SOFT",
                    description=f"[Downgraded — known conflict] {violation.description}",
                    blocking=False,
                ))
            else:
                filtered_hard.append(violation)

        passed = len(filtered_hard) == 0
        return LoreValidationResult(
            scene_id=raw_result.scene_id,
            passed=passed,
            hard_violations=filtered_hard,
            soft_warnings=filtered_soft,
            rules_checked=raw_result.rules_checked,
        )

    def validate_text(
        self, text: str, scene_id: str = "inline", story_era: Optional[str] = None
    ) -> LoreValidationResult:
        """Validate raw text with conflict suppression."""
        from .rules import _extract_context_from_text
        context = _extract_context_from_text(text, scene_id, story_era)
        return self.validate_scene_context(context)


# ---------------------------------------------------------------------------
# LoreConflictNeo4jWriter
# ---------------------------------------------------------------------------

class LoreConflictNeo4jWriter:
    """Write LoreConflict nodes and CONFLICTS_WITH edges to Neo4j."""

    def __init__(self, driver=None) -> None:
        self._driver = driver

    @property
    def driver(self):
        if self._driver is None:
            from ..graph.connection import get_driver
            self._driver = get_driver()
        return self._driver

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def ensure_schema(self) -> None:
        """Create constraint on LoreConflict.id. Idempotent."""
        with self.driver.session() as session:
            try:
                session.run(
                    "CREATE CONSTRAINT lore_conflict_id IF NOT EXISTS "
                    "FOR (c:LoreConflict) REQUIRE c.id IS UNIQUE"
                )
            except Exception:
                pass

    def upsert_conflict(self, conflict: LoreConflict) -> None:
        """Create or update a LoreConflict node (idempotent MERGE)."""
        with self.driver.session() as session:
            session.run(
                "MERGE (c:LoreConflict {id: $id}) SET c += $props",
                id=conflict.id,
                props=conflict.to_neo4j_props(),
            )

    def upsert_many(self, conflicts: list[LoreConflict]) -> int:
        """Write many conflicts. Returns count written."""
        count = 0
        with self.driver.session() as session:
            for conflict in conflicts:
                session.run(
                    "MERGE (c:LoreConflict {id: $id}) SET c += $props",
                    id=conflict.id,
                    props=conflict.to_neo4j_props(),
                )
                count += 1
        return count

    def create_conflicts_with_edge(
        self, rule_id_a: str, rule_id_b: str, conflict_id: str
    ) -> None:
        """Create (LoreRule)-[:CONFLICTS_WITH {conflict_id}]->(LoreRule) edge."""
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:LoreRule {id: $a})
                MATCH (b:LoreRule {id: $b})
                MERGE (a)-[r:CONFLICTS_WITH {conflict_id: $cid}]->(b)
                """,
                a=rule_id_a,
                b=rule_id_b,
                cid=conflict_id,
            )

    def resolve_conflict(
        self,
        conflict_id: str,
        policy: str,
        notes: str = "",
    ) -> None:
        """Update a LoreConflict node's resolution policy in Neo4j."""
        resolved = policy not in (
            ResolutionPolicy.FLAG_FOR_HUMAN,
            ResolutionPolicy.IRRESOLVABLE,
        )
        with self.driver.session() as session:
            session.run(
                """
                MATCH (c:LoreConflict {id: $id})
                SET c.resolution_policy = $policy,
                    c.resolution_notes = $notes,
                    c.resolved = $resolved
                """,
                id=conflict_id,
                policy=policy,
                notes=notes,
                resolved=resolved,
            )

    def query_conflicts(
        self,
        conflict_type: Optional[str] = None,
        resolved: Optional[bool] = None,
        needs_human: bool = False,
    ) -> list[dict]:
        """Query LoreConflict nodes from Neo4j with optional filters."""
        where_parts = []
        params: dict = {}
        if conflict_type:
            where_parts.append("c.conflict_type = $conflict_type")
            params["conflict_type"] = conflict_type
        if resolved is not None:
            where_parts.append("c.resolved = $resolved")
            params["resolved"] = resolved
        if needs_human:
            where_parts.append("c.resolution_policy = 'flag_for_human'")
            where_parts.append("c.resolved = false")

        where = "WHERE " + " AND ".join(where_parts) if where_parts else ""
        with self.driver.session() as session:
            result = session.run(
                f"MATCH (c:LoreConflict) {where} RETURN c ORDER BY c.resolved, c.id",
                **params,
            )
            return [dict(row["c"]) for row in result]
