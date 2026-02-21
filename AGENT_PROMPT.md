You are implementing GitHub issue #2 for the book-graph-analyzer project.

REPO: C:\Users\Tom\.openclaw\workspace\book-graph-analyzer
BRANCH: feat/issue-2-generic-entity-inference (already checked out)

## YOUR TASK
Implement generic entity inference â€” the ability to bootstrap canonical entity records from raw text WITHOUT any pre-existing seed files. This is the P0 blocker: entity resolution currently only works for Tolkien (hard-coded seeds) and produces garbage like 'Ã¢â‚¬Å“Well', 'None', 'Baggins' because spaCy en_core_web_sm has never seen fantasy names.

## STEP 0: Fix encoding crashes first (5 min, do this first)
In src/book_graph_analyzer/cli.py, add at the very top of the file after imports:
```python
import sys, io
if hasattr(sys.stdout, 'buffer') and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'buffer') and sys.stderr.encoding.lower() != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
```

Also grep for all `open(` calls in the lore/ directory and ensure they all have `encoding='utf-8'`.

## STEP 1: Build src/book_graph_analyzer/extract/bootstrap.py

This is the core deliverable. Implement a multi-pass EntityBootstrapper class:

```python
from dataclasses import dataclass, field
from typing import Optional
import re, json
from pathlib import Path
from rapidfuzz import fuzz
from ..llm import get_llm_client

@dataclass
class EntityCandidate:
    text: str
    frequency: int = 0
    contexts: list = field(default_factory=list)  # up to 5 x 150-char windows
    source: str = 'pattern'  # 'ner' | 'pattern' | 'caps'

@dataclass
class EntityCluster:
    variants: list  # list of str â€” all surface forms
    canonical_name: str = ''
    entity_type: str = 'unknown'  # 'character' | 'place' | 'object' | 'concept'
    frequency: int = 0
    contexts: list = field(default_factory=list)
    cluster_confidence: float = 0.0
    inferred_attributes: dict = field(default_factory=dict)
    needs_review: bool = False
    source: str = 'inferred'

@dataclass
class BootstrapResult:
    entities: list  # list of EntityCluster (accepted)
    flagged: list   # list of EntityCluster (needs_review=True)
    stats: dict = field(default_factory=dict)


class EntityBootstrapper:
    """Bootstrap canonical entity records from raw text without seed files.
    
    Multi-pass pipeline:
    1. Extract candidates (all capitalized name-like tokens)
    2. Cluster aliases (string similarity + transitivity)
    3. Canonicalize via LLM (confirm cluster, elect canonical name, infer type)
    4. Gate by confidence (auto-accept / flag / skip)
    """
    
    STRING_SIM_THRESHOLD = 80   # rapidfuzz ratio 0-100
    MIN_FREQUENCY = 2           # ignore hapax legomena
    ACCEPT_THRESHOLD = 0.80     # auto-accept clusters above this confidence
    REVIEW_THRESHOLD = 0.55     # flag-for-review between this and ACCEPT
    CONTEXT_WINDOW = 150        # chars either side of mention
    MAX_CONTEXTS_PER_ENTITY = 5
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = get_llm_client() if use_llm else None
    
    def extract_candidates(self, text: str): list:
        """Pass 1: Extract all capitalized name-like tokens with frequency and context."""
        # Pattern: capitalized word sequences (2-4 words), excluding sentence starts
        # Also catches: 'the Grey Wizard', 'Lord of the Rings' style epithets
        candidates = {}
        
        # Pattern for proper nouns: 1-4 consecutive capitalized words
        pattern = re.compile(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b')
        
        # Epithet patterns: 'the [Adj] [Noun]', '[Title] [Name]'
        epithet_pattern = re.compile(
            r'\b(?:the\s+)?(?:King|Queen|Lord|Lady|Prince|Captain|Wizard|Grey|White|Dark|Black|High|Great)\s+(?:of\s+)?([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b',
            re.IGNORECASE
        )
        
        for match in pattern.finditer(text):
            name = match.group(1).strip()
            if len(name) < 3 or name in ('The', 'And', 'But', 'For', 'Not', 'He', 'She', 'It', 'They', 'Was', 'Had', 'Did'):
                continue
            
            start = max(0, match.start() - self.CONTEXT_WINDOW)
            end = min(len(text), match.end() + self.CONTEXT_WINDOW)
            context = text[start:end].replace('\n', ' ').strip()
            
            if name not in candidates:
                candidates[name] = EntityCandidate(text=name, frequency=0, contexts=[], source='pattern')
            candidates[name].frequency += 1
            if len(candidates[name].contexts) < self.MAX_CONTEXTS_PER_ENTITY:
                candidates[name].contexts.append(context)
        
        # Filter by minimum frequency
        return [c for c in candidates.values() if c.frequency >= self.MIN_FREQUENCY]
    
    def cluster_aliases(self, candidates: list): list:
        """Pass 2: Group candidates that are likely aliases using string similarity + transitivity.
        
        Algorithm:
        1. Build similarity pairs where string ratio > threshold
        2. Apply transitivity: if A~B and B~C then A,B,C are one cluster
        3. Merge candidate metadata within each cluster
        """
        names = [c.text for c in candidates]
        cand_map = {c.text: c for c in candidates}
        
        # Build similarity pairs
        pairs = set()
        for i, a in enumerate(names):
            for b in names[i+1:]:
                # Check if b is a substring of a or vice versa (alias pattern)
                a_lower, b_lower = a.lower(), b.lower()
                is_substring = a_lower in b_lower or b_lower in a_lower
                
                ratio = fuzz.ratio(a, b)
                token_ratio = fuzz.token_set_ratio(a, b)
                
                if is_substring or ratio >= self.STRING_SIM_THRESHOLD or token_ratio >= self.STRING_SIM_THRESHOLD:
                    pairs.add((a, b))
        
        # Apply transitivity to build clusters
        clusters = []
        remaining = set(names)
        
        while remaining:
            # Start a new cluster with any remaining name
            seed = next(iter(remaining))
            cluster_members = {seed}
            remaining.discard(seed)
            
            # Expand by transitivity
            changed = True
            while changed:
                changed = False
                for a, b in list(pairs):
                    if a in cluster_members and b in remaining:
                        cluster_members.add(b)
                        remaining.discard(b)
                        changed = True
                    elif b in cluster_members and a in remaining:
                        cluster_members.add(a)
                        remaining.discard(a)
                        changed = True
            
            # Build cluster from members
            members = list(cluster_members)
            total_freq = sum(cand_map[m].frequency for m in members if m in cand_map)
            all_contexts = []
            for m in members:
                if m in cand_map:
                    all_contexts.extend(cand_map[m].contexts)
            
            # Confidence: how well do members agree? 
            # Single-member clusters (no aliases found) get lower confidence
            conf = 0.7 if len(members) == 1 else 0.85
            
            clusters.append(EntityCluster(
                variants=members,
                frequency=total_freq,
                contexts=all_contexts[:self.MAX_CONTEXTS_PER_ENTITY],
                cluster_confidence=conf,
                needs_review=(conf < self.ACCEPT_THRESHOLD)
            ))
        
        return sorted(clusters, key=lambda c: c.frequency, reverse=True)
    
    def canonicalize_clusters(self, clusters: list): list:
        """Pass 3: Use LLM to confirm clusters, elect canonical names, infer entity types."""
        if not self.use_llm or not self._llm:
            # Without LLM: elect longest name as canonical, type=unknown
            for cluster in clusters:
                cluster.canonical_name = max(cluster.variants, key=len)
                cluster.entity_type = 'unknown'
            return clusters
        
        canonicalized = []
        for cluster in clusters:
            if cluster.frequency < self.MIN_FREQUENCY:
                continue
            
            variants_str = ', '.join(f'"{v}"' for v in cluster.variants[:10])
            contexts_str = '\n'.join(f'  - ...{c[:200]}...' for c in cluster.contexts[:3])
            
            prompt = f"""Analyze these name variants found in a literary text:

Variants: {variants_str}
Frequency: {cluster.frequency} total mentions

Context examples:
{contexts_str}

Answer in JSON only, no explanation:
{{
  "same_entity": true/false,
  "canonical_name": "the best canonical name for this entity",
  "entity_type": "character" or "place" or "object" or "concept" or "unknown",
  "confidence": 0.0-1.0,
  "inferred_attributes": {{
    "notes": "any inferred info like race, role, era"
  }}
}}"""
            
            try:
                response = self._llm.generate(prompt, temperature=0.1, max_tokens=300)
                data = self._llm.extract_json(response)
                
                if data and isinstance(data, dict):
                    if not data.get('same_entity', True):
                        # Cluster should be split - flag for review
                        cluster.needs_review = True
                        cluster.canonical_name = max(cluster.variants, key=len)
                        cluster.cluster_confidence = 0.4
                    else:
                        cluster.canonical_name = data.get('canonical_name', cluster.variants[0])
                        cluster.entity_type = data.get('entity_type', 'unknown')
                        cluster.cluster_confidence = float(data.get('confidence', 0.7))
                        cluster.inferred_attributes = data.get('inferred_attributes', {})
                        cluster.needs_review = cluster.cluster_confidence < self.ACCEPT_THRESHOLD
                else:
                    cluster.canonical_name = max(cluster.variants, key=len)
                    cluster.needs_review = True
            except Exception as e:
                cluster.canonical_name = max(cluster.variants, key=len)
                cluster.needs_review = True
            
            canonicalized.append(cluster)
        
        return canonicalized
    
    def bootstrap(self, text: str): BootstrapResult:
        """Full pipeline: text: canonical entities with confidence scores."""
        print(f"Bootstrapping entities from {len(text)} chars...")
        
        # Pass 1
        candidates = self.extract_candidates(text)
        print(f"  Pass 1: {len(candidates)} candidates extracted")
        
        # Pass 2
        clusters = self.cluster_aliases(candidates)
        print(f"  Pass 2: {len(clusters)} clusters formed")
        
        # Pass 3
        clusters = self.canonicalize_clusters(clusters)
        print(f"  Pass 3: Canonicalized {len(clusters)} clusters")
        
        # Gate by confidence
        accepted = [c for c in clusters if c.cluster_confidence >= self.ACCEPT_THRESHOLD]
        flagged = [c for c in clusters if self.REVIEW_THRESHOLD <= c.cluster_confidence < self.ACCEPT_THRESHOLD]
        skipped = [c for c in clusters if c.cluster_confidence < self.REVIEW_THRESHOLD]
        
        print(f"  Result: {len(accepted)} accepted, {len(flagged)} flagged for review, {len(skipped)} skipped")
        
        return BootstrapResult(
            entities=accepted,
            flagged=flagged,
            stats={
                'candidates': len(candidates),
                'clusters': len(clusters),
                'accepted': len(accepted),
                'flagged': len(flagged),
                'skipped': len(skipped),
            }
        )
```

## STEP 2: Add CLI command in cli.py

Find the main CLI group and add:

```python
@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), help='Output JSON file')
@click.option('--neo4j', is_flag=True, help='Write accepted entities to Neo4j')
@click.option('--no-llm', is_flag=True, help='Skip LLM canonicalization (faster, less accurate)')
@click.option('--min-frequency', default=2, help='Minimum mentions to consider')
def bootstrap(input_path, output, neo4j, no_llm, min_frequency):
    """Bootstrap canonical entities from text without seed files."""
    from .extract.bootstrap import EntityBootstrapper
    import json
    
    path = Path(input_path)
    if path.is_dir():
        texts = list(path.glob('*.txt'))
    else:
        texts = [path]
    
    bootstrapper = EntityBootstrapper(use_llm=not no_llm)
    bootstrapper.MIN_FREQUENCY = min_frequency
    
    all_results = []
    for text_file in texts:
        print(f"Processing {text_file.name}...")
        text = text_file.read_text(encoding='utf-8', errors='replace')
        result = bootstrapper.bootstrap(text)
        all_results.append({'file': str(text_file.name), 'result': result})
        
        # Print summary
        print(f"  Accepted: {result.stats['accepted']} entities")
        print(f"  Flagged for review: {result.stats['flagged']}")
        for entity in result.entities[:10]:
            print(f"    [{entity.entity_type}] {entity.canonical_name} (variants: {entity.variants}, conf: {entity.cluster_confidence:.2f})")
    
    if output:
        # Serialize to JSON
        output_data = []
        for r in all_results:
            for entity in r['result'].entities + r['result'].flagged:
                output_data.append({
                    'canonical_name': entity.canonical_name,
                    'variants': entity.variants,
                    'entity_type': entity.entity_type,
                    'frequency': entity.frequency,
                    'cluster_confidence': entity.cluster_confidence,
                    'needs_review': entity.needs_review,
                    'inferred_attributes': entity.inferred_attributes,
                    'source': 'inferred',
                    'sample_context': entity.contexts[0] if entity.contexts else ''
                })
        Path(output).write_text(json.dumps(output_data, indent=2), encoding='utf-8')
        print(f"Written to {output}")
```

## STEP 3: Write tests/test_bootstrap.py

```python
import pytest
from pathlib import Path
from book_graph_analyzer.extract.bootstrap import EntityBootstrapper, EntityCandidate

# Sample text with known entities - Bilbo has multiple name forms
SAMPLE_TEXT = """
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,
filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.

Bilbo Baggins was a very well-to-do hobbit. Mr. Baggins had lived in his hobbit-hole
at Bag End his whole life. Bilbo was respected and prosperous.

One morning Gandalf came by. The wizard Gandalf sat down on a bench outside the door.
Gandalf was known throughout the Shire as a maker of remarkable fireworks.

The Shire was a comfortable land. In all the Shire there was not a man or a hobbit
who had not heard of Bilbo Baggins. Bag End sat at the end of The Water.

Gandalf and Mr. Baggins spoke for some time. Bilbo was not pleased to be interrupted.
"""

def test_extract_candidates():
    b = EntityBootstrapper(use_llm=False)
    candidates = b.extract_candidates(SAMPLE_TEXT)
    names = [c.text for c in candidates]
    # Bilbo and Baggins should appear (both high frequency)
    assert any('Bilbo' in n for n in names), f"Bilbo not found in {names}"
    assert any('Gandalf' in n for n in names), f"Gandalf not found in {names}"

def test_cluster_aliases():
    b = EntityBootstrapper(use_llm=False)
    candidates = b.extract_candidates(SAMPLE_TEXT)
    clusters = b.cluster_aliases(candidates)
    
    # Find cluster containing Gandalf
    gandalf_clusters = [c for c in clusters if any('Gandalf' in v for v in c.variants)]
    assert len(gandalf_clusters) >= 1, "Gandalf should be in at least one cluster"
    
    # Find cluster containing Bilbo / Mr. Baggins
    bilbo_clusters = [c for c in clusters if any('Bilbo' in v or 'Baggins' in v for v in c.variants)]
    assert len(bilbo_clusters) >= 1, "Bilbo/Baggins cluster not found"

def test_bootstrap_no_llm():
    b = EntityBootstrapper(use_llm=False)
    result = b.bootstrap(SAMPLE_TEXT)
    
    assert result.stats['candidates'] > 0
    assert result.stats['clusters'] > 0
    assert len(result.entities) + len(result.flagged) > 0
    
    all_entities = result.entities + result.flagged
    all_names = [e.canonical_name for e in all_entities] + [v for e in all_entities for v in e.variants]
    
    assert any('Gandalf' in n for n in all_names), "Gandalf should appear in results"

def test_candidate_frequency():
    b = EntityBootstrapper(use_llm=False)
    b.MIN_FREQUENCY = 1
    candidates = b.extract_candidates(SAMPLE_TEXT)
    # All candidates should have frequency >= 1
    assert all(c.frequency >= 1 for c in candidates)
    # Gandalf appears 4 times in sample
    gandalf = next((c for c in candidates if c.text == 'Gandalf'), None)
    assert gandalf is not None
    assert gandalf.frequency >= 3

def test_context_windows():
    b = EntityBootstrapper(use_llm=False)
    candidates = b.extract_candidates(SAMPLE_TEXT)
    # Every candidate should have at least one context
    for c in candidates:
        assert len(c.contexts) >= 1, f"{c.text} has no contexts"
        # Context should be a non-empty string
        assert len(c.contexts[0]) > 10
```

## STEP 4: Run tests and verify

```bash
cd C:\Users\Tom\.openclaw\workspace\book-graph-analyzer
pip install -e . -q
pytest tests/test_bootstrap.py -v
bga bootstrap data/texts/istari_chapter.txt --no-llm
```

## STEP 5: Commit and push

```bash
git add -A
git commit -m "feat: generic entity bootstrapping pipeline (issue #2)

- Multi-pass EntityBootstrapper: candidates: clustering: canonicalization
- Transitivity-based alias clustering using rapidfuzz string similarity
- LLM canonicalization with confidence gating (accept/review/skip)
- Context-aware candidate extraction with 150-char windows
- CLI command: bga bootstrap
- Tests: tests/test_bootstrap.py
- Fix: utf-8 encoding in stdout/stderr (resolves cp1252 crashes)

Closes #2"

git push origin feat/issue-2-generic-entity-inference
```

Then output a summary of what you built. When completely done run:
openclaw system event --text "Issue #2 done: entity bootstrapper built and pushed. Branch feat/issue-2-generic-entity-inference ready for PR." --mode now

