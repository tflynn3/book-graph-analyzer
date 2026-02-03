# Lore-Consistent Story Generation: Research Report

## Executive Summary

This report evaluates approaches for generating Tolkien-style narrative content while maintaining lore consistency. The goal: explore many possible story paths, let you flag "non-Tolkien" moments, and produce canonically sound output.

**Recommendation:** Hybrid approach combining **MCTS for exploration** + **Constitutional-style critique** for consistency, with human-in-the-loop feedback.

---

## The Challenge

Generating lore-consistent fiction requires balancing:
1. **Creativity** - Exploring diverse narrative possibilities
2. **Consistency** - Respecting established rules (world bible, timelines, character voices)
3. **Style** - Maintaining authorial voice (Burrows' Delta, voice profiles)
4. **Human oversight** - Catching "non-Tolkien" moments before they propagate

Traditional LLM generation is **linear** - one path, no backtracking. We need **branching exploration** with **constraint checking**.

---

## Approaches Evaluated

### 1. Genetic Algorithms + LLM (GA-LLM)

**How it works:**
- Treat each story/passage as a "gene" (individual)
- Population of N story variants evolve over generations
- **Selection**: Keep highest-quality stories
- **Crossover**: LLM combines elements from two parent stories
- **Mutation**: LLM introduces variations

**Fitness function for our use case:**
```
fitness = (style_score * 0.3) +      # Burrows' Delta vs Tolkien
          (lore_consistency * 0.4) +  # World bible rule violations
          (narrative_quality * 0.3)   # Coherence, engagement
```

**Pros:**
- Natural fit for exploring "many paths" (population diversity)
- Human can inject preferences as fitness modifiers
- Works with black-box LLMs (no gradients needed)
- Recent research (GA-LLM 2025) shows strong results for constrained generation

**Cons:**
- Computationally expensive (many LLM calls per generation)
- Crossover in narrative space is tricky - combining story A's plot with story B's dialogue can produce incoherence
- Convergence can be slow for long-form content

**Best for:** Exploring diverse short passages, generating multiple scene variants

---

### 2. Monte Carlo Tree Search (MCTS)

**How it works:**
- Build a tree where each node is a story state
- **Selection**: UCB1 to balance exploration/exploitation
- **Expansion**: LLM generates next passage options
- **Simulation**: Rollout to estimate story quality
- **Backpropagation**: Update node values

**Key paper:** "Narrative Studio" (2025) - exactly this use case!
- Tree-based UI for branching narratives
- Knowledge graph grounding for entity consistency
- MCTS auto-expands promising branches

**Pros:**
- Natural "what-if" exploration (user can branch at any point)
- Backtracking is built-in (unlike linear generation)
- Can integrate lore-checker as simulation evaluator
- Visual tree UI makes human oversight intuitive

**Cons:**
- Rollouts are expensive (simulate full story to evaluate)
- Deep trees = exponential paths
- Need good heuristic for node evaluation

**Best for:** Interactive co-writing, exploring plot branches, long-form planning

---

### 3. Quality-Diversity (MAP-Elites)

**How it works:**
- Maintain an "archive" of diverse high-quality solutions
- Define a behavior space (e.g., tone × pacing × character focus)
- Each cell in the archive holds the best story for that behavior niche
- Evolution fills the archive with diverse yet high-quality options

**Pros:**
- Guarantees diversity (not just optimization)
- Archive provides menu of alternatives
- Naturally avoids mode collapse

**Cons:**
- Defining the behavior space for narratives is non-obvious
- Archive size grows with behavior dimensions
- Less interactive than MCTS

**Best for:** Generating a diverse portfolio of story options upfront

---

### 4. Hierarchical Planning (DOME, Re3)

**How it works:**
- Generate outline first (plot beats, chapter structure)
- Recursively elaborate: outline → scenes → passages → sentences
- Memory-enhanced: maintain context across expansions
- Dynamic: adjust outline as generation reveals new possibilities

**Pros:**
- Coherent long-form output (plot doesn't drift)
- Natural checkpoints for human review
- Lore-checking can happen at outline level (cheap)

**Cons:**
- Rigid if outline is fixed
- Doesn't explore alternative plot directions

**Best for:** Generating a single coherent long-form story

---

### 5. Constitutional AI / Self-Critique

**How it works:**
- Generate candidate passage
- Self-critique against a "constitution" (rules)
- Revise to fix violations
- Iterate until satisfactory

**Our constitution could include:**
```
1. Does this passage violate any world bible rules?
2. Is the character's voice consistent with their profile?
3. Does the timeline make sense given known events?
4. Would Tolkien write this sentence? (style check)
```

**Pros:**
- Direct enforcement of constraints
- No training required (prompt-based)
- Human can add rules dynamically

**Cons:**
- Single-path (no exploration)
- Self-critique can miss subtle violations
- Iterative revision is slow

**Best for:** Post-generation cleanup, enforcing hard constraints

---

## Recommendation: Hybrid Architecture

Combine strengths of multiple approaches:

```
┌─────────────────────────────────────────────────────────┐
│                    GENERATION PIPELINE                   │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. HIERARCHICAL PLANNING (outline generation)          │
│     - Generate plot outline                             │
│     - Check outline against world bible (cheap)         │
│     - Human approves/modifies outline                   │
│                                                         │
│  2. MCTS EXPLORATION (scene generation)                 │
│     - For each outline beat, explore branches           │
│     - UCB1 selection with lore-consistency bonus        │
│     - Backprop includes style score (Burrows' Delta)    │
│     - Human can prune "non-Tolkien" branches            │
│                                                         │
│  3. CONSTITUTIONAL CRITIQUE (passage refinement)        │
│     - Self-critique against world bible rules           │
│     - Voice profile consistency check                   │
│     - Timeline validation                               │
│     - Revise until constraints satisfied                │
│                                                         │
│  4. HUMAN-IN-THE-LOOP                                   │
│     - Flag passages for review                          │
│     - Feedback updates fitness/evaluation               │
│     - Can inject "this feels non-Tolkien" as signal     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Why Not Pure GA?

Genetic algorithms are great for:
- Optimizing short passages
- Exploring parameter spaces
- Generating diverse variants

But for **narrative**, the problem is **crossover**. Combining plot elements from two stories often produces incoherence. MCTS handles branching more naturally because each path is coherent by construction.

That said, GA could work well for:
- Evolving prompts/system instructions
- Optimizing outline structures
- Generating diverse opening hooks

### Human-in-the-Loop Design

```python
class LoreAwareGenerator:
    def generate_with_oversight(self, outline, max_flags=3):
        """Generate with human feedback integration."""
        
        for beat in outline.beats:
            # MCTS exploration
            tree = self.mcts_explore(beat, iterations=50)
            
            # Get top branches
            candidates = tree.get_top_k(k=3)
            
            # Constitutional critique
            for candidate in candidates:
                candidate.score = self.critique(candidate)
            
            # Flag low-confidence or style-violation passages
            flagged = [c for c in candidates if c.needs_review]
            
            if flagged:
                # Human reviews flagged content
                human_choice = yield FlaggedContent(flagged)
                # Update evaluator with human preference
                self.update_from_feedback(human_choice)
            else:
                # Auto-select best candidate
                yield candidates[0]
```

### Evaluation Function (Fitness)

```python
def evaluate_passage(passage, context):
    """Multi-objective evaluation for MCTS/GA."""
    
    scores = {
        # Style consistency (Burrows' Delta)
        'style': burrows_delta(passage, tolkien_reference),
        
        # World bible compliance
        'lore': world_bible.check_violations(passage),
        
        # Character voice match
        'voice': voice_profile.similarity(passage, character),
        
        # Timeline consistency
        'temporal': event_graph.validate_sequence(passage),
        
        # Narrative quality (LLM judge)
        'quality': llm_judge.score(passage, criteria=[
            'coherence', 'engagement', 'prose_quality'
        ]),
    }
    
    # Weighted combination (tunable)
    weights = {'style': 0.2, 'lore': 0.35, 'voice': 0.15, 
               'temporal': 0.15, 'quality': 0.15}
    
    return sum(scores[k] * weights[k] for k in scores)
```

---

## Implementation Roadmap

### Phase 1: Constitutional Critique (Quick Win)
- Add `bga generate passage --critique` command
- Use world bible rules as constitution
- Self-revision loop until no violations
- **Effort:** 1-2 days

### Phase 2: MCTS Exploration
- Implement tree structure for story states
- UCB1 selection with lore-consistency bonus
- Simple rollout using LLM continuation
- Tree visualization (optional)
- **Effort:** 3-5 days

### Phase 3: Human-in-the-Loop
- CLI interface for reviewing flagged passages
- Feedback storage and evaluator updates
- "Non-Tolkien" flag propagates to evaluation
- **Effort:** 2-3 days

### Phase 4: Hierarchical Planning
- Outline generation with world bible grounding
- Beat-by-beat expansion
- Memory context across sections
- **Effort:** 3-4 days

### Phase 5: GA for Prompt Evolution (Optional)
- Evolve system prompts for better style
- Population of instruction variants
- Fitness = output quality on test passages
- **Effort:** 2-3 days

---

## Cost Estimation (HuggingFace)

Using Llama-3.1-70B-Instruct:

| Operation | Calls/Story | Tokens/Call | Cost/Story |
|-----------|-------------|-------------|------------|
| Outline generation | 1 | ~2000 | $0.06 |
| MCTS exploration (50 iterations) | 50 | ~500 | $0.75 |
| Constitutional critique (3 passes) | 3 | ~1000 | $0.09 |
| **Total per story chapter** | - | - | **~$0.90** |

With $2 budget: ~2 chapters with heavy exploration, or ~5 chapters with lighter exploration.

---

## References

1. "GA-LLM: Hybrid Framework for Structured Task Optimization" (2025)
2. "Narrative Studio: Visual narrative exploration using LLMs and MCTS" (2025)
3. "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
4. "DOME: Dynamic Hierarchical Outlining with Memory-Enhancement" (2024)
5. "MAP-Elites: Quality-Diversity Search" (Mouret & Clune, 2015)
6. "When Large Language Models Meet Evolutionary Algorithms" (2024)

---

## TL;DR

- **GA is cool but crossover is weird for narratives**
- **MCTS is the winner for "explore many paths"**
- **Constitutional critique for constraint enforcement**
- **Hierarchical planning for coherent long-form**
- **Human-in-the-loop via flagging system**

Start with Constitutional Critique (easy win), then add MCTS for exploration. GA is better suited for evolving prompts than stories themselves.
