# Style Analysis Research Document

**Book Graph Analyzer - Phase 4: Style Analysis**

This document synthesizes current academic research on stylometry, author style fingerprinting, and computational approaches to literary style analysis. It serves as the theoretical foundation for implementing Phase 4 of the Book Graph Analyzer.

---

## Executive Summary

Stylometry—the quantitative analysis of writing style—provides a robust framework for extracting measurable features that form an author's unique "fingerprint." These features can be used to:

1. **Identify authorship** of anonymous or disputed texts
2. **Characterize writing style** for comparative analysis
3. **Guide text generation** to mimic a specific author's voice

For our Book Graph Analyzer, we will extract a comprehensive style fingerprint that can later inform style-constrained generation (Phase 8).

---

## 1. Foundations of Stylometry

### 1.1 Definition and History

Stylometry is "the analysis of style features that can be statistically quantified, such as sentence length, vocabulary diversity, and frequencies (of words, word forms, etc.)" (Scielo, 2018). The field has roots in the 19th century, with early applications including the Federalist Papers authorship debate (Mosteller & Wallace, 1963).

### 1.2 Core Principle: The Stylistic Fingerprint

> "Stylometry operates by analysing measurable linguistic features such as word frequency, sentence length, syntactic structures, or punctuation patterns, which collectively form a unique 'fingerprint' of an author's or text's style." (Nature, 2025)

Key insight: Style is revealed through **how** a text is written, not **what** it is about. Function words (the, of, and, etc.) and structural patterns are more reliable indicators than content words.

---

## 2. Stylometric Features Taxonomy

Based on the surveyed literature, stylometric features fall into several categories:

### 2.1 Lexical Features

| Feature | Description | Relevance |
|---------|-------------|-----------|
| **Vocabulary richness** | Type-token ratio (unique words / total words) | Measures lexical diversity |
| **Word frequency distribution** | Most frequent words (MFW) | Core of Burrows' Delta method |
| **Average word length** | Mean characters per word | Correlates with complexity |
| **Hapax legomena** | Words appearing only once | Indicator of vocabulary breadth |
| **Function word frequency** | Usage of the, and, of, to, etc. | Most stable stylistic markers |

### 2.2 Syntactic Features

| Feature | Description | Relevance |
|---------|-------------|-----------|
| **Sentence length** | Mean words per sentence | Basic complexity indicator |
| **Sentence length variance** | Standard deviation | Measures consistency |
| **Clause depth** | Nested subordinate clauses | Syntactic complexity |
| **POS tag distribution** | Noun/verb/adjective ratios | Grammatical preferences |
| **Dependency patterns** | Subject-verb-object structures | Syntactic fingerprint |

### 2.3 Structural Features

| Feature | Description | Relevance |
|---------|-------------|-----------|
| **Paragraph length** | Mean sentences per paragraph | Organizational style |
| **Dialogue ratio** | Dialogue / narration balance | Genre indicator |
| **Punctuation patterns** | Comma, semicolon, dash usage | Rhythmic signature |
| **Passive vs. active voice** | Voice ratio | Narrative distance |

### 2.4 Readability Metrics

| Metric | Formula Basis | What It Measures |
|--------|--------------|------------------|
| **Flesch Reading Ease** | Sentence & word length | Accessibility |
| **Flesch-Kincaid Grade** | Same, scaled to grade level | Education level required |
| **Gunning Fog Index** | Complex words & sentence length | Text complexity |
| **SMOG Index** | Polysyllabic words | Years of education needed |

### 2.5 Advanced Features (Deep Learning Era)

| Feature | Source | Description |
|---------|--------|-------------|
| **Character n-grams** | Zamir et al. (2024) | Subword patterns, typos, formatting |
| **Document embeddings** | Markov et al. (2017) | Dense vector representations |
| **Transformer features** | BERT/GPT | Contextual linguistic patterns |

---

## 3. Key Methods from Literature

### 3.1 Burrows' Delta (Classic)

The gold standard for computational literary studies:

1. Calculate frequency of Most Frequent Words (typically 100-500)
2. Normalize using z-scores
3. Compute average absolute difference between texts

> "A lower Delta value indicates greater stylistic similarity." (O'Sullivan, 2024)

**Strengths**: Interpretable, language-independent, well-validated
**Limitations**: Requires sufficient text length (~5000+ words ideal)

### 3.2 Deep Learning Approaches

From the IJRASET review (2024):

| Architecture | Application | Strength |
|--------------|-------------|----------|
| **CNN** | Character-level patterns | Captures local features |
| **RNN/LSTM** | Sequential dependencies | Models writing flow |
| **Transformers** | Contextual embeddings | State-of-the-art accuracy |

> "Deep learning models can significantly outperform traditional methods in identifying authorship, even in challenging scenarios involving multiple authors, short texts, and cross-genre analysis." (IJRASET, 2024)

### 3.3 Style Representations for Generation

From Andrews et al. (2024) - "Learning to Generate Text in Arbitrary Writing Styles":

- **Contrastively-trained representations** capture stylometric features as dense vectors
- These vectors can **guide text generation** toward a target style
- Approach: regression model predicts style vector from partial text
- Enables style transfer while preserving meaning

**Key insight for Phase 8**: Style vectors extracted in Phase 4 can directly inform generation.

---

## 4. Human vs. AI Writing Differences

From the Nature stylometric comparison (2025):

| Characteristic | Human Writing | AI Writing |
|----------------|---------------|------------|
| **Clustering** | Dispersed, varied | Tight, uniform |
| **Stylistic diversity** | High (individual voices) | Low (model-specific) |
| **Pattern consistency** | Variable across authors | Predictable within model |

> "Human-authored texts display a much greater degree of variability, forming broader and less compact clusters. This variability likely reflects the diversity of individual writing styles, levels of creativity, and engagement." (Nature, 2025)

**Implication**: A good style fingerprint should capture the *variability* within an author's work, not just averages.

---

## 5. Implementation Strategy for Phase 4

Based on the research, our Phase 4 implementation should extract:

### 5.1 Sentence-Level Metrics (Per Passage)

```python
@dataclass
class SentenceMetrics:
    length: int                    # Word count
    clause_depth: int             # Nested clauses
    voice: str                    # active/passive
    has_dialogue: bool            # Contains quoted speech
    punctuation_density: float    # Punctuation marks per word
    avg_word_length: float        # Mean characters per word
```

### 5.2 Vocabulary Analysis (Full Text)

```python
@dataclass
class VocabularyProfile:
    type_token_ratio: float       # Lexical diversity
    hapax_ratio: float            # Unique words ratio
    avg_word_length: float        # Overall
    word_length_distribution: dict  # {1: n, 2: n, ...}
    function_word_frequencies: dict  # Top 100 function words
    archaism_count: int           # From curated list
    invented_word_candidates: list  # Unknown to dictionary
```

### 5.3 Passage Classification

```python
class PassageType(Enum):
    DIALOGUE = "dialogue"
    ACTION = "action"
    DESCRIPTION = "description"
    TRAVEL = "travel"
    BATTLE = "battle"
    EXPOSITION = "exposition"
```

### 5.4 Author Voice Profile (Aggregate)

```python
@dataclass
class AuthorStyleFingerprint:
    # Distributions (mean, std, min, max, percentiles)
    sentence_length_dist: Distribution
    word_length_dist: Distribution
    paragraph_length_dist: Distribution
    
    # Ratios
    dialogue_ratio: float
    passive_voice_ratio: float
    question_ratio: float
    exclamation_ratio: float
    
    # Vocabulary signature
    vocabulary_profile: VocabularyProfile
    
    # Function word signature (Burrows' Delta compatible)
    function_word_zscores: dict[str, float]
    
    # Scene type distribution
    passage_type_distribution: dict[PassageType, float]
    
    # Readability
    flesch_reading_ease: float
    flesch_kincaid_grade: float
```

---

## 6. Validation Criteria

From the literature, our style analysis should:

1. **Distinguish authors**: Tolkien's fingerprint should differ measurably from other fantasy authors
2. **Show internal consistency**: Different Tolkien works should cluster together
3. **Reveal known differences**: The Hobbit should show simpler metrics than The Silmarillion
4. **Capture character voices**: (Phase 5) Gandalf should speak differently than Bilbo

---

## 7. References

### Primary Sources

1. **Zamir, M.T. et al. (2024)**. "Stylometry Analysis of Multi-authored Documents for Authorship and Author Style Change Detection." arXiv:2401.06752. 
   - Merit-based fusion of NLP algorithms
   - Importance of special characters in style detection

2. **O'Sullivan et al. (2025)**. "Stylometric comparisons of human versus AI-generated creative writing." Nature Humanities & Social Sciences Communications.
   - Burrows' Delta methodology
   - Human vs. AI clustering patterns

3. **Andrews, N. et al. (2024)**. "Learning to Generate Text in Arbitrary Writing Styles." arXiv:2312.17242.
   - Contrastive style representations
   - StyleMC approach for guided generation

4. **IJRASET (2024)**. "Deep Learning for Stylometry and Authorship Attribution: A Review."
   - CNN/RNN/Transformer architectures
   - Comparison with traditional methods

5. **Lagutina, K. et al. (2019)**. "A Survey on Stylometric Text Features." FRUCT Conference.
   - Comprehensive feature taxonomy
   - Author profiling approaches

### Supporting Sources

6. Stamatatos, E. (2009). "A Survey of Modern Authorship Attribution Methods." JASIST.

7. Neal, T. et al. (2017). "Surveying Stylometry Techniques and Applications." ACM Computing Surveys.

8. Eder, M. (2017). "Short Samples in Authorship Attribution." Digital Scholarship in the Humanities.

9. Can, F. & Patton, J.M. (2004). "Change of Writing Style with Time." Computers and the Humanities.

10. Hirst, G. & Feng, V.W. (2012). "Changes in Style in Authors with Alzheimer's Disease." English Studies.

---

## 8. Next Steps

1. **Implement sentence-level metrics** using spaCy for POS tagging and dependency parsing
2. **Build vocabulary analyzer** with function word extraction
3. **Create passage classifier** (rule-based initially, LLM-enhanced later)
4. **Aggregate into AuthorStyleFingerprint** dataclass
5. **Validate on The Hobbit** - compare early vs. late chapters
6. **Cross-validate** - compare Tolkien to a control author (e.g., C.S. Lewis)

---

*Document prepared for Book Graph Analyzer Phase 4 implementation*
*Last updated: 2026-02-02*
