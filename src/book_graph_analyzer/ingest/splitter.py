"""Split text into structured passages."""

import re
from dataclasses import dataclass

_STANDALONE_CHAPTER_RE = re.compile(
    r"^\s*_?(Chapter[ \t]+[IVXLC\d]+)_?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_PROLOGUE_RE = re.compile(r"^\s*PROLOGUE\s*$", re.MULTILINE | re.IGNORECASE)
_APPENDIX_RE = re.compile(r"^\s*APPENDICES?\s*$", re.MULTILINE | re.IGNORECASE)
_INDEX_RE = re.compile(r"^\s*INDEXES?\s*$", re.MULTILINE | re.IGNORECASE)
_STRUCTURAL_TOKENS = {
    "appendix",
    "appendices",
    "book",
    "chapter",
    "contents",
    "epilogue",
    "foreword",
    "index",
    "indexes",
    "part",
    "preface",
    "prologue",
}
_LOWERCASE_TITLE_WORDS = {"a", "an", "and", "at", "in", "of", "on", "the", "to"}


@dataclass
class Passage:
    """A passage (typically a sentence) with its location metadata."""

    id: str
    text: str
    book: str
    chapter: str
    chapter_num: int
    paragraph_num: int
    sentence_num: int
    char_offset: int

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "text": self.text,
            "book": self.book,
            "chapter": self.chapter,
            "chapter_num": self.chapter_num,
            "paragraph_num": self.paragraph_num,
            "sentence_num": self.sentence_num,
            "char_offset": self.char_offset,
        }


def split_into_passages(text: str, book_title: str) -> list[Passage]:
    """
    Split text into passages (sentences) with metadata.

    Attempts to detect chapter boundaries and maintain paragraph structure.
    """
    passages: list[Passage] = []

    # Detect chapters
    chapters = split_into_chapters(text)

    char_offset = 0
    global_sentence_num = 0

    for chapter_num, (chapter_title, chapter_text) in enumerate(chapters, start=1):
        # Split chapter into paragraphs
        paragraphs = split_into_paragraphs(chapter_text)

        for para_num, paragraph in enumerate(paragraphs, start=1):
            if is_structural_paragraph(paragraph):
                char_offset += len(paragraph) + 1
                continue

            # Split paragraph into sentences
            sentences = split_into_sentences(paragraph)

            for sent_num, sentence in enumerate(sentences, start=1):
                global_sentence_num += 1

                # Generate stable ID
                passage_id = generate_passage_id(
                    book_title, chapter_num, para_num, sent_num
                )

                passage = Passage(
                    id=passage_id,
                    text=sentence,
                    book=book_title,
                    chapter=chapter_title,
                    chapter_num=chapter_num,
                    paragraph_num=para_num,
                    sentence_num=sent_num,
                    char_offset=char_offset,
                )
                passages.append(passage)

                char_offset += len(sentence) + 1  # +1 for space/newline

    return passages


def split_into_chapters(text: str) -> list[tuple[str, str]]:
    """
    Split text into chapters.

    Returns list of (chapter_title, chapter_text) tuples.
    """
    text = normalize_text_structure(text)

    standalone_chapters = list(_STANDALONE_CHAPTER_RE.finditer(text))
    if standalone_chapters:
        return _split_on_standalone_chapters(text, standalone_chapters)

    # Common chapter patterns
    # NOTE: use [ \t]* (not \s*) before .* to prevent matching across newlines,
    # since re.MULTILINE makes ^ / $ line-anchored but \s also matches \n.
    chapter_patterns = [
        r"^(Chapter[ \t]+[IVXLC\d]+[:\.]?[ \t]*.*)$",  # Chapter I, Chapter 1, etc.
        r"^(CHAPTER[ \t]+[IVXLC\d]+[:\.]?[ \t]*.*)$",  # CHAPTER I
        r"^(\d+\.[ \t]+.+)$",  # 1. Title
        r"^(Part[ \t]+[IVXLC\d]+[:\.]?[ \t]*.*)$",  # Part I
    ]

    combined_pattern = "|".join(f"({p})" for p in chapter_patterns)

    # Find all chapter markers
    splits = list(re.finditer(combined_pattern, text, re.MULTILINE | re.IGNORECASE))

    if not splits:
        # No chapters detected, treat whole text as one chapter
        return [("Chapter 1", text)]

    chapters: list[tuple[str, str]] = []

    for i, match in enumerate(splits):
        title = match.group(0).strip()

        # Get text until next chapter (or end)
        start = match.end()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)

        chapter_text = text[start:end].strip()

        if chapter_text:  # Skip empty chapters
            chapters.append((title, chapter_text))

    # If there's content before the first chapter marker, include it
    if splits and splits[0].start() > 0:
        preamble = text[: splits[0].start()].strip()
        if preamble and len(preamble) > 100:  # Only if substantial
            chapters.insert(0, ("Prologue", preamble))

    return chapters if chapters else [("Chapter 1", text)]


def normalize_text_structure(text: str) -> str:
    """Trim obvious structural envelope noise from raw public-domain text dumps."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    standalone_chapters = list(_STANDALONE_CHAPTER_RE.finditer(normalized))
    if not standalone_chapters:
        return normalized

    start = standalone_chapters[0].start()
    prologue_matches = [
        match
        for match in _PROLOGUE_RE.finditer(normalized)
        if match.start() < start
    ]
    if prologue_matches:
        start = prologue_matches[-1].start()

    end = len(normalized)
    for pattern in (_APPENDIX_RE, _INDEX_RE):
        match = pattern.search(normalized, pos=start)
        if match:
            end = min(end, match.start())

    return normalized[start:end].strip()


def is_structural_paragraph(paragraph: str) -> bool:
    """Drop headings, TOC fragments, and other non-narrative structural paragraphs."""
    cleaned = " ".join(paragraph.split()).strip()
    if not cleaned:
        return True

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", cleaned)
    lowered = [token.lower() for token in tokens]
    if not lowered:
        return True

    if sum(token == "chapter" for token in lowered) >= 1:
        return True
    if sum(token in _STRUCTURAL_TOKENS for token in lowered) >= 2:
        return True
    if any(token.isdigit() for token in tokens) and any(
        token in _STRUCTURAL_TOKENS for token in lowered
    ):
        return True
    if cleaned.lower() == "prologue":
        return True
    if len(tokens) == 1 and tokens[0][:1].isupper() and not _has_sentence_ending(cleaned):
        return True
    if len(tokens) <= 8 and _looks_like_heading_line(cleaned):
        return True
    return False


def _split_on_standalone_chapters(text: str, matches: list[re.Match[str]]) -> list[tuple[str, str]]:
    chapters: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            title, body = _split_leading_section(preamble, fallback_title="Prologue")
            if body:
                chapters.append((title, body))

    for idx, match in enumerate(matches):
        title = _clean_heading(match.group(1))
        start = match.end()
        title_line, start = _consume_following_title_line(text, start)
        if title_line:
            title = f"{title} {title_line}".strip()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapter_text = text[start:end].strip()
        if chapter_text:
            chapters.append((title, chapter_text))

    return chapters if chapters else [("Chapter 1", text)]


def _split_leading_section(section_text: str, fallback_title: str) -> tuple[str, str]:
    lines = [line.strip() for line in section_text.splitlines() if line.strip()]
    if not lines:
        return fallback_title, ""

    first = _clean_heading(lines[0])
    if _looks_like_heading_line(first):
        body = "\n".join(lines[1:]).strip()
        return first.title() if first.isupper() else first, body

    return fallback_title, section_text.strip()


def _consume_following_title_line(text: str, start: int) -> tuple[str, int]:
    index = start
    length = len(text)
    while index < length and text[index] in {" ", "\t", "\n"}:
        index += 1

    line_end = text.find("\n", index)
    if line_end == -1:
        line_end = length

    line = _clean_heading(text[index:line_end])
    if not _looks_like_heading_line(line):
        return "", index

    return line, line_end


def _looks_like_heading_line(text: str) -> bool:
    cleaned = _clean_heading(text)
    if not cleaned or _has_sentence_ending(cleaned):
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'_-]*", cleaned)
    if not tokens or len(tokens) > 10:
        return False

    if any(token.lower() in _STRUCTURAL_TOKENS for token in tokens):
        return True

    titleish = 0
    for token in tokens:
        lowered = token.lower()
        if lowered in _LOWERCASE_TITLE_WORDS:
            titleish += 1
        elif token[:1].isupper() or token.isupper():
            titleish += 1
    return titleish / len(tokens) >= 0.8


def _has_sentence_ending(text: str) -> bool:
    return text.endswith((".", "!", "?"))


def _clean_heading(text: str) -> str:
    cleaned = text.strip().strip("_").strip()
    return re.sub(r"\s+", " ", cleaned)


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs."""
    # Split on double newlines or multiple newlines
    paragraphs = re.split(r"\n\s*\n+", text)

    # Clean up and filter empty
    paragraphs = [p.strip() for p in paragraphs]
    paragraphs = [p for p in paragraphs if p]

    return paragraphs


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences.

    Handles common abbreviations and edge cases.
    """
    # Normalize whitespace
    text = " ".join(text.split())

    # Abbreviations that don't end sentences
    abbreviations = {
        "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs", "etc",
        "i.e", "e.g", "cf", "al", "St", "Mt", "Ft",
    }

    # Protect abbreviations by replacing periods temporarily
    for abbr in abbreviations:
        text = re.sub(rf"\b{abbr}\.", f"{abbr}<<<DOT>>>", text, flags=re.IGNORECASE)

    # Split on sentence-ending punctuation
    # Look for . ! ? followed by space and capital letter (or end of string)
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z"])'
    sentences = re.split(sentence_pattern, text)

    # Restore protected periods
    sentences = [s.replace("<<<DOT>>>", ".") for s in sentences]

    # Clean up
    sentences = [s.strip() for s in sentences]
    sentences = [s for s in sentences if s]

    return sentences


def generate_passage_id(book: str, chapter: int, para: int, sent: int) -> str:
    """Generate a stable, readable passage ID."""
    # Slugify book title
    slug = re.sub(r"[^a-z0-9]+", "_", book.lower()).strip("_")

    return f"p_{slug}_c{chapter}_p{para}_s{sent}"
