"""Split text into structured passages."""

import re
from dataclasses import dataclass


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
    # Common chapter patterns
    chapter_patterns = [
        r"^(Chapter\s+[IVXLC\d]+[:\.]?\s*.*)$",  # Chapter I, Chapter 1, etc.
        r"^(CHAPTER\s+[IVXLC\d]+[:\.]?\s*.*)$",  # CHAPTER I
        r"^(\d+\.\s+.+)$",  # 1. Title
        r"^(Part\s+[IVXLC\d]+[:\.]?\s*.*)$",  # Part I
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
