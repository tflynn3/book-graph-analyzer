"""Tests for text splitting."""

from book_graph_analyzer.ingest.splitter import (
    is_structural_paragraph,
    split_into_chapters,
    split_into_paragraphs,
    split_into_passages,
    split_into_sentences,
)


class TestSentenceSplitting:
    """Test sentence boundary detection."""

    def test_simple_sentences(self):
        text = "This is sentence one. This is sentence two. And a third!"
        sentences = split_into_sentences(text)
        assert len(sentences) == 3
        assert sentences[0] == "This is sentence one."
        assert sentences[1] == "This is sentence two."
        assert sentences[2] == "And a third!"

    def test_abbreviations(self):
        text = "Mr. Baggins went to see Dr. Gandalf. They talked for hours."
        sentences = split_into_sentences(text)
        assert len(sentences) == 2
        assert "Mr. Baggins" in sentences[0]
        assert "Dr. Gandalf" in sentences[0]

    def test_dialogue(self):
        text = '"Hello," said Frodo. "Where are you going?" asked Sam.'
        sentences = split_into_sentences(text)
        assert len(sentences) == 2

    def test_question_and_exclamation(self):
        text = "What is this? It is the Ring! We must destroy it."
        sentences = split_into_sentences(text)
        assert len(sentences) == 3


class TestParagraphSplitting:
    """Test paragraph boundary detection."""

    def test_double_newline(self):
        text = "First paragraph.\n\nSecond paragraph."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 2

    def test_multiple_newlines(self):
        text = "First.\n\n\n\nSecond."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 2

    def test_empty_paragraphs_filtered(self):
        text = "First.\n\n   \n\nSecond."
        paragraphs = split_into_paragraphs(text)
        assert len(paragraphs) == 2


class TestChapterSplitting:
    """Test chapter detection."""

    def test_chapter_roman(self):
        text = "Prologue text.\n\nChapter I\n\nFirst chapter.\n\nChapter II\n\nSecond chapter."
        chapters = split_into_chapters(text)
        assert len(chapters) >= 2

    def test_chapter_arabic(self):
        text = "Chapter 1\n\nContent one.\n\nChapter 2\n\nContent two."
        chapters = split_into_chapters(text)
        assert len(chapters) == 2

    def test_no_chapters(self):
        text = "Just some text without any chapter markers. It goes on and on."
        chapters = split_into_chapters(text)
        assert len(chapters) == 1
        assert chapters[0][0] == "Chapter 1"


class TestFullSplitting:
    """Test the complete splitting pipeline."""

    def test_basic_passage_creation(self):
        text = "Chapter 1\n\nFirst paragraph here. Second sentence.\n\nAnother paragraph."
        passages = split_into_passages(text, "Test Book")

        assert len(passages) >= 3
        assert passages[0].book == "Test Book"
        assert passages[0].chapter_num == 1
        assert passages[0].id.startswith("p_test_book_")

    def test_passage_ordering(self):
        text = "Chapter 1\n\nA. B. C.\n\nD. E."
        passages = split_into_passages(text, "Test")

        # Should be in order
        for i in range(1, len(passages)):
            prev = passages[i - 1]
            curr = passages[i]
            # Either same paragraph with increasing sentence, or new paragraph
            assert (
                curr.paragraph_num > prev.paragraph_num
                or (
                    curr.paragraph_num == prev.paragraph_num
                    and curr.sentence_num > prev.sentence_num
                )
            )

    def test_splitter_prefers_body_chapter_markers_over_toc_entries(self):
        text = """
CONTENTS
Chapter 1 A Long-expected Party
Chapter 2 The Shadow of the Past

FOREWORD
Editorial notes that should not be in the graph.

PROLOGUE
Hobbits are an unobtrusive but very ancient people.

_Chapter 1_
A Long-expected Party

Mr. Bilbo never did a kinder deed.

_Chapter 2_
The Shadow of the Past

Gandalf spoke quietly.

APPENDICES
A genealogical note.
"""
        passages = split_into_passages(text, "Test Book")
        passage_texts = [passage.text for passage in passages]

        assert "Mr. Bilbo never did a kinder deed." in passage_texts
        assert "Gandalf spoke quietly." in passage_texts
        assert "Hobbits are an unobtrusive but very ancient people." in passage_texts
        assert all("Editorial notes" not in text for text in passage_texts)
        assert all("A Long-expected Party" not in text for text in passage_texts)
        assert all("The Shadow of the Past" not in text for text in passage_texts)
        assert all("genealogical note" not in text for text in passage_texts)

    def test_structural_paragraph_filter_rejects_short_heading_lines(self):
        assert is_structural_paragraph("A Long-expected Party")
        assert is_structural_paragraph("CONTENTS")
        assert is_structural_paragraph("Christopher")
        assert not is_structural_paragraph("Bilbo laughed.")
