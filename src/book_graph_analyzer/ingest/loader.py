"""Load books from various formats."""

from pathlib import Path

from bs4 import BeautifulSoup


def load_book(path: Path) -> str:
    """
    Load a book from file and return plain text.

    Supports:
    - .txt files (read directly)
    - .epub files (extract text from HTML)
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt(path)
    elif suffix == ".epub":
        return load_epub(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_txt(path: Path) -> str:
    """Load a plain text file."""
    # Try common encodings
    for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"Could not decode {path} with any common encoding")


def load_epub(path: Path) -> str:
    """Load an EPUB file and extract text."""
    import ebooklib
    from ebooklib import epub

    book = epub.read_epub(str(path))
    texts: list[str] = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Parse HTML content
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style"]):
                element.decompose()

            # Get text
            text = soup.get_text(separator="\n")

            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)

            if text:
                texts.append(text)

    return "\n\n".join(texts)
