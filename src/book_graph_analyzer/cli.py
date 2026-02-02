"""Command-line interface for Book Graph Analyzer."""

import click
from rich.console import Console

from book_graph_analyzer import __version__

console = Console()


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    """Book Graph Analyzer - Transform novels into queryable knowledge graphs."""
    pass


@main.command()
def status() -> None:
    """Check system status (Neo4j connection, models, etc.)."""
    from book_graph_analyzer.config import get_settings
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    console.print("[bold]Book Graph Analyzer Status[/bold]\n")

    # Check Neo4j
    settings = get_settings()
    console.print(f"Neo4j URI: {settings.neo4j_uri}")

    if check_neo4j_connection():
        console.print("[green]✓[/green] Neo4j connected")
    else:
        console.print("[red]✗[/red] Neo4j not reachable")

    # TODO: Check for local LLM (Ollama)
    # TODO: Check for spaCy models


@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", help="Book title (inferred from filename if not provided)")
def ingest(path: str, title: str | None) -> None:
    """Ingest a text file or EPUB into the system."""
    from pathlib import Path

    from book_graph_analyzer.ingest.loader import load_book
    from book_graph_analyzer.ingest.splitter import split_into_passages

    file_path = Path(path)
    book_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()

    console.print(f"[bold]Ingesting:[/bold] {book_title}")
    console.print(f"[dim]Source: {file_path}[/dim]\n")

    # Load the book
    with console.status("Loading book..."):
        text = load_book(file_path)

    console.print(f"[green]✓[/green] Loaded {len(text):,} characters")

    # Split into passages
    with console.status("Splitting into passages..."):
        passages = split_into_passages(text, book_title)

    console.print(f"[green]✓[/green] Split into {len(passages):,} passages")

    # Preview
    console.print("\n[bold]Sample passages:[/bold]")
    for p in passages[:3]:
        console.print(f"  [dim]{p.book} / Ch.{p.chapter_num} / P{p.paragraph_num} / S{p.sentence_num}[/dim]")
        console.print(f"  {p.text[:100]}{'...' if len(p.text) > 100 else ''}\n")

    # TODO: Write to database


@main.command()
@click.argument("query")
def search(query: str) -> None:
    """Search passages by text content."""
    console.print(f"[bold]Searching:[/bold] {query}")
    console.print("[yellow]Not yet implemented - need to ingest data first[/yellow]")


@main.group()
def graph() -> None:
    """Graph database commands."""
    pass


@graph.command(name="stats")
def graph_stats() -> None:
    """Show graph statistics."""
    from book_graph_analyzer.graph.connection import get_driver

    driver = get_driver()
    if not driver:
        console.print("[red]Cannot connect to Neo4j[/red]")
        return

    with driver.session() as session:
        # Count nodes by type
        result = session.run("""
            MATCH (n)
            RETURN labels(n)[0] as label, count(*) as count
            ORDER BY count DESC
        """)

        console.print("[bold]Node counts:[/bold]")
        total = 0
        for record in result:
            label = record["label"] or "Unlabeled"
            count = record["count"]
            total += count
            console.print(f"  {label}: {count:,}")

        console.print(f"\n[bold]Total nodes:[/bold] {total:,}")

        # Count relationships
        result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
        rel_count = result.single()["count"]
        console.print(f"[bold]Total relationships:[/bold] {rel_count:,}")

    driver.close()


if __name__ == "__main__":
    main()
