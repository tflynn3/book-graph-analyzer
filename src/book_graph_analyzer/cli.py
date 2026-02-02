"""Command-line interface for Book Graph Analyzer."""

import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

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
        console.print("[green]âœ“[/green] Neo4j connected")
    else:
        console.print("[red]âœ—[/red] Neo4j not reachable")

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

    console.print(f"[green]âœ“[/green] Loaded {len(text):,} characters")

    # Split into passages
    with console.status("Splitting into passages..."):
        passages = split_into_passages(text, book_title)

    console.print(f"[green]âœ“[/green] Split into {len(passages):,} passages")

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


# ============================================================================
# Extract Commands
# ============================================================================

@main.group()
def extract() -> None:
    """Entity extraction commands."""
    pass


@extract.command(name="entities")
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", help="Book title (inferred from filename if not provided)")
@click.option("--no-llm", is_flag=True, help="Disable LLM-based extraction (faster, less accurate)")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
@click.option("--show-new", is_flag=True, help="Show suggested new entities not in seed database")
def extract_entities(path: str, title: str | None, no_llm: bool, output: str | None, show_new: bool) -> None:
    """Extract entities from a text file."""
    from book_graph_analyzer.extract import EntityExtractor

    file_path = Path(path)
    book_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()

    console.print(f"[bold]Extracting entities from:[/bold] {book_title}")
    console.print(f"[dim]Source: {file_path}[/dim]")
    console.print(f"[dim]LLM: {'disabled' if no_llm else 'enabled'}[/dim]\n")

    extractor = EntityExtractor(use_llm=not no_llm)

    # Show seed database stats
    console.print("[bold]Seed database:[/bold]")
    stats = extractor.resolver.stats
    console.print(f"  Characters: {stats['characters']:,}")
    console.print(f"  Places: {stats['places']:,}")
    console.print(f"  Objects: {stats['objects']:,}")
    console.print(f"  Total aliases: {stats['total_aliases']:,}\n")

    # Extract with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting entities...", total=None)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        results, extraction_stats = extractor.extract_from_file(
            file_path,
            book_title=book_title,
            progress_callback=update_progress,
        )

    # Display results
    console.print("\n[bold green]âœ“ Extraction complete![/bold green]\n")

    # Stats table
    table = Table(title="Extraction Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total passages", f"{extraction_stats.total_passages:,}")
    table.add_row("Entities extracted", f"{extraction_stats.total_entities_extracted:,}")
    table.add_row("Entities resolved", f"{extraction_stats.total_entities_resolved:,}")
    table.add_row("New entities found", f"{extraction_stats.new_entities_found:,}")

    console.print(table)

    # Entities by type
    console.print("\n[bold]Entities by type:[/bold]")
    for etype, count in sorted(extraction_stats.entities_by_type.items()):
        console.print(f"  {etype}: {count:,}")

    # Top characters
    if extraction_stats.top_characters:
        console.print("\n[bold]Top characters:[/bold]")
        for name, count in extraction_stats.top_characters[:10]:
            console.print(f"  {name}: {count:,} mentions")

    # Top places
    if extraction_stats.top_places:
        console.print("\n[bold]Top places:[/bold]")
        for name, count in extraction_stats.top_places[:10]:
            console.print(f"  {name}: {count:,} mentions")

    # New entity suggestions
    if show_new:
        suggestions = extractor.get_new_entity_suggestions(results, min_occurrences=3)
        if suggestions:
            console.print(f"\n[bold]Suggested new entities ({len(suggestions)}):[/bold]")
            for s in suggestions[:20]:
                console.print(f"  [{s['type']}] {s['text']}: {s['count']} occurrences")

    # Save output
    if output:
        output_path = Path(output)
        output_data = {
            "book": book_title,
            "stats": {
                "total_passages": extraction_stats.total_passages,
                "total_entities_extracted": extraction_stats.total_entities_extracted,
                "total_entities_resolved": extraction_stats.total_entities_resolved,
                "new_entities_found": extraction_stats.new_entities_found,
                "entities_by_type": dict(extraction_stats.entities_by_type),
            },
            "top_characters": extraction_stats.top_characters,
            "top_places": extraction_stats.top_places,
            "entities": [
                {
                    "passage": r.passage.text[:200],
                    "entities": [
                        {
                            "text": e.extracted.text,
                            "type": e.entity_type,
                            "canonical_id": e.canonical_id,
                            "canonical_name": e.canonical_name,
                            "is_new": e.is_new,
                        }
                        for e in r.entities
                    ],
                }
                for r in results[:100]  # Limit for file size
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\n[green]âœ“[/green] Results saved to {output_path}")


@extract.command(name="test")
@click.argument("text")
@click.option("--no-llm", is_flag=True, help="Disable LLM-based extraction")
def extract_test(text: str, no_llm: bool) -> None:
    """Test entity extraction on a single sentence."""
    from book_graph_analyzer.extract import EntityExtractor

    extractor = EntityExtractor(use_llm=not no_llm)

    console.print(f"[bold]Input:[/bold] {text}\n")

    # Extract
    results = extractor.extract_from_text(text)

    if results and results[0].entities:
        console.print("[bold]Extracted entities:[/bold]")
        for entity in results[0].entities:
            status = "[green]OK[/green]" if entity.canonical_id else "[yellow]??[/yellow]"
            canonical = f" -> {entity.canonical_name}" if entity.canonical_name else ""
            console.print(
                f"  {status} [{entity.entity_type}] \"{entity.extracted.text}\"{canonical}"
            )
    else:
        console.print("[yellow]No entities found[/yellow]")


@extract.command(name="seeds")
def extract_seeds() -> None:
    """Show seed database statistics."""
    from book_graph_analyzer.extract import EntityResolver

    resolver = EntityResolver()
    stats = resolver.stats

    console.print("[bold]Seed Database Statistics[/bold]\n")

    table = Table()
    table.add_column("Entity Type", style="cyan")
    table.add_column("Count", style="green", justify="right")

    table.add_row("Characters", str(stats["characters"]))
    table.add_row("Places", str(stats["places"]))
    table.add_row("Objects", str(stats["objects"]))
    table.add_row("Total Aliases", str(stats["total_aliases"]))

    console.print(table)

    # Sample some entries
    console.print("\n[bold]Sample characters:[/bold]")
    for char in list(resolver.db.characters.values())[:5]:
        aliases = ", ".join(char.aliases[:3])
        if len(char.aliases) > 3:
            aliases += f", +{len(char.aliases) - 3} more"
        console.print(f"  {char.canonical_name}")
        console.print(f"    [dim]Aliases: {aliases}[/dim]")


@extract.command(name="relationships")
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", help="Book title")
@click.option("--no-llm", is_flag=True, help="Disable LLM-based extraction")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
@click.option("--limit", "-l", type=int, help="Limit passages to process")
def extract_relationships_cmd(
    path: str, title: str | None, no_llm: bool, output: str | None, limit: int | None
) -> None:
    """Extract relationships from a text file."""
    from collections import defaultdict

    from book_graph_analyzer.extract import EntityExtractor, RelationshipExtractor

    file_path = Path(path)
    book_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()

    console.print(f"[bold]Extracting relationships from:[/bold] {book_title}")
    console.print(f"[dim]Source: {file_path}[/dim]")
    console.print(f"[dim]LLM: {'disabled' if no_llm else 'enabled'}[/dim]\n")

    # First, extract entities
    console.print("[bold]Step 1: Entity Extraction[/bold]")
    entity_extractor = EntityExtractor(use_llm=not no_llm)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting entities...", total=None)

        def entity_progress(current: int, total: int) -> None:
            progress.update(task, completed=current, total=total)

        entity_results, entity_stats = entity_extractor.extract_from_file(
            file_path,
            book_title=book_title,
            progress_callback=entity_progress,
        )

    console.print(f"  Found {entity_stats.total_entities_resolved:,} resolved entities\n")

    # Limit if requested
    if limit:
        entity_results = entity_results[:limit]

    # Now extract relationships
    console.print("[bold]Step 2: Relationship Extraction[/bold]")
    rel_extractor = RelationshipExtractor(
        resolver=entity_extractor.resolver,
        use_llm=not no_llm,
    )

    relationship_results = []
    rel_counts: dict[str, int] = defaultdict(int)
    total_relationships = 0

    total_to_process = len(entity_results)
    for i, rel_result in enumerate(rel_extractor.extract_from_results(entity_results)):
        relationship_results.append(rel_result)
        for rel in rel_result.relationships:
            rel_counts[rel.predicate.value] += 1
            total_relationships += 1
        
        # Simple progress indicator every 100 passages
        if (i + 1) % 100 == 0 or i + 1 == total_to_process:
            console.print(f"  Processed {i + 1}/{total_to_process} passages, found {total_relationships} relationships")

    # Display results
    console.print(f"\n[bold green]OK Extraction complete![/bold green]\n")

    table = Table(title="Relationship Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Passages with 2+ entities", str(len(relationship_results)))
    table.add_row("Total relationships found", str(total_relationships))
    table.add_row("Unique relationship types", str(len(rel_counts)))

    console.print(table)

    # Relationship types breakdown
    console.print("\n[bold]Relationships by type:[/bold]")
    for rel_type, count in sorted(rel_counts.items(), key=lambda x: -x[1])[:15]:
        console.print(f"  {rel_type}: {count:,}")

    # Sample relationships - only show ones with resolved entities
    console.print("\n[bold]Sample relationships (resolved entities only):[/bold]")
    sample_count = 0
    for result in relationship_results:
        for rel in result.relationships:
            if sample_count >= 10:
                break
            # Only show relationships where both entities are resolved
            if rel.subject_id and rel.object_id:
                console.print(f"  ({rel.subject_id})-[{rel.predicate.value}]->({rel.object_id})")
                sample_count += 1
        if sample_count >= 10:
            break
    
    if sample_count == 0:
        console.print("  [dim]No fully resolved relationships to display[/dim]")

    # Save output
    if output:
        output_path = Path(output)
        output_data = {
            "book": book_title,
            "stats": {
                "passages_processed": len(relationship_results),
                "total_relationships": total_relationships,
                "relationship_counts": dict(rel_counts),
            },
            "relationships": [
                {
                    "passage_id": r.passage_id,
                    "passage_text": r.passage_text[:200],
                    "relationships": [
                        {
                            "subject": rel.subject_text,
                            "subject_id": rel.subject_id,
                            "predicate": rel.predicate.value,
                            "object": rel.object_text,
                            "object_id": rel.object_id,
                            "confidence": rel.confidence,
                            "method": rel.extraction_method,
                        }
                        for rel in r.relationships
                    ],
                }
                for r in relationship_results
                if r.relationships
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\n[green]OK[/green] Results saved to {output_path}")


@extract.command(name="rel-test")
@click.argument("text")
@click.option("--no-llm", is_flag=True, help="Disable LLM-based extraction")
def extract_rel_test(text: str, no_llm: bool) -> None:
    """Test relationship extraction on a single sentence."""
    from book_graph_analyzer.extract import EntityExtractor, RelationshipExtractor

    extractor = EntityExtractor(use_llm=not no_llm)
    rel_extractor = RelationshipExtractor(
        resolver=extractor.resolver,
        use_llm=not no_llm,
    )

    console.print(f"[bold]Input:[/bold] {text}\n")

    # Extract entities first
    results = extractor.extract_from_text(text)

    if not results or not results[0].entities:
        console.print("[yellow]No entities found[/yellow]")
        return

    console.print("[bold]Entities found:[/bold]")
    for entity in results[0].entities:
        canonical = f" -> {entity.canonical_name}" if entity.canonical_name else ""
        console.print(f"  [{entity.entity_type}] {entity.extracted.text}{canonical}")

    # Extract relationships
    rel_result = rel_extractor.extract_relationships(
        text=text,
        passage_id="test",
        entities=results[0].entities,
    )

    if rel_result.relationships:
        console.print("\n[bold]Relationships found:[/bold]")
        for rel in rel_result.relationships:
            subj = rel.subject_id or rel.subject_text
            obj = rel.object_id or rel.object_text
            console.print(f"  ({subj})-[{rel.predicate.value}]->({obj})")
            console.print(f"    [dim]method: {rel.extraction_method}, confidence: {rel.confidence}[/dim]")
    else:
        console.print("\n[yellow]No relationships found[/yellow]")


if __name__ == "__main__":
    main()

