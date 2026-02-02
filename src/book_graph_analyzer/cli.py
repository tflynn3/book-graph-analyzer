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


# ============================================================================
# Analyze Command (Generic Zero-Seed Extraction)
# ============================================================================

@main.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", help="Book title")
@click.option("--no-llm", is_flag=True, help="Disable LLM-based extraction")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
def analyze(path: str, title: str | None, no_llm: bool, output: str | None) -> None:
    """Analyze a book with zero-seed generic extraction.
    
    This command extracts entities and relationships without requiring
    a pre-seeded entity database. Works on any novel.
    """
    from book_graph_analyzer.extract import GenericExtractor

    file_path = Path(path)
    book_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()

    console.print(f"[bold]Analyzing:[/bold] {book_title}")
    console.print(f"[dim]Source: {file_path}[/dim]")
    console.print(f"[dim]Mode: Zero-seed generic extraction[/dim]")
    console.print(f"[dim]LLM: {'disabled' if no_llm else 'enabled'}[/dim]\n")

    extractor = GenericExtractor(use_llm=not no_llm)

    # Progress tracking
    current_phase = ""
    def progress_callback(phase: str, current: int, total: int, message: str) -> None:
        nonlocal current_phase
        if phase != current_phase:
            current_phase = phase
            console.print(f"\n[bold]Phase: {phase.title()}[/bold]")
        if current > 0 and (current % 500 == 0 or current == total):
            console.print(f"  {message} ({current}/{total})")

    # Run analysis
    analysis = extractor.analyze_book(
        file_path=file_path,
        title=book_title,
        progress_callback=progress_callback,
    )

    # Display results
    console.print(f"\n[bold green]OK Analysis complete![/bold green]\n")

    # Entity stats
    table = Table(title="Entity Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total passages", f"{analysis.total_passages:,}")
    table.add_row("Unique entities", f"{len(analysis.entity_clusters):,}")
    table.add_row("Total mentions", f"{analysis.total_mentions:,}")
    table.add_row("Total relationships", f"{len(analysis.relationships):,}")

    console.print(table)

    # Entities by type
    by_type = {}
    for cluster in analysis.entity_clusters.values():
        by_type[cluster.entity_type] = by_type.get(cluster.entity_type, 0) + 1

    console.print("\n[bold]Entities by type:[/bold]")
    for etype, count in sorted(by_type.items(), key=lambda x: -x[1]):
        console.print(f"  {etype}: {count:,}")

    # Top entities
    console.print("\n[bold]Top entities (by mentions):[/bold]")
    top_entities = sorted(
        analysis.entity_clusters.values(),
        key=lambda c: c.mention_count,
        reverse=True,
    )[:15]
    
    for cluster in top_entities:
        aliases_str = ""
        if cluster.aliases:
            # Filter and encode aliases for safe printing
            alias_list = [a.encode('ascii', 'replace').decode('ascii') for a in list(cluster.aliases)[:3]]
            aliases_str = f" (aliases: {', '.join(alias_list)})"
        name = cluster.canonical_name.encode('ascii', 'replace').decode('ascii')
        console.print(f"  [{cluster.entity_type}] {name}: {cluster.mention_count} mentions{aliases_str}")

    # Relationship stats
    if analysis.relationships:
        console.print("\n[bold]Relationships by type:[/bold]")
        for rel_type, count in sorted(analysis.relationship_counts.items(), key=lambda x: -x[1])[:10]:
            console.print(f"  {rel_type}: {count:,}")

        # Sample relationships
        console.print("\n[bold]Sample relationships:[/bold]")
        sample_count = 0
        for rel in analysis.relationships:
            if rel.subject_id and rel.object_id and sample_count < 10:
                console.print(f"  ({rel.subject_id})-[{rel.predicate.value}]->({rel.object_id})")
                sample_count += 1

    # Save output
    if output:
        output_path = Path(output)
        
        # Export to JSON
        output_data = {
            "title": analysis.title,
            "stats": {
                "total_passages": analysis.total_passages,
                "unique_entities": len(analysis.entity_clusters),
                "total_mentions": analysis.total_mentions,
                "total_relationships": len(analysis.relationships),
                "entities_by_type": by_type,
                "relationship_counts": analysis.relationship_counts,
            },
            "entities": [
                {
                    "id": c.id,
                    "canonical_name": c.canonical_name,
                    "type": c.entity_type,
                    "mentions": c.mention_count,
                    "aliases": list(c.aliases),
                }
                for c in sorted(analysis.entity_clusters.values(), key=lambda x: -x.mention_count)
            ],
            "relationships": [
                {
                    "subject_id": r.subject_id,
                    "predicate": r.predicate.value,
                    "object_id": r.object_id,
                    "passage_id": r.passage_id,
                }
                for r in analysis.relationships
                if r.subject_id and r.object_id
            ],
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)

        console.print(f"\n[green]OK[/green] Results saved to {output_path}")

        # Also export seed file for future use
        seed_path = output_path.with_suffix(".seeds.json")
        seed_data = {
            "characters": [
                {"id": c.id, "canonical_name": c.canonical_name, "aliases": list(c.aliases)}
                for c in analysis.entity_clusters.values()
                if c.entity_type == "character" and c.mention_count >= 3
            ],
            "places": [
                {"id": c.id, "canonical_name": c.canonical_name, "aliases": list(c.aliases)}
                for c in analysis.entity_clusters.values()
                if c.entity_type == "place" and c.mention_count >= 2
            ],
            "objects": [
                {"id": c.id, "canonical_name": c.canonical_name, "aliases": list(c.aliases)}
                for c in analysis.entity_clusters.values()
                if c.entity_type == "object" and c.mention_count >= 2
            ],
        }
        
        with open(seed_path, "w") as f:
            json.dump(seed_data, f, indent=2)
        
        console.print(f"[green]OK[/green] Seed file saved to {seed_path} (for future re-analysis)")


# ============================================================================
# Style Analysis Commands (Phase 4)
# ============================================================================

@main.group()
def style() -> None:
    """Style analysis commands - extract author fingerprints."""
    pass


@style.command(name="analyze")
@click.argument("path", type=click.Path(exists=True))
@click.option("--author", "-a", default="Unknown", help="Author name")
@click.option("--output", "-o", type=click.Path(), help="Output file for fingerprint (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def style_analyze(path: str, author: str, output: str | None, verbose: bool) -> None:
    """Analyze a text file and extract its style fingerprint.
    
    Example:
        bga style analyze data/texts/the_hobbit.txt -a "J.R.R. Tolkien" -o hobbit_style.json
    """
    from book_graph_analyzer.style import StyleAnalyzer

    file_path = Path(path)
    
    console.print(f"[bold]Style Analysis:[/bold] {file_path.name}")
    console.print(f"[dim]Author: {author}[/dim]\n")

    # Progress callback
    def progress_callback(progress):
        if verbose:
            console.print(f"  [{progress.phase}] {progress.message}")

    analyzer = StyleAnalyzer(progress_callback=progress_callback if verbose else None)
    
    with console.status("Analyzing style..."):
        fingerprint = analyzer.analyze_file(file_path, author_name=author)
    
    # Display summary
    console.print(fingerprint.summary())
    
    # Save if output specified
    if output:
        output_path = Path(output)
        analyzer.save_fingerprint(fingerprint, output_path)
        console.print(f"\n[green]OK[/green] Fingerprint saved to {output_path}")


@style.command(name="compare")
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option("--json-input", "-j", is_flag=True, help="Input files are JSON fingerprints (not text)")
@click.option("--author1", "-a1", default="Author 1", help="Name for first author")
@click.option("--author2", "-a2", default="Author 2", help="Name for second author")
def style_compare(file1: str, file2: str, json_input: bool, author1: str, author2: str) -> None:
    """Compare style fingerprints of two texts or fingerprint files.
    
    Examples:
        bga style compare book1.txt book2.txt -a1 "Tolkien" -a2 "Lewis"
        bga style compare tolkien.json lewis.json -j
    """
    from book_graph_analyzer.style import StyleAnalyzer

    analyzer = StyleAnalyzer()
    
    if json_input:
        # Load pre-computed fingerprints
        fp1 = analyzer.load_fingerprint(file1)
        fp2 = analyzer.load_fingerprint(file2)
    else:
        # Analyze texts
        console.print("[bold]Analyzing first text...[/bold]")
        with console.status(f"Analyzing {Path(file1).name}..."):
            fp1 = analyzer.analyze_file(file1, author_name=author1)
        
        console.print("[bold]Analyzing second text...[/bold]")
        with console.status(f"Analyzing {Path(file2).name}..."):
            fp2 = analyzer.analyze_file(file2, author_name=author2)
    
    # Compare
    comparison = analyzer.compare(fp1, fp2)
    
    # Display results
    console.print(f"\n[bold]Style Comparison: {comparison['author1']} vs {comparison['author2']}[/bold]\n")
    
    table = Table(title="Comparison Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Burrows' Delta", f"{comparison['burrows_delta']:.3f}")
    table.add_row("Similarity Score", f"{comparison['similarity_score']:.2%}")
    table.add_row("Interpretation", comparison['interpretation'])
    
    console.print(table)
    
    # Details
    details = comparison['details']
    console.print("\n[bold]Details:[/bold]")
    console.print(f"  Sentence length: {details['sentence_length']['author1_mean']:.1f} vs {details['sentence_length']['author2_mean']:.1f} words")
    console.print(f"  Flesch-Kincaid: {details['readability']['author1_fk_grade']:.1f} vs {details['readability']['author2_fk_grade']:.1f} grade level")
    console.print(f"  Dialogue ratio: {details['dialogue_ratio']['author1']*100:.1f}% vs {details['dialogue_ratio']['author2']*100:.1f}%")


@style.command(name="batch")
@click.argument("directory", type=click.Path(exists=True))
@click.option("--author", "-a", default="Unknown", help="Author name")
@click.option("--pattern", "-p", default="*.txt", help="File pattern to match")
@click.option("--output", "-o", type=click.Path(), help="Output file for combined fingerprint (JSON)")
def style_batch(directory: str, author: str, pattern: str, output: str | None) -> None:
    """Analyze multiple files and create a combined fingerprint.
    
    Example:
        bga style batch data/texts/lotr-corpus/ -a "Tolkien" -p "*.txt" -o tolkien_combined.json
    """
    from book_graph_analyzer.style import StyleAnalyzer
    import glob
    
    dir_path = Path(directory)
    files = list(dir_path.glob(pattern))
    
    if not files:
        console.print(f"[red]No files matching '{pattern}' found in {directory}[/red]")
        return
    
    console.print(f"[bold]Batch Style Analysis[/bold]")
    console.print(f"[dim]Author: {author}[/dim]")
    console.print(f"[dim]Files: {len(files)}[/dim]\n")
    
    for f in files[:10]:
        console.print(f"  - {f.name}")
    if len(files) > 10:
        console.print(f"  ... and {len(files) - 10} more")
    
    console.print()
    
    analyzer = StyleAnalyzer()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing files...", total=len(files))
        
        # Custom progress callback
        def progress_callback(p):
            progress.update(task, description=p.message)
        
        analyzer.progress_callback = progress_callback
        
        fingerprint = analyzer.analyze_files(files, author_name=author)
        progress.update(task, completed=len(files))
    
    # Display summary
    console.print(fingerprint.summary())
    
    # Save if output specified
    if output:
        output_path = Path(output)
        analyzer.save_fingerprint(fingerprint, output_path)
        console.print(f"\n[green]OK[/green] Combined fingerprint saved to {output_path}")


@style.command(name="report")
@click.argument("fingerprint_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file for report (Markdown)")
def style_report(fingerprint_path: str, output: str | None) -> None:
    """Generate a detailed style report from a fingerprint file.
    
    Example:
        bga style report tolkien_style.json -o tolkien_report.md
    """
    from book_graph_analyzer.style import StyleAnalyzer
    
    analyzer = StyleAnalyzer()
    fingerprint = analyzer.load_fingerprint(fingerprint_path)
    
    # Generate markdown report
    report = _generate_style_report(fingerprint)
    
    if output:
        output_path = Path(output)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        console.print(f"[green]OK[/green] Report saved to {output_path}")
    else:
        console.print(report)


def _generate_style_report(fingerprint) -> str:
    """Generate a markdown style report from a fingerprint."""
    lines = [
        f"# Style Analysis Report: {fingerprint.author_name}",
        "",
        "## Overview",
        "",
        f"- **Total Words Analyzed**: {fingerprint.total_word_count:,}",
        f"- **Total Sentences**: {fingerprint.total_sentence_count:,}",
        f"- **Source Texts**: {', '.join(fingerprint.source_texts)}",
        "",
        "## Sentence Structure",
        "",
    ]
    
    if fingerprint.sentence_length_dist:
        sl = fingerprint.sentence_length_dist
        lines.extend([
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Mean sentence length | {sl.mean:.1f} words |",
            f"| Median | {sl.median:.1f} words |",
            f"| Range | {sl.min:.0f} - {sl.max:.0f} words |",
            f"| Std deviation | {sl.std:.1f} |",
            "",
        ])
    
    lines.extend([
        "## Style Characteristics",
        "",
        f"| Characteristic | Percentage |",
        f"|----------------|------------|",
        f"| Dialogue passages | {fingerprint.dialogue_ratio*100:.1f}% |",
        f"| Passive voice | {fingerprint.passive_voice_ratio*100:.1f}% |",
        f"| Questions | {fingerprint.question_ratio*100:.1f}% |",
        f"| Exclamations | {fingerprint.exclamation_ratio*100:.1f}% |",
        "",
        "## Readability",
        "",
        f"| Metric | Score | Interpretation |",
        f"|--------|-------|----------------|",
        f"| Flesch Reading Ease | {fingerprint.flesch_reading_ease:.1f} | {_interpret_flesch(fingerprint.flesch_reading_ease)} |",
        f"| Flesch-Kincaid Grade | {fingerprint.flesch_kincaid_grade:.1f} | Grade {int(fingerprint.flesch_kincaid_grade)} reading level |",
        f"| Gunning Fog | {fingerprint.gunning_fog:.1f} | {int(fingerprint.gunning_fog)} years of education |",
        "",
    ])
    
    if fingerprint.vocabulary_profile:
        vp = fingerprint.vocabulary_profile
        lines.extend([
            "## Vocabulary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Unique words | {vp.unique_words:,} |",
            f"| Type-token ratio | {vp.type_token_ratio:.3f} |",
            f"| Average word length | {vp.avg_word_length:.2f} chars |",
            f"| Hapax legomena | {vp.hapax_count:,} ({vp.hapax_ratio*100:.1f}%) |",
            "",
        ])
        
        if vp.archaisms_found:
            lines.extend([
                "### Archaic Language",
                "",
                f"Archaisms found: {', '.join(vp.archaisms_found)}",
                "",
                f"Density: {fingerprint.archaism_density:.2f} per 1000 words",
                "",
            ])
    
    if fingerprint.passage_type_distribution:
        lines.extend([
            "## Passage Types",
            "",
            f"| Type | Percentage |",
            f"|------|------------|",
        ])
        for ptype, ratio in sorted(fingerprint.passage_type_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"| {ptype.title()} | {ratio*100:.1f}% |")
        lines.append("")
    
    return "\n".join(lines)


def _interpret_flesch(score: float) -> str:
    """Interpret Flesch Reading Ease score."""
    if score >= 90:
        return "Very easy (5th grade)"
    elif score >= 80:
        return "Easy (6th grade)"
    elif score >= 70:
        return "Fairly easy (7th grade)"
    elif score >= 60:
        return "Standard (8th-9th grade)"
    elif score >= 50:
        return "Fairly difficult (10th-12th grade)"
    elif score >= 30:
        return "Difficult (college level)"
    else:
        return "Very difficult (college graduate)"


# ============================================================================
# Voice Analysis Commands (Phase 5)
# ============================================================================

@main.group()
def voice() -> None:
    """Character voice analysis - extract how each character speaks."""
    pass


@voice.command(name="analyze")
@click.argument("path", type=click.Path(exists=True))
@click.option("--min-lines", "-m", default=3, help="Minimum lines for profile")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed output")
def voice_analyze(path: str, min_lines: int, output: str | None, verbose: bool) -> None:
    """Extract character voice profiles from a text.
    
    Example:
        bga voice analyze data/texts/the_hobbit.txt -o hobbit_voices.json
    """
    from book_graph_analyzer.voice import VoiceAnalyzer

    file_path = Path(path)
    
    console.print(f"[bold]Voice Analysis:[/bold] {file_path.name}")
    console.print(f"[dim]Min lines for profile: {min_lines}[/dim]\n")

    def progress_callback(message):
        if verbose:
            console.print(f"  {message}")

    analyzer = VoiceAnalyzer(
        min_lines_for_profile=min_lines,
        progress_callback=progress_callback if verbose else None,
    )
    
    with console.status("Analyzing character voices..."):
        result = analyzer.analyze_file(file_path)
    
    # Display results
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Total dialogue lines: {result.total_dialogue_lines:,}")
    console.print(f"  Attribution rate: {result.attribution_rate*100:.1f}%")
    console.print(f"  Characters with profiles: {result.total_characters}")
    
    # Top speakers
    console.print(f"\n[bold]Top Speakers:[/bold]")
    table = Table()
    table.add_column("Character", style="cyan")
    table.add_column("Lines", justify="right")
    table.add_column("Avg Length", justify="right")
    table.add_column("Questions", justify="right")
    
    for speaker, line_count in result.top_speakers(15):
        profile = result.profiles.get(speaker)
        if profile:
            table.add_row(
                speaker,
                str(line_count),
                f"{profile.avg_utterance_length:.1f}",
                f"{profile.question_ratio*100:.0f}%"
            )
        else:
            table.add_row(speaker, str(line_count), "-", "-")
    
    console.print(table)
    
    # Save if output specified
    if output:
        output_path = Path(output)
        analyzer.save_results(result, output_path)
        console.print(f"\n[green]OK[/green] Results saved to {output_path}")


@voice.command(name="profile")
@click.argument("results_path", type=click.Path(exists=True))
@click.argument("character")
def voice_profile(results_path: str, character: str) -> None:
    """Show detailed voice profile for a character.
    
    Example:
        bga voice profile hobbit_voices.json Gandalf
    """
    from book_graph_analyzer.voice import VoiceAnalyzer

    analyzer = VoiceAnalyzer()
    result = analyzer.load_results(results_path)
    
    profile = result.get_profile(character)
    
    if not profile:
        # Try fuzzy match
        available = list(result.profiles.keys())
        console.print(f"[red]Character '{character}' not found.[/red]")
        console.print(f"\nAvailable characters:")
        for name in sorted(available):
            console.print(f"  - {name}")
        return
    
    console.print(profile.summary())


@voice.command(name="compare")
@click.argument("results_path", type=click.Path(exists=True))
@click.argument("char1")
@click.argument("char2")
def voice_compare(results_path: str, char1: str, char2: str) -> None:
    """Compare voice profiles of two characters.
    
    Example:
        bga voice compare hobbit_voices.json Gandalf Bilbo
    """
    from book_graph_analyzer.voice import VoiceAnalyzer

    analyzer = VoiceAnalyzer()
    result = analyzer.load_results(results_path)
    
    profile1 = result.get_profile(char1)
    profile2 = result.get_profile(char2)
    
    if not profile1:
        console.print(f"[red]Character '{char1}' not found.[/red]")
        return
    if not profile2:
        console.print(f"[red]Character '{char2}' not found.[/red]")
        return
    
    comparison = analyzer.compare_voices(profile1, profile2)
    
    console.print(f"\n[bold]Voice Comparison: {char1} vs {char2}[/bold]\n")
    
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column(char1, justify="right")
    table.add_column(char2, justify="right")
    table.add_column("Difference", justify="right")
    
    for metric, values in comparison["metrics"].items():
        metric_display = metric.replace("_", " ").title()
        if "ratio" in metric:
            table.add_row(
                metric_display,
                f"{values['char1']*100:.1f}%",
                f"{values['char2']*100:.1f}%",
                f"{values['difference']*100:.1f}%"
            )
        else:
            table.add_row(
                metric_display,
                f"{values['char1']:.1f}",
                f"{values['char2']:.1f}",
                f"{values['difference']:.1f}"
            )
    
    console.print(table)
    console.print(f"\n[bold]Similarity Score:[/bold] {comparison['similarity_score']:.2f}")
    
    if comparison.get("shared_distinctive_words"):
        console.print(f"\n[bold]Shared Distinctive Words:[/bold]")
        console.print(f"  {', '.join(comparison['shared_distinctive_words'])}")


@voice.command(name="quotes")
@click.argument("results_path", type=click.Path(exists=True))
@click.argument("character")
@click.option("--limit", "-n", default=10, help="Number of quotes to show")
def voice_quotes(results_path: str, character: str, limit: int) -> None:
    """Show sample quotes from a character.
    
    Example:
        bga voice quotes hobbit_voices.json Gandalf -n 5
    """
    from book_graph_analyzer.voice import VoiceAnalyzer

    analyzer = VoiceAnalyzer()
    result = analyzer.load_results(results_path)
    
    profile = result.get_profile(character)
    
    if not profile:
        console.print(f"[red]Character '{character}' not found.[/red]")
        return
    
    console.print(f"\n[bold]Quotes from {character}:[/bold]\n")
    
    # Get dialogue lines for this character
    char_lines = result.dialogue_by_speaker.get(character, [])
    
    if not char_lines:
        # Fall back to sample quotes in profile
        for quote in profile.sample_quotes[:limit]:
            console.print(f'  "{quote}"')
    else:
        # Show actual lines
        shown = 0
        for line in char_lines:
            if shown >= limit:
                break
            text = line.text if hasattr(line, 'text') else str(line)
            console.print(f'  "{text}"')
            shown += 1
    
    console.print(f"\n[dim]Total lines: {profile.total_lines}[/dim]")


# ============================================================================
# Pipeline Commands - Unified Analysis
# ============================================================================

@main.group()
def pipeline() -> None:
    """Unified analysis pipelines - run multiple phases together."""
    pass


@pipeline.command(name="full")
@click.argument("path", type=click.Path(exists=True))
@click.option("--title", "-t", help="Book title")
@click.option("--author", "-a", default="Unknown", help="Author name")
@click.option("--no-neo4j", is_flag=True, help="Skip writing to Neo4j")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory for JSON files")
def pipeline_full(path: str, title: str | None, author: str, no_neo4j: bool, output_dir: str | None) -> None:
    """Run full analysis pipeline: entities, style, and voice.
    
    Processes a book through all analysis phases and writes results
    to Neo4j (if available) and JSON files.
    
    Example:
        bga pipeline full data/texts/the_hobbit.txt -t "The Hobbit" -a "Tolkien" -o output/
    """
    from book_graph_analyzer.ingest.loader import load_book
    from book_graph_analyzer.ingest.splitter import split_into_passages
    from book_graph_analyzer.extract import EntityExtractor, RelationshipExtractor
    from book_graph_analyzer.style import StyleAnalyzer
    from book_graph_analyzer.voice import VoiceAnalyzer
    from book_graph_analyzer.graph.writer import GraphWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    file_path = Path(path)
    book_title = title or file_path.stem.replace("_", " ").replace("-", " ").title()
    book_id = book_title.lower().replace(" ", "_").replace("'", "")

    console.print(f"[bold]Full Analysis Pipeline[/bold]")
    console.print(f"  Book: {book_title}")
    console.print(f"  Author: {author}")
    console.print(f"  Source: {file_path}")
    
    # Check Neo4j
    neo4j_available = not no_neo4j and check_neo4j_connection()
    if not no_neo4j and not neo4j_available:
        console.print(f"  [yellow]Neo4j not available - will save to JSON only[/yellow]")
    elif neo4j_available:
        console.print(f"  [green]Neo4j connected[/green]")
    
    console.print()

    # Setup output directory
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = Path("data/output") / book_id
        out_path.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Phase 1: Load and Split
    # =========================================================================
    console.print("[bold]Phase 1: Loading text...[/bold]")
    
    with console.status("Loading book..."):
        text = load_book(file_path)
    console.print(f"  Loaded {len(text):,} characters")

    with console.status("Splitting into passages..."):
        passages = split_into_passages(text, book_title)
    console.print(f"  Split into {len(passages):,} passages")

    # =========================================================================
    # Phase 2-3: Entity & Relationship Extraction
    # =========================================================================
    console.print("\n[bold]Phase 2-3: Entity & Relationship Extraction...[/bold]")
    
    extractor = EntityExtractor(use_llm=False)  # Fast mode
    rel_extractor = RelationshipExtractor(resolver=extractor.resolver, use_llm=False)

    entity_results = []
    relationship_results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting entities...", total=len(passages))
        
        for passage in passages:
            results = extractor.extract_from_passage(passage)
            if results:
                entity_results.append(results)
                
                # Extract relationships
                rel_result = rel_extractor.extract_relationships(
                    text=passage.text,
                    passage_id=passage.id,
                    entities=results.entities,
                )
                if rel_result.relationships:
                    relationship_results.append(rel_result)
            
            progress.update(task, advance=1)

    # Count unique entities
    entity_ids = set()
    for result in entity_results:
        for entity in result.entities:
            if entity.canonical_id:
                entity_ids.add(entity.canonical_id)
    
    total_rels = sum(len(r.relationships) for r in relationship_results)
    console.print(f"  Unique entities: {len(entity_ids)}")
    console.print(f"  Relationships: {total_rels}")

    # =========================================================================
    # Phase 4: Style Analysis
    # =========================================================================
    console.print("\n[bold]Phase 4: Style Analysis...[/bold]")
    
    style_analyzer = StyleAnalyzer()
    
    with console.status("Analyzing style..."):
        fingerprint = style_analyzer.analyze_text(text, author_name=author, source_name=file_path.name)
    
    console.print(f"  Avg sentence length: {fingerprint.sentence_length_dist.mean:.1f} words")
    console.print(f"  Flesch-Kincaid Grade: {fingerprint.flesch_kincaid_grade:.1f}")
    console.print(f"  Dialogue ratio: {fingerprint.dialogue_ratio*100:.1f}%")

    # Save style fingerprint
    style_path = out_path / "style_fingerprint.json"
    style_analyzer.save_fingerprint(fingerprint, style_path)

    # =========================================================================
    # Phase 5: Voice Analysis
    # =========================================================================
    console.print("\n[bold]Phase 5: Voice Analysis...[/bold]")
    
    voice_analyzer = VoiceAnalyzer(min_lines_for_profile=3)
    
    with console.status("Analyzing character voices..."):
        voice_result = voice_analyzer.analyze_text(text, source_name=file_path.name)
    
    console.print(f"  Dialogue lines: {voice_result.total_dialogue_lines}")
    console.print(f"  Attribution rate: {voice_result.attribution_rate*100:.1f}%")
    console.print(f"  Character profiles: {voice_result.total_characters}")

    # Save voice analysis
    voice_path = out_path / "voice_profiles.json"
    voice_analyzer.save_results(voice_result, voice_path)

    # =========================================================================
    # Write to Neo4j
    # =========================================================================
    if neo4j_available:
        console.print("\n[bold]Writing to Neo4j...[/bold]")
        
        writer = GraphWriter()
        
        with console.status("Writing book style..."):
            writer.write_book_style(book_id, book_title, author, fingerprint)
        console.print("  Book style written")

        with console.status("Writing entities and relationships..."):
            stats = writer.write_extraction_results(
                entity_results=entity_results,
                relationship_results=relationship_results,
                book=book_title,
            )
        console.print(f"  Entities: {stats['entities_written']}")
        console.print(f"  Relationships: {stats['relationships_written']}")

        # Build entity ID map for voice profiles
        entity_map = {}
        for result in entity_results:
            for entity in result.entities:
                if entity.canonical_id and entity.canonical_name:
                    entity_map[entity.canonical_name] = entity.canonical_id
                    # Also map extracted text
                    entity_map[entity.extracted.text] = entity.canonical_id

        with console.status("Writing voice profiles..."):
            voice_stats = writer.write_voice_analysis_results(
                voice_result=voice_result,
                book_id=book_id,
                entity_id_map=entity_map,
            )
        console.print(f"  Voice profiles: {voice_stats['profiles_written']}")

        writer.close()

    # =========================================================================
    # Summary
    # =========================================================================
    console.print("\n[bold green]Pipeline Complete![/bold green]")
    console.print(f"\nOutput saved to: {out_path}")
    console.print(f"  - style_fingerprint.json")
    console.print(f"  - voice_profiles.json")
    
    if neo4j_available:
        console.print(f"\nNeo4j populated with:")
        console.print(f"  - Book node with style metrics")
        console.print(f"  - {len(entity_ids)} entity nodes")
        console.print(f"  - {total_rels} relationships")
        console.print(f"  - {voice_result.total_characters} character voice profiles")


# ============================================================================
# Corpus Commands (Phase 6)
# ============================================================================

@main.group()
def corpus() -> None:
    """Manage multi-book corpus analysis."""
    pass


@corpus.command(name="create")
@click.argument("name")
@click.option("--author", "-a", required=True, help="Author name")
def corpus_create(name: str, author: str) -> None:
    """Create a new corpus for an author's works.
    
    Example:
        bga corpus create tolkien_works -a "J.R.R. Tolkien"
    """
    from book_graph_analyzer.corpus import CorpusManager
    
    manager = CorpusManager(name, author=author)
    console.print(f"[green]Created corpus:[/green] {name}")
    console.print(f"  Author: {author}")
    console.print(f"  Data dir: {manager.data_dir}")


@corpus.command(name="add")
@click.argument("corpus_name")
@click.argument("title")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--series", "-s", help="Series name")
@click.option("--order", "-n", type=int, help="Order in series")
def corpus_add(corpus_name: str, title: str, file_path: str, series: str | None, order: int | None) -> None:
    """Add a book to a corpus.
    
    Example:
        bga corpus add tolkien_works "The Hobbit" data/texts/the_hobbit.txt
        bga corpus add tolkien_works "Fellowship" data/texts/fellowship.txt -s "LOTR" -n 1
    """
    from book_graph_analyzer.corpus import CorpusManager
    
    manager = CorpusManager(corpus_name)
    book = manager.add_book(title, file_path, series=series, series_order=order)
    
    console.print(f"[green]Added to corpus:[/green] {title}")
    console.print(f"  ID: {book.id}")
    console.print(f"  File: {file_path}")
    if series:
        console.print(f"  Series: {series} #{order or '?'}")


@corpus.command(name="list")
@click.argument("corpus_name")
def corpus_list(corpus_name: str) -> None:
    """List books in a corpus.
    
    Example:
        bga corpus list tolkien_works
    """
    from book_graph_analyzer.corpus import CorpusManager
    
    manager = CorpusManager(corpus_name)
    console.print(manager.corpus_summary())


@corpus.command(name="process")
@click.argument("corpus_name")
@click.option("--book", "-b", help="Process specific book ID only")
@click.option("--skip-processed", is_flag=True, help="Skip already processed books")
def corpus_process(corpus_name: str, book: str | None, skip_processed: bool) -> None:
    """Process all books in a corpus with cross-book entity resolution.
    
    Example:
        bga corpus process tolkien_works
        bga corpus process tolkien_works -b the_hobbit
    """
    from book_graph_analyzer.corpus import CorpusManager, CrossBookResolver
    from book_graph_analyzer.ingest.loader import load_book
    from book_graph_analyzer.ingest.splitter import split_into_passages
    from book_graph_analyzer.extract import EntityExtractor, RelationshipExtractor
    from book_graph_analyzer.style import StyleAnalyzer
    from book_graph_analyzer.voice import VoiceAnalyzer
    from book_graph_analyzer.graph.writer import GraphWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection
    
    manager = CorpusManager(corpus_name)
    resolver = CrossBookResolver(corpus_name)
    
    # Determine which books to process
    if book:
        books_to_process = [manager.get_book(book)]
        if not books_to_process[0]:
            console.print(f"[red]Book not found: {book}[/red]")
            return
    elif skip_processed:
        books_to_process = manager.get_unprocessed_books()
    else:
        books_to_process = manager.list_books()
    
    if not books_to_process:
        console.print("[yellow]No books to process[/yellow]")
        return
    
    console.print(f"[bold]Processing {len(books_to_process)} book(s) in corpus: {corpus_name}[/bold]\n")
    
    neo4j_ok = check_neo4j_connection()
    if neo4j_ok:
        writer = GraphWriter()
    
    for book_info in books_to_process:
        console.print(f"\n[bold cyan]>>> {book_info.title}[/bold cyan]")
        
        # Load text
        text = load_book(Path(book_info.file_path))
        passages = split_into_passages(text, book_info.title)
        console.print(f"  Loaded {len(text):,} chars, {len(passages):,} passages")
        
        # Entity extraction with cross-book resolution
        extractor = EntityExtractor(use_llm=False)
        rel_extractor = RelationshipExtractor(resolver=extractor.resolver, use_llm=False)
        
        entity_results = []
        relationship_results = []
        entity_ids = set()
        
        with console.status("Extracting entities..."):
            for passage in passages:
                results = extractor.extract_from_passage(passage)
                if results:
                    entity_results.append(results)
                    
                    # Resolve to corpus-wide canonical IDs
                    for entity in results.entities:
                        if entity.canonical_name:
                            canonical_id = resolver.resolve(
                                entity.canonical_name,
                                entity.entity_type,
                                book_info.id,
                            )
                            entity.canonical_id = canonical_id
                            entity_ids.add(canonical_id)
                    
                    rel_result = rel_extractor.extract_relationships(
                        text=passage.text,
                        passage_id=passage.id,
                        entities=results.entities,
                    )
                    if rel_result.relationships:
                        relationship_results.append(rel_result)
        
        total_rels = sum(len(r.relationships) for r in relationship_results)
        console.print(f"  Entities: {len(entity_ids)}, Relationships: {total_rels}")
        
        # Style analysis
        style_analyzer = StyleAnalyzer()
        fingerprint = style_analyzer.analyze_text(text, author_name=manager.corpus.author)
        console.print(f"  Style: {fingerprint.sentence_length_dist.mean:.1f} avg words/sent, FK grade {fingerprint.flesch_kincaid_grade:.1f}")
        
        # Voice analysis
        voice_analyzer = VoiceAnalyzer(min_lines_for_profile=3)
        voice_result = voice_analyzer.analyze_text(text)
        console.print(f"  Voice: {voice_result.total_dialogue_lines} lines, {voice_result.total_characters} profiles")
        
        # Update book stats
        manager.update_book_stats(
            book_id=book_info.id,
            total_words=fingerprint.total_word_count,
            total_passages=len(passages),
            entity_count=len(entity_ids),
            relationship_count=total_rels,
            dialogue_lines=voice_result.total_dialogue_lines,
            character_profiles=voice_result.total_characters,
            avg_sentence_length=fingerprint.sentence_length_dist.mean if fingerprint.sentence_length_dist else 0,
            flesch_kincaid_grade=fingerprint.flesch_kincaid_grade,
        )
        
        # Write to Neo4j
        if neo4j_ok:
            writer.write_book_style(book_info.id, book_info.title, manager.corpus.author, fingerprint)
            writer.write_extraction_results(entity_results, relationship_results, book_info.title)
        
        console.print(f"  [green]OK[/green] Processed")
    
    # Save cross-book resolver state
    resolver.save()
    
    if neo4j_ok:
        writer.close()
    
    console.print(f"\n[bold green]Corpus processing complete![/bold green]")
    console.print(resolver.summary())


@corpus.command(name="entities")
@click.argument("corpus_name")
@click.option("--cross-book", "-x", is_flag=True, help="Show only cross-book entities")
@click.option("--type", "-t", "entity_type", help="Filter by type (character, place, object)")
def corpus_entities(corpus_name: str, cross_book: bool, entity_type: str | None) -> None:
    """Show entities resolved across books.
    
    Example:
        bga corpus entities tolkien_works -x
        bga corpus entities tolkien_works -t character
    """
    from book_graph_analyzer.corpus import CrossBookResolver
    
    resolver = CrossBookResolver(corpus_name)
    
    if cross_book:
        entities = resolver.get_cross_book_entities()
        console.print(f"[bold]Cross-Book Entities ({len(entities)})[/bold]\n")
    elif entity_type:
        entities = resolver.get_entities_by_type(entity_type)
        console.print(f"[bold]{entity_type.title()} Entities ({len(entities)})[/bold]\n")
    else:
        entities = list(resolver.entities.values())
        console.print(f"[bold]All Entities ({len(entities)})[/bold]\n")
    
    # Sort by total appearances
    entities.sort(key=lambda e: -sum(e.appearances.values()))
    
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Books")
    table.add_column("Mentions", justify="right")
    
    for entity in entities[:30]:
        books = ", ".join(entity.appearances.keys())
        mentions = sum(entity.appearances.values())
        table.add_row(entity.canonical_name, entity.entity_type, books, str(mentions))
    
    console.print(table)
    
    if len(entities) > 30:
        console.print(f"\n[dim]...and {len(entities) - 30} more[/dim]")


@corpus.command(name="compare")
@click.argument("corpus_name")
def corpus_compare(corpus_name: str) -> None:
    """Compare style metrics across all books in corpus.
    
    Example:
        bga corpus compare tolkien_works
    """
    from book_graph_analyzer.corpus import CorpusManager
    
    manager = CorpusManager(corpus_name)
    processed = manager.get_processed_books()
    
    if not processed:
        console.print("[yellow]No processed books in corpus[/yellow]")
        return
    
    console.print(f"[bold]Style Comparison: {corpus_name}[/bold]\n")
    
    table = Table()
    table.add_column("Book", style="cyan")
    table.add_column("Words", justify="right")
    table.add_column("Avg Sent", justify="right")
    table.add_column("FK Grade", justify="right")
    table.add_column("Entities", justify="right")
    table.add_column("Dialogue", justify="right")
    
    for book in sorted(processed, key=lambda b: b.series_order or 999):
        table.add_row(
            book.title[:25],
            f"{book.total_words:,}",
            f"{book.avg_sentence_length:.1f}",
            f"{book.flesch_kincaid_grade:.1f}",
            str(book.entity_count),
            str(book.dialogue_lines),
        )
    
    console.print(table)


# ============================================================================
# World Bible Commands (Phase 7)
# ============================================================================

@main.group()
def worldbible() -> None:
    """World bible extraction - rules and patterns of fictional worlds."""
    pass


@worldbible.command(name="extract")
@click.argument("path", type=click.Path(exists=True))
@click.option("--world", "-w", required=True, help="World name (e.g., 'Middle-earth')")
@click.option("--use-llm", is_flag=True, help="Use LLM for synthesis (requires Ollama)")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
def worldbible_extract(path: str, world: str, use_llm: bool, output: str | None) -> None:
    """Extract world bible from a text file.
    
    Example:
        bga worldbible extract the_hobbit.txt -w "Middle-earth" -o hobbit_bible.json
    """
    from book_graph_analyzer.worldbible import WorldBibleExtractor, ExtractionConfig
    
    file_path = Path(path)
    
    console.print(f"[bold]World Bible Extraction[/bold]")
    console.print(f"  World: {world}")
    console.print(f"  Source: {file_path.name}")
    console.print(f"  Mode: {'LLM-assisted' if use_llm else 'Keyword-based'}")
    console.print()
    
    config = ExtractionConfig(use_llm=use_llm)
    
    def progress_callback(msg):
        console.print(f"  {msg}")
    
    extractor = WorldBibleExtractor(config=config, progress_callback=progress_callback)
    
    bible = extractor.extract_from_file(file_path, world)
    
    console.print(f"\n{bible.summary()}")
    
    if output:
        output_path = Path(output)
        extractor.save_bible(bible, output_path)
        console.print(f"\n[green]OK[/green] Saved to {output_path}")


@worldbible.command(name="show")
@click.argument("bible_path", type=click.Path(exists=True))
@click.option("--category", "-c", help="Filter by category")
def worldbible_show(bible_path: str, category: str | None) -> None:
    """Show world bible contents.
    
    Example:
        bga worldbible show hobbit_bible.json
        bga worldbible show hobbit_bible.json -c magic
    """
    from book_graph_analyzer.worldbible import WorldBibleExtractor, WorldBibleCategory
    
    extractor = WorldBibleExtractor()
    bible = extractor.load_bible(bible_path)
    
    console.print(f"[bold]=== World Bible: {bible.name} ===[/bold]\n")
    
    if category:
        try:
            cat = WorldBibleCategory(category.lower())
            rules = bible.get_rules(cat)
            console.print(f"[bold]{cat.value.title()} ({len(rules)} rules)[/bold]\n")
            
            for rule in rules:
                console.print(f"[cyan]{rule.title}[/cyan]")
                console.print(f"  {rule.description[:200]}{'...' if len(rule.description) > 200 else ''}")
                console.print(f"  [dim]Sources: {len(rule.source_passages)} passages | Confidence: {rule.confidence:.0%}[/dim]")
                console.print()
        except ValueError:
            console.print(f"[red]Unknown category: {category}[/red]")
            console.print(f"Valid: {', '.join(c.value for c in WorldBibleCategory)}")
    else:
        # Show all categories
        for cat in WorldBibleCategory:
            rules = bible.get_rules(cat)
            if rules:
                console.print(f"[bold]{cat.value.title()} ({len(rules)} rules)[/bold]")
                for rule in rules[:3]:
                    console.print(f"  - {rule.title}")
                if len(rules) > 3:
                    console.print(f"  [dim]...and {len(rules) - 3} more[/dim]")
                console.print()


@worldbible.command(name="cultures")
@click.argument("bible_path", type=click.Path(exists=True))
@click.option("--culture", "-c", help="Show specific culture")
def worldbible_cultures(bible_path: str, culture: str | None) -> None:
    """Show cultural profiles from world bible.
    
    Example:
        bga worldbible cultures hobbit_bible.json
        bga worldbible cultures hobbit_bible.json -c Elves
    """
    from book_graph_analyzer.worldbible import WorldBibleExtractor
    
    extractor = WorldBibleExtractor()
    bible = extractor.load_bible(bible_path)
    
    if not bible.cultures:
        console.print("[yellow]No cultural profiles found[/yellow]")
        return
    
    if culture:
        # Find matching culture
        profile = None
        for c in bible.cultures.values():
            if c.name.lower() == culture.lower():
                profile = c
                break
        
        if not profile:
            console.print(f"[red]Culture not found: {culture}[/red]")
            console.print(f"Available: {', '.join(c.name for c in bible.cultures.values())}")
            return
        
        console.print(f"[bold]=== {profile.name} ===[/bold]\n")
        
        if profile.values:
            console.print(f"[cyan]Values:[/cyan] {', '.join(profile.values)}")
        if profile.customs:
            console.print(f"[cyan]Customs:[/cyan] {', '.join(profile.customs)}")
        if profile.homeland:
            console.print(f"[cyan]Homeland:[/cyan] {profile.homeland}")
        if profile.lifespan:
            console.print(f"[cyan]Lifespan:[/cyan] {profile.lifespan}")
        
        console.print(f"\n[dim]Based on {len(profile.source_passages)} passages[/dim]")
    else:
        console.print(f"[bold]Cultural Profiles ({len(bible.cultures)})[/bold]\n")
        
        for profile in bible.cultures.values():
            console.print(f"[cyan]{profile.name}[/cyan]")
            console.print(f"  Passages: {len(profile.source_passages)}")


@worldbible.command(name="query")
@click.argument("bible_path", type=click.Path(exists=True))
@click.argument("query")
def worldbible_query(bible_path: str, query: str) -> None:
    """Search world bible for relevant rules.
    
    Example:
        bga worldbible query hobbit_bible.json "ring"
        bga worldbible query hobbit_bible.json "dragon"
    """
    from book_graph_analyzer.worldbible import WorldBibleExtractor
    
    extractor = WorldBibleExtractor()
    bible = extractor.load_bible(bible_path)
    
    query_lower = query.lower()
    matches = []
    
    # Search rules
    for rules in bible.rules.values():
        for rule in rules:
            if (query_lower in rule.title.lower() or 
                query_lower in rule.description.lower()):
                matches.append(('rule', rule))
    
    # Search cultures
    for culture in bible.cultures.values():
        if query_lower in culture.name.lower():
            matches.append(('culture', culture))
    
    console.print(f"[bold]Results for '{query}' ({len(matches)} matches)[/bold]\n")
    
    for match_type, item in matches:
        if match_type == 'rule':
            console.print(f"[cyan][Rule][/cyan] {item.title}")
            console.print(f"  {item.description[:150]}...")
            console.print(f"  [dim]Category: {item.category.value}[/dim]")
        else:
            console.print(f"[cyan][Culture][/cyan] {item.name}")
        console.print()


if __name__ == "__main__":
    main()

