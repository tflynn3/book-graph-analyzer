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


if __name__ == "__main__":
    main()

