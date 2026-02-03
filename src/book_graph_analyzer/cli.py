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
@click.option("--no-llm", is_flag=True, help="Disable LLM-based extraction")
def corpus_process(corpus_name: str, book: str | None, skip_processed: bool, no_llm: bool) -> None:
    """Process all books in a corpus with cross-book entity resolution.
    
    Example:
        bga corpus process tolkien_works
        bga corpus process tolkien_works -b the_hobbit
    """
    from book_graph_analyzer.corpus import CorpusManager, CrossBookResolver
    from book_graph_analyzer.extract.dynamic_resolver import DynamicEntityResolver
    from book_graph_analyzer.extract.ner import NERPipeline
    from book_graph_analyzer.ingest.loader import load_book
    from book_graph_analyzer.ingest.splitter import split_into_passages
    from book_graph_analyzer.style import StyleAnalyzer
    from book_graph_analyzer.voice import VoiceAnalyzer
    from book_graph_analyzer.graph.writer import GraphWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection
    
    manager = CorpusManager(corpus_name)
    cross_resolver = CrossBookResolver(corpus_name, use_llm=not no_llm)
    
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
    
    # Initialize NER pipeline once
    ner_pipeline = NERPipeline(use_llm=not no_llm)
    
    for book_info in books_to_process:
        console.print(f"\n[bold cyan]>>> {book_info.title}[/bold cyan]")
        
        # Load text
        text = load_book(Path(book_info.file_path))
        passages = split_into_passages(text, book_info.title)
        console.print(f"  Loaded {len(text):,} chars, {len(passages):,} passages")
        
        # Entity extraction using DynamicEntityResolver (per-book)
        dynamic_resolver = DynamicEntityResolver(use_llm=not no_llm)
        
        entity_ids = set()
        
        with console.status("Extracting entities..."):
            for passage in passages:
                ner_entities = ner_pipeline.extract_entities(passage.text)
                for entity in ner_entities:
                    cluster = dynamic_resolver.process_mention(
                        entity=entity,
                        passage_id=passage.id,
                        passage_text=passage.text,
                    )
                    entity_ids.add(cluster.id)
        
        # Consolidate within-book aliases
        merge_count = dynamic_resolver.consolidate_clusters()
        console.print(f"  Extracted {len(dynamic_resolver.clusters)} unique entities ({merge_count} alias merges)")
        
        # Register book's entities with cross-book resolver
        cross_resolver.register_book_entities(book_info.id, dynamic_resolver.clusters)
        
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
            entity_count=len(dynamic_resolver.clusters),
            relationship_count=0,  # TODO: Add relationship extraction
            dialogue_lines=voice_result.total_dialogue_lines,
            character_profiles=voice_result.total_characters,
            avg_sentence_length=fingerprint.sentence_length_dist.mean if fingerprint.sentence_length_dist else 0,
            flesch_kincaid_grade=fingerprint.flesch_kincaid_grade,
        )
        
        # Write to Neo4j
        if neo4j_ok:
            writer.write_book_style(book_info.id, book_info.title, manager.corpus.author, fingerprint)
        
        console.print(f"  [green]OK[/green] Processed")
    
    # Resolve cross-book entities
    console.print("\n[bold]Resolving cross-book entities...[/bold]")
    resolution_stats = cross_resolver.resolve_all()
    console.print(f"  New entities: {resolution_stats['new_entities']}")
    console.print(f"  Merged across books: {resolution_stats['merged_entities']}")
    
    if neo4j_ok:
        writer.close()
    
    console.print(f"\n[bold green]Corpus processing complete![/bold green]")
    console.print(cross_resolver.summary())


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
        entities = resolver.get_multi_book_entities()
        console.print(f"[bold]Cross-Book Entities ({len(entities)})[/bold]\n")
    elif entity_type:
        entities = resolver.get_entities_by_type(entity_type)
        console.print(f"[bold]{entity_type.title()} Entities ({len(entities)})[/bold]\n")
    else:
        entities = list(resolver.entities.values())
        console.print(f"[bold]All Entities ({len(entities)})[/bold]\n")
    
    # Sort by total mentions
    entities.sort(key=lambda e: -e.total_mentions)
    
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Type")
    table.add_column("Books")
    table.add_column("Mentions", justify="right")
    
    for entity in entities[:30]:
        books = ", ".join(entity.book_clusters.keys())
        table.add_row(entity.canonical_name, entity.entity_type, books, str(entity.total_mentions))
    
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


@corpus.command(name="events")
@click.argument("corpus_name")
@click.option("--output", "-o", type=click.Path(), help="Output file (JSON)")
@click.option("--neo4j", is_flag=True, help="Also write to Neo4j")
@click.option("--chunk-size", default=3000, help="Characters per chunk (default: 3000)")
@click.option("--skip-processed", is_flag=True, help="Skip books already in events file")
def corpus_events(corpus_name: str, output: str | None, neo4j: bool, chunk_size: int, skip_processed: bool) -> None:
    """Extract events from all books in corpus with cross-book linking.
    
    Creates a unified event graph with temporal ordering across books.
    Events are linked to entities from the corpus entity resolver.
    
    Examples:
        bga corpus events tolkien_works -o tolkien_events.json
        bga corpus events tolkien_works --neo4j
    """
    from book_graph_analyzer.corpus import CorpusManager
    from book_graph_analyzer.lore import EventExtractor, EventGraph, Event, EventRelation
    from book_graph_analyzer.ingest.loader import load_book
    
    manager = CorpusManager(corpus_name)
    books = manager.get_processed_books()
    
    if not books:
        all_books = manager.list_books()
        if all_books:
            books = all_books
        else:
            console.print("[yellow]No books in corpus. Add books with 'bga corpus add'[/yellow]")
            return
    
    # Output path
    output_path = Path(output) if output else Path(f"data/output/{corpus_name}_events.json")
    
    # Load existing events if skip_processed
    existing_books = set()
    unified_graph = EventGraph()
    if skip_processed and output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            unified_graph = EventGraph.from_dict(data)
            # Track which books we've already processed
            for event in unified_graph.events.values():
                if event.source_book:
                    existing_books.add(event.source_book.lower())
        console.print(f"[dim]Loaded {len(unified_graph.events)} existing events[/dim]")
    
    # Progress tracking
    total_books = len([b for b in books if b.title.lower() not in existing_books])
    processed_books = 0
    
    console.print(f"[bold]Extracting events from {corpus_name}[/bold]")
    console.print(f"Books to process: {total_books}")
    
    for book in books:
        if book.title.lower() in existing_books:
            console.print(f"[dim]Skipping {book.title} (already processed)[/dim]")
            continue
        
        processed_books += 1
        console.print(f"\n[bold][{processed_books}/{total_books}] {book.title}[/bold]")
        
        # Load book text
        try:
            text = load_book(Path(book.file_path))
        except Exception as e:
            console.print(f"[red]Error loading {book.file_path}: {e}[/red]")
            continue
        
        console.print(f"[dim]Loaded {len(text):,} characters[/dim]")
        
        # Progress for this book
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting events...", total=100)
            
            def update_progress(current, total, message):
                progress.update(task, completed=int(current / total * 100), description=message)
            
            extractor = EventExtractor(use_llm=True, progress_callback=update_progress)
            
            # Extract events
            if len(text) > chunk_size * 2:
                book_graph = extractor.extract_from_book(text, source_book=book.title, chunk_size=chunk_size)
            else:
                book_graph = extractor.extract_from_text(text, source_book=book.title)
        
        console.print(f"  Events: {len(book_graph.events)}")
        console.print(f"  Relations: {len(book_graph.relations)}")
        
        # Merge into unified graph
        for event in book_graph.events.values():
            # Check for duplicates across books
            event_key = f"{event.agent}|{event.action}|{event.patient}".lower()
            duplicate = False
            for existing in unified_graph.events.values():
                existing_key = f"{existing.agent}|{existing.action}|{existing.patient}".lower()
                if event_key == existing_key:
                    duplicate = True
                    break
            
            if not duplicate:
                # Prefix ID with book name to avoid collisions
                book_prefix = book.title.lower().replace(" ", "_")[:10]
                event.id = f"{book_prefix}_{event.id}"
                unified_graph.add_event(event)
        
        # Add relations (update IDs)
        for rel in book_graph.relations:
            book_prefix = book.title.lower().replace(" ", "_")[:10]
            unified_graph.add_relation(EventRelation(
                event1_id=f"{book_prefix}_{rel.event1_id}",
                relation=rel.relation,
                event2_id=f"{book_prefix}_{rel.event2_id}",
                confidence=rel.confidence,
            ))
    
    # Infer cross-book ordering from era/year
    console.print("\n[bold]Inferring cross-book temporal ordering...[/bold]")
    cross_book_relations = 0
    events_with_era = [e for e in unified_graph.events.values() if e.era]
    
    for i, e1 in enumerate(events_with_era):
        for e2 in events_with_era[i+1:]:
            if e1.source_book != e2.source_book:  # Cross-book
                if e1.era and e2.era and e1.era != e2.era:
                    if e1.era.order < e2.era.order:
                        unified_graph.add_relation(EventRelation(
                            event1_id=e1.id,
                            relation="before",
                            event2_id=e2.id,
                            confidence=0.95,
                        ))
                        cross_book_relations += 1
                    elif e1.era.order > e2.era.order:
                        unified_graph.add_relation(EventRelation(
                            event1_id=e2.id,
                            relation="before",
                            event2_id=e1.id,
                            confidence=0.95,
                        ))
                        cross_book_relations += 1
    
    console.print(f"  Cross-book relations added: {cross_book_relations}")
    
    # Summary
    console.print(f"\n[bold]Unified Event Graph:[/bold]")
    console.print(f"  Total events: {len(unified_graph.events)}")
    console.print(f"  Total relations: {len(unified_graph.relations)}")
    
    # Events by book
    by_book: dict[str, int] = {}
    for event in unified_graph.events.values():
        book = event.source_book or "Unknown"
        by_book[book] = by_book.get(book, 0) + 1
    
    console.print(f"\n[bold]Events by book:[/bold]")
    for book, count in sorted(by_book.items()):
        console.print(f"  {book}: {count}")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(unified_graph.to_dict(), f, indent=2)
    
    console.print(f"\n[green]OK[/green] Events saved to {output_path}")
    
    # Write to Neo4j if requested
    if neo4j:
        from book_graph_analyzer.graph.writer import GraphWriter
        from book_graph_analyzer.graph.connection import check_neo4j_connection
        
        if not check_neo4j_connection():
            console.print("[red]Error:[/red] Cannot connect to Neo4j")
            return
        
        console.print("\n[bold]Writing to Neo4j...[/bold]")
        
        writer = GraphWriter()
        stats = writer.write_event_graph(
            unified_graph,
            book=corpus_name,
            link_entities=True,
        )
        writer.close()
        
        console.print(f"  Events written: {stats['events_written']}")
        console.print(f"  Relations written: {stats['relations_written']}")
        console.print(f"  Entity links created: {stats['entity_links']}")
        console.print(f"[green]OK[/green] Events written to Neo4j")


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


# ============================================================================
# Lore Checking Commands
# ============================================================================

@main.group()
def lore() -> None:
    """Lore consistency checking - validate facts against extracted knowledge."""
    pass


@lore.command(name="check")
@click.argument("claim")
@click.option("--bible", "-b", type=click.Path(exists=True), help="World bible file")
@click.option("--corpus", "-c", help="Corpus name for entity lookup")
@click.option("--timeline", "-t", type=click.Path(exists=True), help="Timeline file")
@click.option("--events", "-e", type=click.Path(exists=True), help="Events file for temporal ordering")
@click.option("--neo4j", is_flag=True, help="Query Neo4j for relationships")
def lore_check(claim: str, bible: str | None, corpus: str | None, timeline: str | None, events: str | None, neo4j: bool) -> None:
    """Check a single claim against world knowledge.
    
    Examples:
        bga lore check "Gandalf is a wizard" -b hobbit_bible.json
        bga lore check "Hobbits have beards" -b hobbit_bible.json
        bga lore check "Turin lived in the Second Age" -t timeline.json
        bga lore check "Bilbo found the ring before Gollum" -e events.json
        bga lore check "Bilbo met Gandalf" --neo4j
    """
    from book_graph_analyzer.lore import LoreChecker
    
    checker = LoreChecker()
    
    if bible:
        checker.load_world_bible(bible)
        console.print(f"[dim]Loaded world bible: {bible}[/dim]")
    
    if corpus:
        checker.load_corpus_entities(corpus)
        console.print(f"[dim]Loaded corpus entities: {corpus}[/dim]")
    
    if timeline:
        checker.load_timeline(timeline)
        console.print(f"[dim]Loaded timeline: {timeline}[/dim]")
    
    if events:
        checker.load_events(events)
        console.print(f"[dim]Loaded events: {events}[/dim]")
    
    if neo4j:
        if checker.connect_neo4j():
            console.print(f"[dim]Connected to Neo4j[/dim]")
        else:
            console.print(f"[yellow]Could not connect to Neo4j[/yellow]")
    
    if not bible and not corpus and not timeline and not events and not neo4j:
        console.print("[yellow]Warning: No knowledge base loaded. Results will be limited.[/yellow]")
    
    console.print()
    
    result = checker.check(claim)
    console.print(result.summary())


@lore.command(name="events")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file (JSON)")
@click.option("--neo4j", is_flag=True, help="Also write events to Neo4j")
@click.option("--chunk-size", default=3000, help="Characters per chunk (default: 3000)")
@click.option("--no-llm", is_flag=True, help="Use pattern matching instead of LLM")
@click.option("--checkpoint", "-c", type=click.Path(), help="Checkpoint file for resume support")
def lore_events(path: str, output: str, neo4j: bool, chunk_size: int, no_llm: bool, checkpoint: str) -> None:
    """Extract events from a text file.
    
    Identifies key events with participants and temporal ordering.
    Uses chunked processing for full books.
    
    Use --checkpoint to save progress and resume on failure.
    
    Examples:
        bga lore events hobbit.txt -o hobbit_events.json
        bga lore events hobbit.txt -o events.json --neo4j
        bga lore events hobbit.txt -o events.json -c hobbit.checkpoint
    """
    from book_graph_analyzer.lore import EventExtractor
    from book_graph_analyzer.ingest.loader import load_book
    
    file_path = Path(path)
    book_name = file_path.stem.replace("_", " ").replace("-", " ").title()
    
    console.print(f"[bold]Extracting events from:[/bold] {file_path.name}")
    
    with console.status("Loading text..."):
        text = load_book(file_path)
    
    console.print(f"[dim]Loaded {len(text):,} characters[/dim]")
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting events...", total=100)
        
        def update_progress(current, total, message):
            progress.update(task, completed=int(current / total * 100), description=message)
        
        extractor = EventExtractor(use_llm=not no_llm, progress_callback=update_progress)
        
        # Use chunked extraction for full books
        if len(text) > chunk_size * 2:
            graph = extractor.extract_from_book(
                text, 
                source_book=book_name, 
                chunk_size=chunk_size,
                checkpoint_file=checkpoint,
            )
        else:
            graph = extractor.extract_from_text(text, source_book=book_name)
    
    # Summary
    console.print(f"\n[bold]Events extracted:[/bold]")
    console.print(f"  Events: {len(graph.events)}")
    console.print(f"  Temporal relations: {len(graph.relations)}")
    
    if graph.events:
        console.print(f"\n[bold]Sample events:[/bold]")
        for event in list(graph.events.values())[:10]:
            time_info = ""
            if event.year:
                time_info = f" (Year {event.year})"
            elif event.era:
                time_info = f" ({event.era.value.replace('_', ' ').title()})"
            console.print(f"  - {event.description}{time_info}")
    
    if graph.relations:
        console.print(f"\n[bold]Sample temporal relations:[/bold]")
        for rel in graph.relations[:5]:
            e1 = graph.events.get(rel.event1_id, None)
            e2 = graph.events.get(rel.event2_id, None)
            e1_name = e1.description if e1 else rel.event1_id
            e2_name = e2.description if e2 else rel.event2_id
            console.print(f"  - {e1_name} --{rel.relation}--> {e2_name}")
    
    # Save to JSON
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(graph.to_dict(), f, indent=2)
    
    console.print(f"\n[green]OK[/green] Events saved to {output_path}")
    
    # Write to Neo4j if requested
    if neo4j:
        from book_graph_analyzer.graph.writer import GraphWriter
        from book_graph_analyzer.graph.connection import check_neo4j_connection
        
        if not check_neo4j_connection():
            console.print("[red]Error:[/red] Cannot connect to Neo4j")
            return
        
        console.print("\n[bold]Writing to Neo4j...[/bold]")
        
        writer = GraphWriter()
        stats = writer.write_event_graph(
            graph,
            book=book_name,
            link_entities=True,
        )
        writer.close()
        
        console.print(f"  Events written: {stats['events_written']}")
        console.print(f"  Relations written: {stats['relations_written']}")
        console.print(f"  Entity links created: {stats['entity_links']}")
        console.print(f"[green]OK[/green] Events written to Neo4j")


@lore.command(name="query-events")
@click.option("--agent", "-a", help="Filter by agent (who did it)")
@click.option("--action", help="Filter by action verb")
@click.option("--patient", "-p", help="Filter by patient (what was acted upon)")
@click.option("--era", "-e", help="Filter by era (first_age, second_age, etc.)")
@click.option("--limit", "-n", default=20, help="Maximum results (default: 20)")
def lore_query_events(agent: str | None, action: str | None, patient: str | None, era: str | None, limit: int) -> None:
    """Query events from Neo4j.
    
    Examples:
        bga lore query-events --agent Bilbo
        bga lore query-events --action found --patient Ring
        bga lore query-events --era third_age --limit 50
    """
    from book_graph_analyzer.graph.writer import GraphWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection
    
    if not check_neo4j_connection():
        console.print("[red]Error:[/red] Cannot connect to Neo4j")
        return
    
    writer = GraphWriter()
    events = writer.query_events(
        agent=agent,
        action=action,
        patient=patient,
        era=era,
        limit=limit,
    )
    writer.close()
    
    if not events:
        console.print("[yellow]No events found matching criteria[/yellow]")
        return
    
    console.print(f"[bold]Found {len(events)} events:[/bold]\n")
    
    table = Table(show_header=True)
    table.add_column("Description", style="cyan")
    table.add_column("Agent")
    table.add_column("Action")
    table.add_column("Patient")
    table.add_column("Era")
    table.add_column("Year")
    
    for e in events:
        era_str = (e.get("era") or "").replace("_", " ").title() if e.get("era") else "-"
        year_str = str(e.get("year")) if e.get("year") else "-"
        table.add_row(
            e.get("description", "-")[:50],
            e.get("agent") or "-",
            e.get("action") or "-",
            e.get("patient") or "-",
            era_str,
            year_str,
        )
    
    console.print(table)


@lore.command(name="timeline")
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True, help="Output file (JSON)")
def lore_timeline(path: str, output: str) -> None:
    """Extract timeline from a text file.
    
    Identifies characters, events, and their temporal relationships.
    
    Example:
        bga lore timeline silmarillion.txt -o silmarillion_timeline.json
    """
    from book_graph_analyzer.lore import TemporalExtractor
    from book_graph_analyzer.ingest.loader import load_book
    
    file_path = Path(path)
    
    console.print(f"[bold]Extracting timeline from:[/bold] {file_path.name}")
    
    with console.status("Loading text..."):
        text = load_book(file_path)
    
    extractor = TemporalExtractor(use_llm=True)
    
    with console.status("Extracting temporal information..."):
        timeline = extractor.extract_from_text(text)
    
    # Summary
    console.print(f"\n[bold]Timeline extracted:[/bold]")
    console.print(f"  Entities: {len(timeline.entities)}")
    console.print(f"  Relations: {len(timeline.relations)}")
    
    if timeline.entities:
        console.print(f"\n[bold]Sample entities:[/bold]")
        for name, entity in list(timeline.entities.items())[:10]:
            era_info = ""
            if entity.birth_era:
                era_info = f" (born: {entity.birth_era.value})"
            if entity.death_era:
                era_info += f" (died: {entity.death_era.value})"
            console.print(f"  - {name}{era_info}")
    
    # Save
    output_path = Path(output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(timeline.to_dict(), f, indent=2)
    
    console.print(f"\n[green]OK[/green] Timeline saved to {output_path}")


@lore.command(name="validate")
@click.argument("text_file", type=click.Path(exists=True))
@click.option("--bible", "-b", type=click.Path(exists=True), required=True, help="World bible file")
@click.option("--corpus", "-c", help="Corpus name for entity lookup")
@click.option("--output", "-o", type=click.Path(), help="Output file for results (JSON)")
def lore_validate(text_file: str, bible: str, corpus: str | None, output: str | None) -> None:
    """Validate all claims in a text file.
    
    Useful for checking draft chapters or generated content.
    
    Example:
        bga lore validate my_chapter.txt -b hobbit_bible.json -o validation_results.json
    """
    from book_graph_analyzer.lore import LoreChecker
    
    checker = LoreChecker()
    checker.load_world_bible(bible)
    
    if corpus:
        checker.load_corpus_entities(corpus)
    
    # Load text
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    console.print(f"[bold]Validating: {text_file}[/bold]")
    console.print(f"[dim]Using: {bible}[/dim]\n")
    
    with console.status("Checking claims..."):
        results = checker.check_text(text)
    
    # Summary
    valid = sum(1 for r in results if r.status.value == "valid")
    invalid = sum(1 for r in results if r.status.value == "invalid")
    unknown = sum(1 for r in results if r.status.value == "unknown")
    plausible = sum(1 for r in results if r.status.value == "plausible")
    
    console.print(f"[bold]Results: {len(results)} claims checked[/bold]")
    console.print(f"  [green]Valid:[/green] {valid}")
    console.print(f"  [red]Invalid:[/red] {invalid}")
    console.print(f"  [yellow]Unknown:[/yellow] {unknown}")
    console.print(f"  [cyan]Plausible:[/cyan] {plausible}")
    
    # Show issues
    if invalid > 0:
        console.print(f"\n[bold red]Issues Found:[/bold red]")
        for r in results:
            if r.status.value == "invalid":
                console.print(r.summary())
                console.print()
    
    # Save output
    if output:
        output_data = {
            "file": text_file,
            "bible": bible,
            "summary": {
                "total": len(results),
                "valid": valid,
                "invalid": invalid,
                "unknown": unknown,
                "plausible": plausible,
            },
            "results": [r.to_dict() for r in results],
        }
        
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"\n[green]OK[/green] Results saved to {output}")


@lore.command(name="interactive")
@click.option("--bible", "-b", type=click.Path(exists=True), help="World bible file")
@click.option("--corpus", "-c", help="Corpus name for entity lookup")
def lore_interactive(bible: str | None, corpus: str | None) -> None:
    """Interactive lore checking session.
    
    Enter claims one at a time and get immediate feedback.
    
    Example:
        bga lore interactive -b hobbit_bible.json
    """
    from book_graph_analyzer.lore import LoreChecker
    
    checker = LoreChecker()
    
    if bible:
        checker.load_world_bible(bible)
        console.print(f"[green]OK[/green] Loaded world bible: {bible}")
    
    if corpus:
        checker.load_corpus_entities(corpus)
        console.print(f"[green]OK[/green] Loaded corpus: {corpus}")
    
    console.print("\n[bold]Lore Checker Interactive Mode[/bold]")
    console.print("Enter claims to check. Type 'quit' to exit.\n")
    
    while True:
        try:
            claim = console.input("[cyan]Claim>[/cyan] ")
            if claim.lower() in ('quit', 'exit', 'q'):
                break
            if not claim.strip():
                continue
            
            result = checker.check(claim)
            console.print(result.summary())
            console.print()
            
        except (KeyboardInterrupt, EOFError):
            break
    
    console.print("\n[dim]Goodbye![/dim]")


# =============================================================================
# Generate Commands
# =============================================================================

@main.group()
def generate() -> None:
    """Generate lore-consistent story content."""
    pass


@generate.command(name="scene")
@click.option("--goal", "-g", required=True, help="What should happen in this scene")
@click.option("--characters", "-c", multiple=True, required=True, help="Characters present (repeat for multiple)")
@click.option("--place", "-p", required=True, help="Where the scene takes place")
@click.option("--objects", "-obj", multiple=True, help="Objects of note in the scene")
@click.option("--context", help="Previous events/context for continuity")
@click.option("--world-bible", "-w", type=click.Path(exists=True), help="World bible JSON for rules")
@click.option("--output", "-o", type=click.Path(), help="Save scene to JSON file")
@click.option("--neo4j", is_flag=True, help="Write scene to Neo4j")
@click.option("--chapter-id", help="Chapter ID to link scene to (for Neo4j)")
def generate_scene(
    goal: str,
    characters: tuple[str],
    place: str,
    objects: tuple[str],
    context: str,
    world_bible: str,
    output: str,
    neo4j: bool,
    chapter_id: str,
) -> None:
    """Generate a single scene grounded in the knowledge graph.
    
    Example:
        bga generate scene -g "Bilbo meets Gandalf" -c Bilbo -c Gandalf -p "Bag End"
    """
    from book_graph_analyzer.generate import SceneGenerator, GenerationWriter
    
    console.print("[bold]Generating Scene[/bold]\n")
    console.print(f"Goal: {goal}")
    console.print(f"Characters: {', '.join(characters)}")
    console.print(f"Place: {place}")
    if objects:
        console.print(f"Objects: {', '.join(objects)}")
    console.print()
    
    generator = SceneGenerator()
    
    if world_bible:
        generator.load_world_bible(world_bible)
        console.print(f"[dim]Loaded world bible: {world_bible}[/dim]")
    
    with console.status("Generating scene..."):
        scene = generator.generate_scene(
            scene_goal=goal,
            characters=list(characters),
            place=place,
            previous_context=context or "",
            objects=list(objects) if objects else None,
        )
    
    # Display results
    console.print(f"\n[bold]Generated Scene[/bold] (ID: {scene.id})")
    console.print(f"Status: {scene.status.value}")
    console.print(f"Word count: {scene.word_count}")
    console.print(f"Revisions: {scene.revision_count}")
    
    console.print(f"\n[bold]Scores:[/bold]")
    console.print(f"  Overall: {scene.scores.overall:.0%}")
    console.print(f"  Lore: {scene.scores.lore_score:.0%}")
    console.print(f"  Style: {scene.scores.style_score:.0%}")
    console.print(f"  Narrative: {scene.scores.narrative_score:.0%}")
    console.print(f"    - Engagement: {scene.scores.engagement:.0%}")
    console.print(f"    - Pacing: {scene.scores.pacing:.0%}")
    console.print(f"    - Dialogue: {scene.scores.dialogue:.0%}")
    console.print(f"    - Imagery: {scene.scores.imagery:.0%}")
    
    if scene.critique_notes:
        console.print(f"\n[bold]Notes:[/bold]")
        for note in scene.critique_notes[:5]:
            console.print(f"  - {note[:100]}...")
    
    console.print(f"\n[bold]Text:[/bold]")
    console.print("-" * 60)
    console.print(scene.text)
    console.print("-" * 60)
    
    # Save to file
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(scene.to_dict(), f, indent=2)
        console.print(f"\n[green]OK[/green] Saved to {output}")
    
    # Write to Neo4j
    if neo4j:
        writer = GenerationWriter()
        writer.ensure_schema()
        stats = writer.write_scene(scene, chapter_id or "standalone")
        console.print(f"\n[green]OK[/green] Written to Neo4j: {stats}")


@generate.command(name="init-schema")
def generate_init_schema() -> None:
    """Initialize Neo4j schema for generated content."""
    from book_graph_analyzer.generate import GenerationWriter
    
    writer = GenerationWriter()
    writer.ensure_schema()
    console.print("[green]OK[/green] Generation schema initialized")


@generate.command(name="flagged")
@click.option("--limit", "-n", default=10, help="Number of scenes to show")
def generate_flagged(limit: int) -> None:
    """Show scenes flagged for human review."""
    from book_graph_analyzer.generate import GenerationWriter
    
    writer = GenerationWriter()
    scenes = writer.get_flagged_scenes(limit)
    
    if not scenes:
        console.print("[dim]No flagged scenes found[/dim]")
        return
    
    console.print(f"[bold]Flagged Scenes ({len(scenes)})[/bold]\n")
    
    for scene in scenes:
        console.print(f"[bold]ID:[/bold] {scene['id']}")
        console.print(f"[bold]Summary:[/bold] {scene['summary']}")
        console.print(f"[bold]Score:[/bold] {scene['score']:.0%}")
        console.print(f"[bold]Characters:[/bold] {', '.join(scene['characters'])}")
        console.print(f"\n{scene['text'][:500]}...")
        console.print("-" * 40)


@generate.command(name="review")
@click.argument("scene_id")
@click.option("--approve", is_flag=True, help="Approve the scene")
@click.option("--reject", is_flag=True, help="Reject the scene")
@click.option("--notes", "-n", help="Review notes")
def generate_review(scene_id: str, approve: bool, reject: bool, notes: str) -> None:
    """Review a flagged scene."""
    from book_graph_analyzer.generate import GenerationWriter
    
    if approve and reject:
        console.print("[red]Cannot both approve and reject[/red]")
        return
    
    if not approve and not reject:
        console.print("[red]Specify --approve or --reject[/red]")
        return
    
    status = "approved" if approve else "flagged"  # rejected stays flagged with notes
    
    writer = GenerationWriter()
    if writer.update_scene_status(scene_id, status, notes or ""):
        console.print(f"[green]OK[/green] Scene {scene_id} marked as {status}")
    else:
        console.print(f"[red]Failed to update scene {scene_id}[/red]")


@generate.command(name="by-character")
@click.argument("character")
@click.option("--min-quality", "-q", default=0.0, help="Minimum quality score")
def generate_by_character(character: str, min_quality: float) -> None:
    """List generated scenes featuring a character."""
    from book_graph_analyzer.generate import GenerationWriter
    
    writer = GenerationWriter()
    scenes = writer.get_scenes_by_character(character, min_quality)
    
    if not scenes:
        console.print(f"[dim]No scenes found for {character}[/dim]")
        return
    
    console.print(f"[bold]Scenes featuring {character} ({len(scenes)})[/bold]\n")
    
    table = Table(show_header=True)
    table.add_column("ID")
    table.add_column("Summary")
    table.add_column("Quality")
    
    for scene in scenes:
        table.add_row(
            scene["id"],
            scene["summary"][:50] + "..." if len(scene.get("summary", "")) > 50 else scene.get("summary", ""),
            f"{scene['score']:.0%}" if scene.get("score") else "N/A",
        )
    
    console.print(table)


if __name__ == "__main__":
    main()

