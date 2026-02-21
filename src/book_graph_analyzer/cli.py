"""Command-line interface for Book Graph Analyzer."""

import io
import json
import sys
from pathlib import Path

# Fix Windows cp1252 encoding — ensure stdout/stderr always speak UTF-8
if hasattr(sys.stdout, "buffer") and getattr(sys.stdout, "encoding", "utf-8").lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer") and getattr(sys.stderr, "encoding", "utf-8").lower() != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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

        # Temporal coverage — how many relationships have era_start set?
        result = session.run(
            "MATCH ()-[r]->() WHERE r.era_start IS NOT NULL RETURN count(r) as cnt"
        )
        temporal_cnt = result.single()["cnt"]
        pct = (temporal_cnt / rel_count * 100) if rel_count else 0
        console.print(f"[bold]Temporally-tagged relationships:[/bold] {temporal_cnt:,} ({pct:.0f}%)")

    driver.close()


@graph.command(name="init-eras")
def graph_init_eras() -> None:
    """Create Era nodes and FOLLOWED_BY chain in Neo4j."""
    from book_graph_analyzer.graph.writer import GraphWriter
    writer = GraphWriter()
    writer.init_era_chain()
    console.print("[green]Era chain initialised:[/green]")
    console.print("  Before Time -> Years of the Lamps -> Years of the Trees")
    console.print("  -> First Age -> Second Age -> Third Age -> Fourth Age")


@graph.command(name="at-time")
@click.option("--character", "-c", required=True, help="Character canonical name")
@click.option("--era", "-e", required=True,
              help="Era name: 'Third Age', 'Second Age', 'First Age', etc.")
@click.option("--year", "-y", type=int, default=None, help="Year within the era")
def graph_at_time(character: str, era: str, year: int | None) -> None:
    """Show everything known about a character at a specific point in time.

    Example:
        bga graph at-time --character Gandalf --era "Third Age" --year 3018
    """
    from book_graph_analyzer.graph.writer import GraphWriter

    writer = GraphWriter()
    snapshot = writer.query_at_time(character, era, year)

    if "error" in snapshot:
        console.print(f"[red]{snapshot['error']}[/red]")
        return

    char = snapshot["character"]
    at = snapshot["at"]
    rels = snapshot["relationships"]
    events = snapshot["events"]

    year_str = f" {at['year']}" if at["year"] else ""
    console.print(f"\n[bold cyan]{char.get('canonical_name', character)}[/bold cyan]"
                  f"  at  [bold]{at['era']}{year_str}[/bold]\n")

    if rels:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Relationship", style="cyan", width=20)
        table.add_column("Entity")
        table.add_column("Type", style="dim", width=12)
        table.add_column("Valid from", style="dim")
        table.add_column("Valid until", style="dim")

        for r in rels:
            era_start = r.get("era_start") or "always"
            era_end   = r.get("era_end") or "ongoing"
            yr_s = f" {r['year_start']}" if r.get("year_start") else ""
            yr_e = f" {r['year_end']}"   if r.get("year_end")   else ""
            table.add_row(
                r.get("rel", "?"),
                r.get("name") or "?",
                r.get("type") or "?",
                f"{era_start}{yr_s}",
                f"{era_end}{yr_e}",
            )
        console.print("[bold]Relationships:[/bold]")
        console.print(table)
    else:
        console.print("[dim]No temporally-filtered relationships found.[/dim]")

    if events:
        console.print(f"\n[bold]Events in {era}:[/bold]")
        for ev in events:
            yr = f" ({ev['year']})" if ev.get("year") else ""
            console.print(f"  [dim]-[/dim] {ev.get('description', '?')}{yr}")
    else:
        console.print(f"\n[dim]No events found for {era}.[/dim]")


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


@lore.command(name="weight")
@click.option("--passage-id", "-p", default=None,
              help="Passage ID to fetch from Neo4j and score.")
@click.option("--text", "-t", default=None,
              help="Raw text to score directly (no Neo4j needed).")
@click.option("--era-refs", default=0, type=int,
              help="Number of era references in the passage (for --text mode).")
@click.option("--temporal-depth-years", default=None, type=float,
              help="Temporal depth in years (for --text mode).")
@click.option("--entity-count", default=0, type=int,
              help="Number of named entities (for --text mode).")
@click.option("--is-dialogue", is_flag=True, default=False,
              help="Treat passage as dialogue.")
@click.option("--suggestions", "-s", default=3, type=int,
              help="Number of improvement suggestions to show (default: 3).")
@click.option("--themes", is_flag=True, default=False,
              help="List detected Tolkien themes.")
def lore_weight(
    passage_id: str | None,
    text: str | None,
    era_refs: int,
    temporal_depth_years: float | None,
    entity_count: int,
    is_dialogue: bool,
    suggestions: int,
    themes: bool,
) -> None:
    """Compute NarrativeWeight breakdown for a passage.

    Analyse why a passage is (or isn't) compelling using the NarrativeWeight
    composite metric. Shows scores for all 14 components and improvement suggestions.

    Examples:
        bga lore weight --text "Gandalf spoke of the Elder Days..."
        bga lore weight --passage-id p_001 --suggestions 5
        bga lore weight --text "Bilbo found a ring" --is-dialogue --themes
    """
    from book_graph_analyzer.lore.narrative_weight import NarrativeWeightComputer
    from book_graph_analyzer.models.narrative_weight import TOLKIEN_THEMES

    computer = NarrativeWeightComputer()

    if not passage_id and not text:
        console.print("[red]Provide --passage-id or --text.[/red]")
        return

    if text:
        # Direct text scoring (no Neo4j needed)
        weight = computer.compute_from_text(
            text=text,
            era_ref_count=era_refs,
            temporal_depth_years=temporal_depth_years,
            entity_count=entity_count,
            is_dialogue=is_dialogue,
        )
        label = "inline text"
    else:
        # Fetch from Neo4j
        from book_graph_analyzer.graph.connection import get_driver, check_neo4j_connection
        from book_graph_analyzer.models.passage import Passage

        if not check_neo4j_connection():
            console.print("[red]Cannot connect to Neo4j. Use --text for offline scoring.[/red]")
            return

        driver = get_driver()
        with driver.session() as session:
            result = session.run("MATCH (p:Passage {id: $id}) RETURN p", id=passage_id)
            row = result.single()
            if not row:
                console.print(f"[red]Passage '{passage_id}' not found in Neo4j.[/red]")
                driver.close()
                return
            node = dict(row["p"])

        driver.close()

        # Reconstruct enough of a Passage to compute weight
        passage = Passage(
            id=node.get("id", passage_id),
            text=node.get("text", ""),
            book=node.get("book", ""),
            chapter=node.get("chapter", ""),
            chapter_num=node.get("chapter_num", 0),
            paragraph_num=node.get("paragraph_num", 0),
            sentence_num=node.get("sentence_num", 0),
            char_offset=node.get("char_offset", 0),
            story_era=node.get("story_era"),
            story_year=node.get("story_year"),
            temporal_depth_era=node.get("temporal_depth_era"),
            temporal_depth_years_back=node.get("temporal_depth_years_back"),
            era_reference_count=node.get("era_reference_count", 0),
            is_dialogue=node.get("is_dialogue", False),
            speaker_ids=node.get("speaker_ids") or [],
        )
        weight = computer.compute_from_passage(passage)
        label = passage_id

    # Display
    console.print(f"\n[bold]NarrativeWeight Analysis[/bold] — {label}\n")
    console.print(weight.summary(label))

    # Themes
    if themes:
        source = text or ""
        if not source and passage_id:
            # Already have node text from above if we fetched from Neo4j
            try:
                source = passage.text  # type: ignore[name-defined]
            except NameError:
                source = ""
        detected = computer.detect_themes(source)
        if detected:
            console.print(f"\n[bold]Detected Tolkien Themes ({len(detected)}):[/bold]")
            for theme in detected:
                ts = "[yellow]★[/yellow]" if theme.tolkien_specific else "  "
                console.print(f"  {ts} [cyan]{theme.name}[/cyan]")
                console.print(f"     {theme.description[:100]}...")
        else:
            console.print("\n[dim]No themes detected in this passage.[/dim]")

    console.print(f"\n[bold]Overall Score:[/bold] [green]{weight.overall:.3f}[/green] / 1.000")


@lore.command(name="themes")
def lore_themes() -> None:
    """List all Tolkien themes in the NarrativeWeight taxonomy.

    Example:
        bga lore themes
    """
    from book_graph_analyzer.models.narrative_weight import TOLKIEN_THEMES

    console.print(f"\n[bold]Tolkien Theme Taxonomy ({len(TOLKIEN_THEMES)} themes)[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=30)
    table.add_column("Name", width=30)
    table.add_column("Tolkien-specific", width=16)
    table.add_column("Description")

    for theme in TOLKIEN_THEMES:
        ts = "[yellow]★ Yes[/yellow]" if theme.tolkien_specific else "No"
        table.add_row(
            theme.id,
            theme.name,
            ts,
            theme.description[:80] + "..." if len(theme.description) > 80 else theme.description,
        )
    console.print(table)


@lore.command(name="top-passages")
@click.option("--limit", "-n", default=20, help="Number of top passages to return")
@click.option("--min-score", default=0.0, type=float, help="Minimum overall score")
def lore_top_passages(limit: int, min_score: float) -> None:
    """Show top passages by NarrativeWeight overall score from Neo4j.

    Requires passages to have NarrativeWeight computed first
    (run 'bga lore weight --passage-id ...' for individual passages).

    Example:
        bga lore top-passages -n 10
        bga lore top-passages --min-score 0.5
    """
    from book_graph_analyzer.lore.narrative_weight import NarrativeWeightNeo4jWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    if not check_neo4j_connection():
        console.print("[red]Cannot connect to Neo4j.[/red]")
        return

    writer = NarrativeWeightNeo4jWriter()
    results = writer.query_top_passages(limit=limit, min_overall=min_score)
    writer.close()

    if not results:
        console.print("[yellow]No scored passages found. Score passages first with 'bga lore weight'.[/yellow]")
        return

    console.print(f"\n[bold]Top {len(results)} Passages by NarrativeWeight[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", width=22)
    table.add_column("Score", justify="right", width=7)
    table.add_column("Story-era", style="cyan", width=14)
    table.add_column("Depth era", style="magenta", width=14)
    table.add_column("Text snippet")

    for r in results:
        snippet = (r.get("text") or "")[:60]
        if len(r.get("text") or "") > 60:
            snippet += "..."
        table.add_row(
            r.get("id") or "-",
            f"{r.get('overall', 0):.3f}",
            r.get("story_era") or "-",
            r.get("depth_era") or "-",
            snippet,
        )
    console.print(table)


@lore.group(name="conflicts")
def lore_conflicts() -> None:
    """Track and manage intra/inter-book lore contradictions."""
    pass


@lore_conflicts.command(name="list")
@click.option("--conflict-type", "-t", default=None,
              help="Filter by type: direct_contradiction, retcon, ambiguity, interpretation")
@click.option("--resolved", is_flag=True, default=False, help="Show only resolved conflicts")
@click.option("--unresolved", is_flag=True, default=False, help="Show only unresolved conflicts")
@click.option("--neo4j", is_flag=True, help="Read from Neo4j instead of built-in conflicts")
def lore_conflicts_list(
    conflict_type: str | None, resolved: bool, unresolved: bool, neo4j: bool
) -> None:
    """List all tracked lore conflicts.

    Examples:
        bga lore conflicts list
        bga lore conflicts list --unresolved
        bga lore conflicts list --conflict-type retcon
        bga lore conflicts list --neo4j
    """
    from book_graph_analyzer.lore.conflicts import ConflictRegistry, LoreConflictNeo4jWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    if neo4j:
        if not check_neo4j_connection():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            return
        writer = LoreConflictNeo4jWriter()
        conflicts_raw = writer.query_conflicts(
            conflict_type=conflict_type,
            resolved=True if resolved else (False if unresolved else None),
            needs_human=False,
        )
        writer.close()
        if not conflicts_raw:
            console.print("[yellow]No conflicts found in Neo4j. Run 'bga lore conflicts init' first.[/yellow]")
            return
        console.print(f"\n[bold]LoreConflicts in Neo4j ({len(conflicts_raw)})[/bold]\n")
        for c in conflicts_raw:
            status = "[green]✓[/green]" if c.get("resolved") else "[yellow]?[/yellow]"
            console.print(f"  {status} [{c.get('conflict_type', '?')}] [cyan]{c.get('id')}[/cyan]")
            console.print(f"    {c.get('summary', '')[:80]}")
            console.print(f"    Policy: {c.get('resolution_policy', '?')}")
            console.print()
        return

    # Built-in registry
    registry = ConflictRegistry.from_tolkien_defaults()
    conflicts = registry.all()
    if conflict_type:
        conflicts = [c for c in conflicts if c.conflict_type == conflict_type]
    if resolved:
        conflicts = [c for c in conflicts if c.is_resolved]
    elif unresolved:
        conflicts = [c for c in conflicts if not c.is_resolved]

    console.print(f"\n[bold]Known Tolkien Lore Conflicts ({len(conflicts)})[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=28)
    table.add_column("Type", width=20)
    table.add_column("Policy", width=22)
    table.add_column("✓", width=3)
    table.add_column("Summary")

    for conflict in sorted(conflicts, key=lambda c: c.id):
        status = "[green]✓[/green]" if conflict.is_resolved else "[yellow]?[/yellow]"
        table.add_row(
            conflict.id,
            conflict.conflict_type,
            conflict.resolution_policy,
            status,
            conflict.summary[:60] + "..." if len(conflict.summary) > 60 else conflict.summary,
        )
    console.print(table)

    res = sum(1 for c in conflicts if c.is_resolved)
    unres = len(conflicts) - res
    console.print(f"\n  {res} resolved · {unres} unresolved")


@lore_conflicts.command(name="show")
@click.argument("conflict_id")
def lore_conflicts_show(conflict_id: str) -> None:
    """Show full detail for a single LoreConflict.

    Example:
        bga lore conflicts show glorfindel_identity
    """
    from book_graph_analyzer.lore.conflicts import ConflictRegistry

    registry = ConflictRegistry.from_tolkien_defaults()
    conflict = registry.get(conflict_id)

    if not conflict:
        console.print(f"[red]Conflict '{conflict_id}' not found.[/red]")
        console.print("\nAvailable conflict IDs:")
        for c in registry.all():
            console.print(f"  {c.id}")
        return

    console.print(f"\n{conflict.detail()}")


@lore_conflicts.command(name="unresolved")
def lore_conflicts_unresolved() -> None:
    """Show conflicts needing human attention (flag_for_human or irresolvable).

    Example:
        bga lore conflicts unresolved
    """
    from book_graph_analyzer.lore.conflicts import ConflictRegistry

    registry = ConflictRegistry.from_tolkien_defaults()
    human_needed = registry.needing_human_review()

    if not human_needed:
        console.print("[green]No conflicts need human review.[/green]")
        return

    console.print(f"\n[bold yellow]Conflicts requiring human review ({len(human_needed)}):[/bold yellow]\n")
    for conflict in human_needed:
        console.print(f"  [yellow]⚠[/yellow] [cyan]{conflict.id}[/cyan]  [{conflict.conflict_type}]")
        console.print(f"    {conflict.summary[:100]}")
        if conflict.claims:
            for i, claim in enumerate(conflict.claims, 1):
                console.print(f"      [{i}] ({claim.author_period} / {claim.source_book}) {claim.statement[:80]}")
        console.print()


@lore_conflicts.command(name="resolve")
@click.option("--id", "conflict_id", required=True, help="Conflict ID to resolve")
@click.option("--policy", required=True,
              type=click.Choice([
                  "use_later_text", "use_earlier_text", "both_valid_in_universe",
                  "flag_for_human", "use_most_cited", "irresolvable"
              ]),
              help="Resolution policy to apply")
@click.option("--notes", default="", help="Notes explaining the resolution")
@click.option("--neo4j", is_flag=True, help="Also persist resolution to Neo4j")
def lore_conflicts_resolve(
    conflict_id: str, policy: str, notes: str, neo4j: bool
) -> None:
    """Apply a resolution policy to a conflict.

    Examples:
        bga lore conflicts resolve --id orc_origin --policy use_later_text
        bga lore conflicts resolve --id bombadil_nature --policy irresolvable --notes "Tolkien's intent"
        bga lore conflicts resolve --id glorfindel_identity --policy use_later_text --neo4j
    """
    from book_graph_analyzer.lore.conflicts import ConflictRegistry, LoreConflictNeo4jWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    registry = ConflictRegistry.from_tolkien_defaults()
    success = registry.resolve(conflict_id, policy, notes)

    if not success:
        console.print(f"[red]Conflict '{conflict_id}' not found in built-in registry.[/red]")
        return

    conflict = registry.get(conflict_id)
    status = "[green]✓ Resolved[/green]" if conflict.resolved else "[yellow]Flagged[/yellow]"
    console.print(f"\n{status}: [{conflict_id}] → {policy}")

    winner = conflict.winning_claim()
    if winner:
        console.print(f"  Active claim: ({winner.author_period} / {winner.source_book}) {winner.statement[:80]}")

    if neo4j:
        if not check_neo4j_connection():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            return
        writer = LoreConflictNeo4jWriter()
        writer.resolve_conflict(conflict_id, policy, notes)
        writer.close()
        console.print("[green]OK[/green] Neo4j updated.")


@lore_conflicts.command(name="init")
@click.option("--dry-run", is_flag=True, help="Show what would be written without writing")
def lore_conflicts_init(dry_run: bool) -> None:
    """Write all known Tolkien conflicts to Neo4j.

    This seeds the conflict database. Safe to run multiple times (idempotent MERGE).

    Example:
        bga lore conflicts init
        bga lore conflicts init --dry-run
    """
    from book_graph_analyzer.lore.conflicts import ConflictRegistry, LoreConflictNeo4jWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    registry = ConflictRegistry.from_tolkien_defaults()
    conflicts = registry.all()

    console.print(f"[bold]Initialising {len(conflicts)} known Tolkien conflicts[/bold]")

    if dry_run:
        console.print("\n[yellow][DRY RUN] Would write:[/yellow]")
        for c in conflicts:
            status = "✓" if c.is_resolved else "?"
            console.print(f"  [{status}] {c.id}: {c.summary[:70]}")
        return

    if not check_neo4j_connection():
        console.print("[red]Cannot connect to Neo4j.[/red]")
        return

    writer = LoreConflictNeo4jWriter()
    writer.ensure_schema()
    count = writer.upsert_many(conflicts)

    # Create CONFLICTS_WITH edges for rule-linked conflicts
    for c in conflicts:
        if len(c.rule_ids) >= 2:
            for i in range(len(c.rule_ids) - 1):
                try:
                    writer.create_conflicts_with_edge(
                        c.rule_ids[i], c.rule_ids[i + 1], c.id
                    )
                except Exception:
                    pass  # Rules may not exist yet

    writer.close()
    console.print(f"[green]OK[/green] {count} LoreConflict nodes written to Neo4j")


@lore.group(name="rules")
def lore_rules() -> None:
    """Manage LoreRule nodes — lore laws as executable Cypher contracts."""
    pass


@lore_rules.command(name="list")
@click.option("--category", "-c", default=None,
              help="Filter by category (race, magic, cosmology, geography, politics, metaphysics, objects, history)")
@click.option("--hardness", "-H", default=None,
              help="Filter by hardness: HARD or SOFT")
@click.option("--neo4j", is_flag=True, help="Read from Neo4j instead of built-in rules")
def lore_rules_list(category: str | None, hardness: str | None, neo4j: bool) -> None:
    """List all LoreRule definitions.

    Examples:
        bga lore rules list
        bga lore rules list --category magic
        bga lore rules list --hardness HARD
        bga lore rules list --neo4j
    """
    from book_graph_analyzer.lore.rules import LoreRuleRegistry, LoreRuleNeo4jWriter, TOLKIEN_LORE_RULES
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    if neo4j:
        if not check_neo4j_connection():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            return
        writer = LoreRuleNeo4jWriter()
        rules_raw = writer.query_rules(category=category, hardness=hardness)
        writer.close()

        if not rules_raw:
            console.print("[yellow]No LoreRule nodes found in Neo4j. Run 'bga lore rules extract' first.[/yellow]")
            return

        console.print(f"\n[bold]LoreRules in Neo4j ({len(rules_raw)})[/bold]\n")
        for r in rules_raw:
            hardness_tag = "[red]HARD[/red]" if r.get("hardness") == "HARD" else "[yellow]SOFT[/yellow]"
            console.print(f"  {hardness_tag} [{r.get('category', '?')}] [cyan]{r.get('id')}[/cyan]")
            console.print(f"    {r.get('statement', '')}")
            console.print()
        return

    # Built-in registry
    registry = LoreRuleRegistry.from_tolkien_defaults()
    rules = registry.all()
    if category:
        rules = [r for r in rules if r.category == category]
    if hardness:
        rules = [r for r in rules if r.hardness == hardness.upper()]

    console.print(f"\n[bold]Built-in Tolkien LoreRules ({len(rules)})[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="cyan", width=28)
    table.add_column("Hard?", width=6)
    table.add_column("Cat", width=11)
    table.add_column("Scope", width=14)
    table.add_column("Statement")

    for rule in sorted(rules, key=lambda r: (r.category, r.hardness)):
        hardness_cell = "[red]HARD[/red]" if rule.is_hard else "[yellow]SOFT[/yellow]"
        scope = rule.scope_entity_type or rule.scope_era or "Universal"
        table.add_row(
            rule.id,
            hardness_cell,
            rule.category,
            scope[:14],
            rule.statement[:80],
        )
    console.print(table)

    hard_count = sum(1 for r in rules if r.is_hard)
    soft_count = len(rules) - hard_count
    console.print(f"\n  {hard_count} HARD rules · {soft_count} SOFT rules")


@lore_rules.command(name="show")
@click.argument("rule_id")
def lore_rules_show(rule_id: str) -> None:
    """Show full detail for a single LoreRule.

    Example:
        bga lore rules show magic_ring_corruption
    """
    from book_graph_analyzer.lore.rules import LoreRuleRegistry

    registry = LoreRuleRegistry.from_tolkien_defaults()
    rule = registry.get(rule_id)

    if not rule:
        console.print(f"[red]Rule '{rule_id}' not found.[/red]")
        console.print("\nAvailable rule IDs:")
        for r in registry.all():
            console.print(f"  {r.id}")
        return

    tag = "[red]HARD — blocks scene acceptance[/red]" if rule.is_hard else "[yellow]SOFT — warning only[/yellow]"
    console.print(f"\n[bold cyan]{rule.id}[/bold cyan]  {tag}")
    console.print(f"\n[bold]Statement:[/bold] {rule.statement}")
    console.print(f"[bold]Category:[/bold] {rule.category}")
    console.print(f"[bold]Confidence:[/bold] {rule.confidence:.0%}")
    if rule.scope_entity_type:
        console.print(f"[bold]Scoped to:[/bold] {rule.scope_entity_type}")
    if rule.scope_era:
        console.print(f"[bold]Era scope:[/bold] {rule.scope_era}")

    if rule.cypher_check:
        console.print(f"\n[bold]Cypher check:[/bold]")
        console.print(f"[dim]{rule.cypher_check}[/dim]")
    else:
        console.print("\n[dim]No Cypher check defined (cultural/contextual rule).[/dim]")


@lore_rules.command(name="extract")
@click.option("--bible", "-b", type=click.Path(exists=True), required=True,
              help="World bible JSON file to extract rules from")
@click.option("--output", "-o", default="json",
              type=click.Choice(["json", "neo4j", "both"]),
              help="Output target (default: json)")
@click.option("--out-file", type=click.Path(), default=None,
              help="JSON output file path (when output=json or both)")
def lore_rules_extract(bible: str, output: str, out_file: str | None) -> None:
    """Extract LoreRules from a world bible JSON file.

    Reads WorldBible rules and maps them to LoreRule objects with HARD/SOFT
    classification and optional Cypher check templates.

    Examples:
        bga lore rules extract -b hobbit_bible.json -o json
        bga lore rules extract -b silmarillion_bible.json -o neo4j
        bga lore rules extract -b lotr_bible.json -o both --out-file lotr_rules.json
    """
    import json as _json
    from book_graph_analyzer.worldbible import WorldBibleExtractor
    from book_graph_analyzer.lore.rules import WorldBibleRuleMapper, LoreRuleNeo4jWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    extractor = WorldBibleExtractor()
    bible_obj = extractor.load_bible(bible)

    mapper = WorldBibleRuleMapper()

    with console.status("Mapping world bible rules to LoreRules..."):
        rules = mapper.map_bible(bible_obj)

    console.print(f"[green]Mapped {len(rules)} rules from world bible[/green]")

    hard_count = sum(1 for r in rules if r.is_hard)
    soft_count = len(rules) - hard_count
    console.print(f"  HARD: {hard_count}  ·  SOFT: {soft_count}")

    # JSON output
    if output in ("json", "both"):
        json_path = out_file or Path(bible).with_suffix(".lore_rules.json")
        with open(json_path, "w", encoding="utf-8") as f:
            _json.dump([r.to_dict() for r in rules], f, indent=2)
        console.print(f"[green]OK[/green] Rules saved to {json_path}")

    # Neo4j output
    if output in ("neo4j", "both"):
        if not check_neo4j_connection():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            return
        writer = LoreRuleNeo4jWriter()
        writer.ensure_schema()
        count = writer.upsert_many(rules)
        writer.close()
        console.print(f"[green]OK[/green] {count} LoreRule nodes written to Neo4j")

    # Sample output
    console.print("\n[bold]Sample extracted rules:[/bold]")
    for rule in rules[:5]:
        tag = "[red]HARD[/red]" if rule.is_hard else "[yellow]SOFT[/yellow]"
        console.print(f"  {tag} [{rule.category}] {rule.statement[:70]}")


@lore_rules.command(name="init")
@click.option("--dry-run", is_flag=True, help="Show what would be written without writing")
def lore_rules_init(dry_run: bool) -> None:
    """Write all built-in Tolkien LoreRules to Neo4j.

    This initialises the lore contract database with the canonical Tolkien rules.
    Safe to run multiple times (idempotent MERGE).

    Example:
        bga lore rules init
    """
    from book_graph_analyzer.lore.rules import LoreRuleRegistry, LoreRuleNeo4jWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection

    registry = LoreRuleRegistry.from_tolkien_defaults()
    rules = registry.all()

    console.print(f"[bold]Initialising {len(rules)} built-in LoreRules[/bold]")

    if dry_run:
        console.print("\n[yellow][DRY RUN] Would write:[/yellow]")
        for rule in rules:
            tag = "HARD" if rule.is_hard else "SOFT"
            console.print(f"  [{tag}] {rule.id}: {rule.statement}")
        return

    if not check_neo4j_connection():
        console.print("[red]Cannot connect to Neo4j.[/red]")
        return

    writer = LoreRuleNeo4jWriter()
    writer.ensure_schema()
    count = writer.upsert_many(rules)
    writer.close()

    console.print(f"[green]OK[/green] {count} LoreRule nodes written to Neo4j")


@lore.command(name="validate-scene")
@click.option("--scene-id", "-s", default=None,
              help="Scene node ID in Neo4j to validate")
@click.option("--text", "-t", default=None,
              help="Raw scene text to validate (offline — no Neo4j needed)")
@click.option("--era", "-e", default=None,
              help="Story-time era for context (e.g. 'Third Age')")
@click.option("--category", "-c", multiple=True,
              help="Limit check to specific categories (repeat for multiple)")
@click.option("--neo4j", is_flag=True,
              help="Run Cypher checks via Neo4j (requires --scene-id)")
@click.option("--output", "-o", type=click.Path(), help="Save result to JSON file")
def lore_validate_scene(
    scene_id: str | None,
    text: str | None,
    era: str | None,
    category: tuple[str],
    neo4j: bool,
    output: str | None,
) -> None:
    """Validate a scene against all applicable LoreRules.

    Runs HARD and SOFT rule checks. Hard violations block acceptance;
    soft violations produce warnings.

    Examples:
        bga lore validate-scene --text "An Elf died peacefully of old age."
        bga lore validate-scene --scene-id scene_83ac87fb --neo4j
        bga lore validate-scene --text "Boromir picked up the One Ring." --era "Third Age"
    """
    import json as _json
    from book_graph_analyzer.lore.rules import LoreRuleValidator, LoreRuleRegistry

    validator = LoreRuleValidator(LoreRuleRegistry.from_tolkien_defaults())

    cats = list(category) if category else None

    if not scene_id and not text:
        console.print("[red]Provide --scene-id or --text.[/red]")
        return

    if neo4j and scene_id:
        from book_graph_analyzer.graph.connection import check_neo4j_connection
        if not check_neo4j_connection():
            console.print("[red]Cannot connect to Neo4j.[/red]")
            return
        console.print(f"[bold]Validating scene:[/bold] {scene_id} (via Neo4j Cypher checks)\n")
        result = validator.validate_scene_neo4j(scene_id)
    elif text:
        console.print(f"[bold]Validating text:[/bold] \"{text[:80]}{'...' if len(text)>80 else ''}\"\n")
        result = validator.validate_text(
            text=text,
            scene_id=scene_id or "inline",
            story_era=era,
        )
    elif scene_id:
        console.print("[yellow]No --neo4j flag — running offline heuristic validation.[/yellow]")
        console.print("[dim]For Cypher-based validation, add --neo4j[/dim]\n")
        result = validator.validate_text(scene_id, scene_id, era)
    else:
        console.print("[red]Cannot validate: provide --text or --scene-id --neo4j[/red]")
        return

    # Display result
    if result.passed:
        console.print(f"[bold green]✓ PASS[/bold green]  —  {result.rules_checked} rules checked")
    else:
        console.print(f"[bold red]✗ FAIL[/bold red]  —  {result.rules_checked} rules checked, "
                      f"{len(result.hard_violations)} hard violation(s)")

    if result.hard_violations:
        console.print("\n[bold red]Hard Violations (BLOCKED):[/bold red]")
        for v in result.hard_violations:
            console.print(f"  • [{v.rule_id}] {v.description}")

    if result.soft_warnings:
        console.print("\n[bold yellow]Soft Warnings (allowed):[/bold yellow]")
        for v in result.soft_warnings:
            console.print(f"  ~ [{v.rule_id}] {v.description}")

    if not result.hard_violations and not result.soft_warnings:
        console.print("  No violations detected.")

    if output:
        with open(output, "w", encoding="utf-8") as f:
            _json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]OK[/green] Result saved to {output}")


@lore.command(name="query-passages")
@click.option("--temporal-depth", "-d", default=None,
              help="Minimum temporal depth era (e.g. 'First Age'). "
                   "Returns passages that reference this era or earlier.")
@click.option("--story-era", "-s", default=None,
              help="Filter by story-time era (when the scene occurs).")
@click.option("--limit", "-n", default=20, help="Maximum results (default: 20)")
@click.option("--show-refs", is_flag=True, help="Show REFERENCES_ERA edges per passage")
@click.option("--show-zoom", is_flag=True, help="Show temporal zoom score")
def lore_query_passages(
    temporal_depth: str | None,
    story_era: str | None,
    limit: int,
    show_refs: bool,
    show_zoom: bool,
) -> None:
    """Query passages by temporal depth or story-time era.

    Examples:
        bga lore query-passages --temporal-depth 'First Age'
        bga lore query-passages --temporal-depth 'Before Time' --show-refs
        bga lore query-passages --story-era 'Third Age' -n 10
        bga lore query-passages --temporal-depth 'Second Age' --show-zoom
    """
    from book_graph_analyzer.graph.passage_writer import PassageTemporalWriter
    from book_graph_analyzer.graph.connection import check_neo4j_connection
    from book_graph_analyzer.graph.temporal import canonicalize_era, era_to_order

    if not check_neo4j_connection():
        console.print("[red]Error:[/red] Cannot connect to Neo4j")
        return

    if not temporal_depth and not story_era:
        console.print("[yellow]Provide --temporal-depth or --story-era (or both).[/yellow]")
        return

    writer = PassageTemporalWriter()

    if temporal_depth:
        depth_canonical = canonicalize_era(temporal_depth) or temporal_depth
        console.print(
            f"[bold]Passages with temporal depth ≤ {depth_canonical}[/bold] "
            f"(era order ≤ {era_to_order(depth_canonical)})\n"
        )
        results = writer.query_passages_by_temporal_depth(
            min_era=depth_canonical,
            limit=limit,
            include_references=show_refs,
        )
    else:
        # story-era only — query directly
        era_canonical = canonicalize_era(story_era) or story_era
        console.print(f"[bold]Passages set in story-era: {era_canonical}[/bold]\n")

        from book_graph_analyzer.graph.connection import get_driver
        from book_graph_analyzer.models.era_reference import TemporalZoomResult

        driver = get_driver()
        results = []
        corpus_avg = writer.compute_corpus_avg_depth()

        with driver.session() as session:
            rows = session.run(
                "MATCH (p:Passage {story_era: $era}) RETURN p LIMIT $limit",
                era=era_canonical, limit=limit,
            )
            for row in rows:
                node = dict(row["p"])
                depth_years = node.get("temporal_depth_years_back")
                zoom = None
                if depth_years is not None and corpus_avg and corpus_avg > 0:
                    zoom = depth_years / corpus_avg
                results.append(TemporalZoomResult(
                    passage_id=node.get("id", ""),
                    passage_text=node.get("text", ""),
                    story_era=node.get("story_era"),
                    story_year=node.get("story_year"),
                    temporal_depth_era=node.get("temporal_depth_era"),
                    temporal_depth_years_back=depth_years,
                    era_reference_count=node.get("era_reference_count", 0),
                    temporal_zoom=zoom,
                ))
        driver.close()

    if not results:
        console.print("[yellow]No passages found matching criteria.[/yellow]")
        writer.close()
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Passage ID", style="dim", width=22)
    table.add_column("Story-time", style="cyan", width=18)
    table.add_column("Depth (oldest ref)", style="magenta", width=18)
    table.add_column("Era refs", justify="right", width=8)
    if show_zoom:
        table.add_column("Zoom", justify="right", width=7)
    table.add_column("Text snippet")

    for r in results:
        story_time = r.story_era or "-"
        if r.story_year:
            story_time += f" {r.story_year}"

        depth = r.temporal_depth_era or "-"
        if r.temporal_depth_years_back is not None:
            depth += f" (~{r.temporal_depth_years_back:,.0f}y)"

        snippet = r.passage_text[:60] + "..." if len(r.passage_text) > 60 else r.passage_text

        row_data = [
            r.passage_id,
            story_time,
            depth,
            str(r.era_reference_count),
        ]
        if show_zoom:
            zoom_str = f"{r.temporal_zoom:.2f}x" if r.temporal_zoom is not None else "-"
            row_data.append(zoom_str)
        row_data.append(snippet)

        table.add_row(*row_data)

    console.print(table)
    console.print(f"\n[dim]{len(results)} passage(s) returned[/dim]")

    if show_refs and results:
        console.print("\n[bold]Era References:[/bold]")
        for r in results:
            if r.references:
                console.print(f"\n  [cyan]{r.passage_id}[/cyan]")
                for ref in r.references:
                    yb = f" (~{ref.years_before_story_time:,.0f}y back)" if ref.years_before_story_time else ""
                    console.print(f"    [{ref.reference_type}] {ref.era}{yb}")

    writer.close()


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


@main.command("bootstrap")
@click.argument("input_path", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Write results to JSON file")
@click.option("--neo4j", is_flag=True, help="Write accepted entities to Neo4j graph")
@click.option("--no-llm", is_flag=True, help="Skip LLM canonicalisation (faster, lower accuracy)")
@click.option("--min-frequency", default=2, show_default=True, help="Minimum mentions to consider an entity")
@click.option("--verbose", is_flag=True, default=True, show_default=True)
def bootstrap_cmd(input_path, output, neo4j, no_llm, min_frequency, verbose):
    """Bootstrap canonical entities from text without seed files.

    INPUT_PATH may be a single .txt file or a directory of .txt files.
    """
    from .extract.bootstrap import EntityBootstrapper

    path = Path(input_path)
    texts = sorted(path.glob("*.txt")) if path.is_dir() else [path]

    if not texts:
        console.print("[red]No .txt files found.[/red]")
        return

    bootstrapper = EntityBootstrapper(use_llm=not no_llm)
    bootstrapper.MIN_FREQUENCY = min_frequency

    all_entities = []
    for text_file in texts:
        console.print(f"\n[bold cyan]Processing:[/bold cyan] {text_file.name}")
        text = text_file.read_text(encoding="utf-8", errors="replace")
        result = bootstrapper.bootstrap(text, verbose=verbose)

        console.print(f"  [green]Accepted:[/green] {result.stats['accepted']}  "
                      f"[yellow]Flagged:[/yellow] {result.stats['flagged']}  "
                      f"[dim]Skipped:[/dim] {result.stats['skipped']}")

        # Print top entities
        table = Table(show_header=True, header_style="bold")
        table.add_column("Type", style="cyan", width=10)
        table.add_column("Canonical Name", style="bold")
        table.add_column("Variants", style="dim")
        table.add_column("Freq", justify="right")
        table.add_column("Conf", justify="right")
        table.add_column("Review?", justify="center")

        for entity in (result.entities + result.flagged)[:30]:
            table.add_row(
                entity.entity_type,
                entity.canonical_name or max(entity.variants, key=len),
                ", ".join(entity.variants[:4]),
                str(entity.frequency),
                f"{entity.cluster_confidence:.2f}",
                "[yellow]YES[/yellow]" if entity.needs_review else "[green]no[/green]",
            )
        console.print(table)

        all_entities.extend(result.to_dict_list())

        if neo4j:
            try:
                from .graph.connection import get_driver
                driver = get_driver()
                written = 0
                with driver.session() as s:
                    for e in result.entities:
                        canonical = e.canonical_name or max(e.variants, key=len)
                        label = {
                            "character": "Character", "place": "Place",
                            "object": "Object", "concept": "Entity",
                        }.get(e.entity_type, "Entity")
                        s.run(
                            f"MERGE (n:{label} {{canonical_name: $name}}) "
                            "SET n.aliases = $aliases, n.bootstrap_confidence = $conf, "
                            "n.needs_review = $review, n.source = 'inferred', "
                            "n.mention_count = $freq",
                            name=canonical, aliases=e.variants,
                            conf=e.cluster_confidence, review=e.needs_review,
                            freq=e.frequency,
                        )
                        written += 1
                console.print(f"  [green]Written {written} entities to Neo4j[/green]")
            except Exception as exc:
                console.print(f"  [red]Neo4j write failed: {exc}[/red]")

    if output:
        import json as _json
        Path(output).write_text(_json.dumps(all_entities, indent=2), encoding="utf-8")
        console.print(f"\n[green]Results written to {output}[/green]")


if __name__ == "__main__":
    main()

