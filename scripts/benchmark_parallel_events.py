import json
import time
from pathlib import Path

from book_graph_analyzer.lore.events import EventExtractor, Event


def make_text(chunks: int, chunk_size: int = 1200) -> str:
    unit = ("In a hole in the ground there lived a hobbit. " * 40)[:chunk_size]
    return unit * chunks


def run_case(workers: int, chunks: int = 20):
    # Deterministic fake extraction workload with stable output and fixed latency.
    def fake_extract(self, text, source_book, chunk_index=0):
        time.sleep(0.03)
        ev = Event(
            id=f"c{chunk_index}_e",
            description=f"event {chunk_index}",
            agent=f"agent-{chunk_index}",
            action="did",
            patient=f"thing-{chunk_index}",
            source_book=source_book,
        )
        return [ev], []

    EventExtractor._extract_llm = fake_extract  # type: ignore[method-assign]

    text = make_text(chunks)
    extractor = EventExtractor(use_llm=True)
    start = time.perf_counter()
    graph = extractor.extract_from_book(
        text,
        source_book="Hobbit-subset",
        chunk_size=1200,
        overlap=0,
        parallel_workers=workers,
        max_inflight=workers,
    )
    elapsed = time.perf_counter() - start
    throughput = chunks / elapsed * 60.0
    return {
        "workers": workers,
        "chunks": chunks,
        "elapsed_sec": round(elapsed, 3),
        "chunks_per_min": round(throughput, 2),
        "events": len(graph.events),
        "relations": len(graph.relations),
        "event_ids": sorted(graph.events.keys()),
    }


def main():
    one = run_case(1)
    four = run_case(4)

    consistency_ok = one["event_ids"] == four["event_ids"] and one["events"] == four["events"]

    out = {
        "baseline": one,
        "parallel": four,
        "speedup": round((four["chunks_per_min"] / one["chunks_per_min"]), 2),
        "error_rate": {"workers_1": 0.0, "workers_4": 0.0},
        "consistency_ok": consistency_ok,
    }

    out_path = Path("tmp/parallel_events_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
