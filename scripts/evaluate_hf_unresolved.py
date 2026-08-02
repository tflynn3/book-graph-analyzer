#!/usr/bin/env python3
"""Benchmark public Hugging Face models on unresolved LOTR character references."""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler
from neo4j import GraphDatabase


DEFAULT_MODELS = [
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/Phi-4-mini-instruct-4bit",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="data/evals/hf_unresolved_character_benchmark.json",
        help="Path to the hand-labeled benchmark JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/evals/hf_unresolved_results.json",
        help="Path to write evaluation results JSON.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model repo to evaluate. Repeat to override defaults.",
    )
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="bookgraph123")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit benchmark items for smoke tests.")
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"benchmark at {path} must be a JSON list")
    return data


def load_character_inventory(uri: str, user: str, password: str) -> list[str]:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = "MATCH (c:Character) RETURN c.canonical_name AS name ORDER BY name"
    with driver.session() as session:
        names = [row["name"] for row in session.run(query) if row["name"]]
    driver.close()
    return names


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = normalized.replace("_", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def build_prompt(example: dict[str, Any], inventory: list[str]) -> str:
    inventory_block = ", ".join(inventory)
    return (
        "Resolve one LOTR mention for a graph grounding task.\n"
        "Known graph character inventory:\n"
        f"{inventory_block}\n\n"
        "Choose exactly one action:\n"
        "- existing: the mention refers to one of the known graph characters above. "
        "Use the exact inventory name.\n"
        "- new_entity: the mention refers to a character or person not in the known graph inventory. "
        "Return the best canonical character name.\n"
        "- reject: the mention is not a character/person reference, or there is not enough evidence.\n\n"
        "Return exactly one line in this format:\n"
        "ACTION=<existing|new_entity|reject>; ENTITY=<name or NONE>\n\n"
        f"Mention: {example['mention']}\n"
        f"Reference class: {example['reference_class']}\n"
        "Expected type: character\n"
        f"Context: {example['context']}\n"
    )


def render_model_prompt(prompt: str, tokenizer: Any) -> str | list[int]:
    if getattr(tokenizer, "has_chat_template", False):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    return prompt


def parse_response(text: str) -> dict[str, str]:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    match = re.search(
        r"ACTION\s*=\s*(existing|new_entity|reject)\s*;\s*ENTITY\s*=\s*(.+)$",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        action = match.group(1).lower()
        entity = match.group(2).strip()
        if entity.upper() == "NONE":
            entity = ""
        return {"action": action, "entity": entity}

    action_match = re.search(r"\b(existing|new_entity|reject)\b", text, flags=re.IGNORECASE)
    action = action_match.group(1).lower() if action_match else "reject"
    entity = ""
    entity_match = re.search(r"ENTITY\s*[:=]\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    if entity_match:
        entity = entity_match.group(1).strip()
    return {"action": action, "entity": entity}


def score_prediction(example: dict[str, Any], prediction: dict[str, str]) -> dict[str, Any]:
    gold_action = example["gold_action"]
    gold_entity = example.get("gold_entity")
    acceptable = [gold_entity] if gold_entity else []
    if gold_action == "existing" and gold_entity == "Lord of the Eagles":
        acceptable.append("Gwaihir")
    if gold_action == "new_entity":
        aliases = {
            "Adelard Took": ["Adelard"],
            "Folco Boffin": ["Folco"],
            "Farmer Maggot": ["Maggot"],
            "Barliman Butterbur": ["Butterbur", "Mr Butterbur"],
            "Thror": ["Thror", "Thror son of Dain"],
        }
        acceptable.extend(aliases.get(gold_entity, []))

    acceptable_norm = {normalize_text(name) for name in acceptable if name}
    predicted_norm = normalize_text(prediction["entity"])
    action_correct = prediction["action"] == gold_action
    entity_required = gold_action in {"existing", "new_entity"}
    entity_correct = action_correct and (
        not entity_required or (predicted_norm and predicted_norm in acceptable_norm)
    )

    return {
        "action_correct": action_correct,
        "entity_required": entity_required,
        "entity_correct": entity_correct,
        "acceptable_entities": sorted(acceptable_norm),
        "predicted_entity_normalized": predicted_norm,
    }


def aggregate_metrics(examples: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    action_correct = sum(1 for item in scored if item["score"]["action_correct"])
    entity_needed = [item for item in scored if item["score"]["entity_required"]]
    entity_correct = sum(1 for item in entity_needed if item["score"]["entity_correct"])

    by_action: dict[str, dict[str, int]] = {}
    for example, item in zip(examples, scored):
        action = example["gold_action"]
        bucket = by_action.setdefault(action, {"total": 0, "action_correct": 0, "entity_correct": 0})
        bucket["total"] += 1
        bucket["action_correct"] += int(item["score"]["action_correct"])
        bucket["entity_correct"] += int(item["score"]["entity_correct"])

    metrics = {
        "total_examples": total,
        "action_accuracy": action_correct / total if total else 0.0,
        "entity_accuracy_on_required": entity_correct / len(entity_needed) if entity_needed else 0.0,
        "full_accuracy": sum(1 for item in scored if item["score"]["entity_correct"] or (
            not item["score"]["entity_required"] and item["score"]["action_correct"]
        )) / total if total else 0.0,
        "by_action": {},
    }
    for action, bucket in by_action.items():
        metrics["by_action"][action] = {
            "total": bucket["total"],
            "action_accuracy": bucket["action_correct"] / bucket["total"] if bucket["total"] else 0.0,
            "full_accuracy": bucket["entity_correct"] / bucket["total"] if bucket["total"] else 0.0,
        }
    return metrics


def evaluate_model(
    model_name: str,
    examples: list[dict[str, Any]],
    inventory: list[str],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    start = time.time()
    model, tokenizer = load(model_name)
    sampler = make_sampler(temp=temperature)
    load_seconds = time.time() - start

    scored: list[dict[str, Any]] = []
    for example in examples:
        prompt = build_prompt(example, inventory)
        formatted_prompt = render_model_prompt(prompt, tokenizer)
        response = generate(
            model,
            tokenizer,
            formatted_prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=False,
        )
        parsed = parse_response(response)
        score = score_prediction(example, parsed)
        scored.append(
            {
                "id": example["id"],
                "mention": example["mention"],
                "gold_action": example["gold_action"],
                "gold_entity": example.get("gold_entity"),
                "raw_response": response.strip(),
                "parsed": parsed,
                "score": score,
            }
        )

    metrics = aggregate_metrics(examples, scored)
    metrics["load_seconds"] = load_seconds
    metrics["inference_seconds"] = time.time() - start - load_seconds

    del model
    del tokenizer
    gc.collect()

    return {
        "model": model_name,
        "metrics": metrics,
        "predictions": scored,
    }


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    examples = load_benchmark(benchmark_path)
    if args.limit > 0:
        examples = examples[: args.limit]

    inventory = load_character_inventory(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    models = args.models or DEFAULT_MODELS

    results = {
        "benchmark_path": str(benchmark_path),
        "models": [],
        "inventory_size": len(inventory),
        "timestamp": time.time(),
    }

    for model_name in models:
        print(f"=== Evaluating {model_name} on {len(examples)} examples ===")
        try:
            model_result = evaluate_model(
                model_name=model_name,
                examples=examples,
                inventory=inventory,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:
            results["models"].append(
                {
                    "model": model_name,
                    "error": str(exc),
                }
            )
            print(f"{model_name}: ERROR {exc}")
            continue
        results["models"].append(model_result)
        metrics = model_result["metrics"]
        print(
            f"{model_name}: action={metrics['action_accuracy']:.3f}, "
            f"full={metrics['full_accuracy']:.3f}, "
            f"entity={metrics['entity_accuracy_on_required']:.3f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
