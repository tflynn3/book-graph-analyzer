#!/usr/bin/env python3
"""Benchmark routed Hugging Face Inference Provider models on unresolved refs."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import InferenceClient


DEFAULT_INVENTORY = [
    "Aragorn",
    "Arwen",
    "Balin",
    "Bard the Bowman",
    "Beorn",
    "Bifur",
    "Bilbo Baggins",
    "Bofur",
    "Bombur",
    "Boromir",
    "Celeborn",
    "Denethor",
    "Dwalin",
    "Elrond",
    "Faramir",
    "Frodo Baggins",
    "Galadriel",
    "Gandalf",
    "Gimli",
    "Gloin",
    "Gollum",
    "Legolas",
    "Lord of the Eagles",
    "Meriadoc Brandybuck",
    "Morgoth",
    "Ori",
    "Peregrin Took",
    "Radagast",
    "Samwise Gamgee",
    "Saruman",
    "Sauron",
    "Smaug",
    "Thorin Oakenshield",
    "Thranduil",
    "Theoden",
    "Tom Bombadil",
    "Treebeard",
    "William",
    "Witch-king of Angmar",
    "Eomer",
    "Eowyn",
    "Oin",
]

STOPWORDS = {
    "a",
    "an",
    "and",
    "of",
    "the",
    "to",
    "in",
    "on",
    "for",
    "with",
    "mr",
    "mrs",
    "sir",
    "lady",
    "lord",
    "king",
}

ALIAS_HINTS = {
    "warily bilbo": ["Bilbo Baggins"],
    "bilbo": ["Bilbo Baggins"],
    "frodo": ["Frodo Baggins"],
    "strider": ["Aragorn"],
    "the lady": ["Galadriel"],
    "lady": ["Galadriel"],
    "gwaihir": ["Lord of the Eagles"],
    "the shadow": ["Sauron"],
    "his shadow": ["Sauron"],
    "shadow": ["Sauron"],
    "merry": ["Meriadoc Brandybuck"],
    "pippin": ["Peregrin Took"],
    "sam": ["Samwise Gamgee"],
    "samwise": ["Samwise Gamgee"],
    "theoden": ["Theoden"],
    "eomer": ["Eomer"],
    "eowyn": ["Eowyn"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="data/evals/hf_unresolved_character_benchmark.json",
        help="Path to the hand-labeled benchmark JSON.",
    )
    parser.add_argument(
        "--output",
        default="data/evals/hf_provider_unresolved_results.json",
        help="Path to write evaluation results JSON.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Model repo to evaluate. Repeat for multiple models.",
    )
    parser.add_argument(
        "--prompt-prefix",
        default="",
        help="Optional prefix inserted before every user prompt.",
    )
    parser.add_argument("--provider", default="auto")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--candidate-shortlist",
        action="store_true",
        help="Prompt with a retrieved existing-candidate shortlist instead of the full inventory.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=6,
        help="Maximum number of existing candidates to show when shortlist mode is enabled.",
    )
    return parser.parse_args()


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"benchmark at {path} must be a JSON list")
    return data


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower().replace("_", " ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def tokenize(value: str | None) -> list[str]:
    normalized = normalize_text(value)
    return [token for token in normalized.split() if token and token not in STOPWORDS]


def build_unique_token_index(inventory: list[str]) -> dict[str, list[str]]:
    token_map: dict[str, list[str]] = defaultdict(list)
    for name in inventory:
        for token in set(tokenize(name)):
            token_map[token].append(name)
    return {token: names for token, names in token_map.items() if len(names) == 1}


def candidate_shortlist(
    example: dict[str, Any],
    inventory: list[str],
    unique_token_index: dict[str, list[str]],
    limit: int,
) -> list[str]:
    mention_norm = normalize_text(example["mention"])
    context_norm = normalize_text(example["context"])
    mention_tokens = tokenize(example["mention"])
    context_tokens = set(tokenize(example["context"]))
    scored: dict[str, float] = {}

    def bump(name: str, points: float) -> None:
        if name in inventory:
            scored[name] = scored.get(name, 0.0) + points

    for alias_key, names in ALIAS_HINTS.items():
        if mention_norm == alias_key or alias_key in mention_norm:
            for name in names:
                bump(name, 100.0)

    for name in inventory:
        name_norm = normalize_text(name)
        name_tokens = set(tokenize(name))
        if mention_norm == name_norm:
            bump(name, 80.0)
        elif mention_norm and mention_norm in name_norm:
            bump(name, 35.0)
        elif name_norm and name_norm in mention_norm:
            bump(name, 25.0)

        overlap = len(set(mention_tokens) & name_tokens)
        if overlap:
            bump(name, overlap * 15.0)

        context_overlap = len(context_tokens & name_tokens)
        if context_overlap:
            bump(name, min(context_overlap, 2) * 3.0)

    for token in mention_tokens:
        for name in unique_token_index.get(token, []):
            bump(name, 20.0)

    ranked = sorted(scored.items(), key=lambda item: (-item[1], item[0]))
    filtered = [name for name, score in ranked if score >= 10.0]
    return filtered[:limit]


def build_prompt(
    example: dict[str, Any],
    inventory: list[str],
    prefix: str,
    shortlist: list[str] | None,
) -> str:
    if shortlist:
        candidate_block = ", ".join(shortlist)
        inventory_section = (
            "Retrieved existing-character candidates from the graph:\n"
            f"{candidate_block}\n\n"
            "If ACTION=existing, ENTITY must be one of the candidate names above exactly.\n"
            "The surface mention may be an alias, epithet, title, or nickname of a candidate.\n"
            "If a candidate is the canonical identity behind the mention, choose existing with that exact candidate name.\n"
            "If none of those candidates fit, choose new_entity or reject.\n"
            "Reject plural groups, families, places, and objects even if they resemble a character name.\n\n"
        )
    else:
        inventory_block = ", ".join(inventory)
        inventory_section = (
            "Known graph character inventory:\n"
            f"{inventory_block}\n\n"
        )

    prompt = (
        "Resolve one LOTR mention for a graph grounding task.\n"
        f"{inventory_section}"
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
    return f"{prefix}\n{prompt}" if prefix else prompt


def parse_response(content: str | None, reasoning_content: str | None) -> dict[str, str]:
    text = "\n".join(part for part in [content, reasoning_content] if part)
    if not text.strip():
        return {"action": "reject", "entity": ""}

    match = re.search(
        r"ACTION\s*=\s*(existing|new_entity|reject)\s*;\s*ENTITY\s*=\s*([^\n\r]+)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        action = match.group(1).lower()
        entity = match.group(2).strip()
        if normalize_text(entity) in {"", "none"}:
            entity = ""
        return {"action": action, "entity": entity}

    action_match = re.search(r"\b(existing|new_entity|reject)\b", text, flags=re.IGNORECASE)
    action = action_match.group(1).lower() if action_match else "reject"
    entity = ""
    entity_match = re.search(r"ENTITY\s*[:=]\s*([^\n\r]+)", text, flags=re.IGNORECASE)
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
    full_correct = entity_correct or (not entity_required and action_correct)
    return {
        "action_correct": action_correct,
        "entity_required": entity_required,
        "entity_correct": entity_correct,
        "full_correct": full_correct,
        "predicted_entity_normalized": predicted_norm,
    }


def aggregate_metrics(examples: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(scored)
    action_correct = sum(1 for item in scored if item["score"]["action_correct"])
    entity_needed = [item for item in scored if item["score"]["entity_required"]]
    entity_correct = sum(1 for item in entity_needed if item["score"]["entity_correct"])
    full_correct = sum(1 for item in scored if item["score"]["full_correct"])

    by_action: dict[str, dict[str, int]] = {}
    for example, item in zip(examples, scored):
        action = example["gold_action"]
        bucket = by_action.setdefault(action, {"total": 0, "action_correct": 0, "full_correct": 0})
        bucket["total"] += 1
        bucket["action_correct"] += int(item["score"]["action_correct"])
        bucket["full_correct"] += int(item["score"]["full_correct"])

    metrics = {
        "total_examples": total,
        "action_accuracy": action_correct / total if total else 0.0,
        "entity_accuracy_on_required": entity_correct / len(entity_needed) if entity_needed else 0.0,
        "full_accuracy": full_correct / total if total else 0.0,
        "by_action": {},
    }
    for action, bucket in by_action.items():
        metrics["by_action"][action] = {
            "total": bucket["total"],
            "action_accuracy": bucket["action_correct"] / bucket["total"] if bucket["total"] else 0.0,
            "full_accuracy": bucket["full_correct"] / bucket["total"] if bucket["total"] else 0.0,
        }
    return metrics


def evaluate_model(
    client: InferenceClient,
    model_name: str,
    examples: list[dict[str, Any]],
    inventory: list[str],
    prompt_prefix: str,
    max_tokens: int,
    temperature: float,
    use_candidate_shortlist: bool,
    candidate_limit: int,
) -> dict[str, Any]:
    start = time.time()
    scored: list[dict[str, Any]] = []
    unique_token_index = build_unique_token_index(inventory)
    for index, example in enumerate(examples, start=1):
        shortlist = (
            candidate_shortlist(example, inventory, unique_token_index, candidate_limit)
            if use_candidate_shortlist
            else None
        )
        prompt = build_prompt(example, inventory, prompt_prefix, shortlist)
        output = client.chat_completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        message = output.choices[0].message
        parsed = parse_response(
            getattr(message, "content", None),
            getattr(message, "reasoning_content", None),
        )
        score = score_prediction(example, parsed)
        scored.append(
            {
                "id": example["id"],
                "mention": example["mention"],
                "gold_action": example["gold_action"],
                "gold_entity": example.get("gold_entity"),
                "shortlist": shortlist,
                "raw_content": getattr(message, "content", None),
                "raw_reasoning_content": getattr(message, "reasoning_content", None),
                "parsed": parsed,
                "score": score,
            }
        )
        print(f"{model_name} {index:02d}/{len(examples)} {example['mention']}: {parsed}", flush=True)

    metrics = aggregate_metrics(examples, scored)
    metrics["elapsed_seconds"] = time.time() - start
    return {"model": model_name, "metrics": metrics, "predictions": scored}


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    examples = load_benchmark(benchmark_path)
    if args.limit > 0:
        examples = examples[: args.limit]

    client = InferenceClient(provider=args.provider, timeout=args.timeout)
    results = {
        "benchmark_path": str(benchmark_path),
        "provider": args.provider,
        "inventory_size": len(DEFAULT_INVENTORY),
        "models": [],
        "timestamp": time.time(),
    }

    for model_name in args.models:
        print(f"=== Evaluating {model_name} on {len(examples)} examples ===", flush=True)
        model_result = evaluate_model(
            client=client,
            model_name=model_name,
            examples=examples,
            inventory=DEFAULT_INVENTORY,
            prompt_prefix=args.prompt_prefix,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            use_candidate_shortlist=args.candidate_shortlist,
            candidate_limit=args.candidate_limit,
        )
        metrics = model_result["metrics"]
        print(
            f"{model_name}: action={metrics['action_accuracy']:.3f}, "
            f"full={metrics['full_accuracy']:.3f}, "
            f"entity={metrics['entity_accuracy_on_required']:.3f}",
            flush=True,
        )
        results["models"].append(model_result)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
