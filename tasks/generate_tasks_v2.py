"""Dataset v2 -> tasks/tasks_v2.jsonl (1,200 tasks, byte-reproducible).

Composition (QUANTONE_V2_PLAN Phase 1):
  - 700 schema-constrained tasks   (v1's struct-0000..0349 + new 0350..0699)
  - 300 tool-selection tasks       (v1's tool-0000..0119   + new 0120..0299)
  - 200 should-not-call tasks      (v1's decline-0000..0029 + new 0030..0199)
    <- the power fix: v1's n=30 decline subset gave ~±17pt CIs; n=200
       brings the CI to roughly ±5-8 pts.

The v1 500 are regenerated through the UNCHANGED v1 generator with its
original seed, so their records are byte-identical to v1's tasks.jsonl and
task ids are preserved — the v1 subset is identified by id ranges (documented
here), not by mutating records. New tasks draw from a SEPARATE RNG stream
(V2_SEED) so extending the dataset can never perturb v1 bytes.

Validators are v1's, unchanged.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasks.generate_tasks import (DECLINE_REQUESTS, GENERATION_SEED,
                                  STRUCTURED_PROMPT, TOOL_PROMPT, TOOLS,
                                  _object_schema, _tool_defs_json,
                                  make_decline_tasks, make_structured_tasks,
                                  make_tool_tasks)

V2_SEED = 20260712
N_STRUCT_TOTAL, N_TOOL_TOTAL, N_DECLINE_TOTAL = 700, 300, 200
V1_STRUCT, V1_TOOL, V1_DECLINE = 350, 120, 30

# ---------------------------------------------------------------------------
# New should-not-call requests: generated from a template grid, topics chosen
# to be unsatisfiable by every tool in the pool (no weather/email/calendar/
# currency/flight/reminder/translation/stock/restaurant/todo/directions/tip
# phrasing). 170 unique requests, disjoint from v1's 30 hand-written ones.
# ---------------------------------------------------------------------------

DECLINE_TEMPLATES = [
    "Write a {form} about {topic}.",
    "Explain how {process} works in simple terms.",
    "What is the {superlative} {thing} in {place}?",
    "Summarize the history of {topic} in two sentences.",
    "Give me three interesting facts about {topic}.",
    "Who is famous for {achievement}?",
    "Describe the difference between {a} and {b}.",
    "Compose a short {form} mentioning {topic}.",
    "Why is {phenomenon} the case?",
    "List five examples of {category}.",
]

FILLERS = {
    "form": ["haiku", "limerick", "sonnet", "riddle", "fable", "toast",
             "lullaby", "proverb"],
    "topic": ["volcanoes", "honeybees", "the printing press", "jazz",
              "glaciers", "chess openings", "the Silk Road", "origami",
              "meteor showers", "coral reefs", "windmills", "calligraphy",
              "fermentation", "lighthouses", "marathon running"],
    "process": ["photosynthesis", "cloud formation", "cheese aging",
                "echolocation", "paper recycling", "glass blowing",
                "beekeeping", "tide generation"],
    "superlative": ["tallest", "oldest", "deepest", "fastest", "largest",
                    "smallest"],
    "thing": ["waterfall", "library", "bridge", "desert", "cave system",
              "tree", "mountain range", "museum"],
    "place": ["South America", "Scandinavia", "Southeast Asia", "Africa",
              "the Pacific", "Eastern Europe"],
    "achievement": ["discovering penicillin", "composing nine symphonies",
                    "painting the Sistine Chapel ceiling",
                    "inventing the telephone", "writing One Hundred Years of "
                    "Solitude", "first summiting Everest"],
    "a": ["a violin", "igneous rock", "a comet", "baroque architecture",
          "a crocodile", "fission"],
    "b": ["a viola", "sedimentary rock", "an asteroid", "gothic architecture",
          "an alligator", "fusion"],
    "phenomenon": ["the sky blue", "the ocean salty", "autumn foliage red",
                   "thunder audible after lightning", "the moon tidally "
                   "locked", "honey resistant to spoiling"],
    "category": ["nocturnal animals", "renewable energy sources",
                 "ancient wonders", "string instruments",
                 "bioluminescent creatures", "palindromic words"],
}


def make_new_decline_requests(rng, n):
    import re
    seen = set(DECLINE_REQUESTS)
    out = []
    while len(out) < n:
        t = rng.choice(DECLINE_TEMPLATES)
        req = t.format(**{k: rng.choice(v) for k, v in FILLERS.items()
                          if "{" + k + "}" in t})
        if req not in seen:
            seen.add(req)
            out.append(req)
    return out


# ---------------------------------------------------------------------------
# Extension generators: same building blocks and record shape as v1, index
# ranges continuing where v1 stopped, fed by the v2 RNG stream.
# ---------------------------------------------------------------------------

def make_new_structured(rng):
    tasks = []
    for i in range(V1_STRUCT, N_STRUCT_TOTAL):
        depth = (i % 4) + 1
        schema = _object_schema(rng, depth)
        tasks.append({
            "prompt_id": f"struct-{i:04d}",
            "family": "structured_output",
            "prompt": STRUCTURED_PROMPT.format(
                schema=json.dumps(schema, sort_keys=True, indent=2)),
            "schema": schema,
            "gold": None,
            "validator_id": "structured_output",
        })
    return tasks


def make_new_tool(rng):
    tasks = []
    tool_names = sorted(TOOLS)
    for i in range(V1_TOOL, N_TOOL_TOTAL):
        correct = tool_names[i % len(tool_names)]
        n_distractors = rng.randint(2, 5)
        distractors = rng.sample([n for n in tool_names if n != correct],
                                 n_distractors)
        shown = [correct] + distractors
        rng.shuffle(shown)
        args = TOOLS[correct]["make"](rng)
        request = TOOLS[correct]["template"].format(**args)
        tasks.append({
            "prompt_id": f"tool-{i:04d}",
            "family": "tool_call",
            "prompt": TOOL_PROMPT.format(tools=_tool_defs_json(shown),
                                         request=request),
            "schema": None,
            "gold": {"tool": correct, "arguments": args},
            "validator_id": "tool_call",
        })
    return tasks


def make_new_decline(rng):
    tasks = []
    tool_names = sorted(TOOLS)
    requests = make_new_decline_requests(rng, N_DECLINE_TOTAL - V1_DECLINE)
    for j, request in enumerate(requests):
        i = V1_DECLINE + j
        shown = rng.sample(tool_names, rng.randint(2, 5))
        tasks.append({
            "prompt_id": f"decline-{i:04d}",
            "family": "tool_decline",
            "prompt": TOOL_PROMPT.format(tools=_tool_defs_json(shown),
                                         request=request),
            "schema": None,
            "gold": {"tool": None},
            "validator_id": "tool_decline",
        })
    return tasks


def generate_v2(out_path):
    # v1 subset: identical stream, identical order, identical bytes
    rng_v1 = random.Random(GENERATION_SEED)
    v1 = (make_structured_tasks(rng_v1) + make_tool_tasks(rng_v1)
          + make_decline_tasks(rng_v1))
    assert len(v1) == 500

    rng_v2 = random.Random(V2_SEED)
    new = make_new_structured(rng_v2) + make_new_tool(rng_v2) \
        + make_new_decline(rng_v2)
    assert len(new) == 700

    # interleave by family so ids sort naturally: all struct, all tool, all decline
    by_family = {"structured_output": [], "tool_call": [], "tool_decline": []}
    for t in v1 + new:
        by_family[t["family"]].append(t)
    tasks = (sorted(by_family["structured_output"], key=lambda t: t["prompt_id"])
             + sorted(by_family["tool_call"], key=lambda t: t["prompt_id"])
             + sorted(by_family["tool_decline"], key=lambda t: t["prompt_id"]))
    assert len(tasks) == 1200

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, sort_keys=True, ensure_ascii=True,
                               separators=(",", ":")) + "\n")
    return tasks


def _schema_depth(schema):
    if not isinstance(schema, dict) or schema.get("type") != "object":
        return 0
    return 1 + max(_schema_depth(p) for p in schema["properties"].values())


def _leaf_kinds(schema, acc):
    for p in schema["properties"].values():
        if p.get("type") == "object":
            _leaf_kinds(p, acc)
        elif p.get("type") == "array":
            acc["array"] += 1
        elif "enum" in p:
            acc["enum"] += 1
        elif p.get("pattern"):
            acc["date"] += 1
        else:
            acc[p.get("type", "?")] += 1


def distribution_report(tasks, path):
    fam = Counter(t["family"] for t in tasks)
    depths = Counter(_schema_depth(t["schema"]) for t in tasks
                     if t["family"] == "structured_output")
    kinds = Counter()
    for t in tasks:
        if t["family"] == "structured_output":
            _leaf_kinds(t["schema"], kinds)
    n_tools = Counter(t["prompt"].count('"name":') for t in tasks
                      if t["family"] in ("tool_call", "tool_decline"))
    lines = ["# Dataset v2 distribution report", "",
             f"Total tasks: {len(tasks)}", "",
             "## Families", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(fam.items())]
    lines += ["", "## Schema nesting depth (structured tasks)", ""]
    lines += [f"- depth {k}: {v}" for k, v in sorted(depths.items())]
    lines += ["", "## Leaf field kinds (all structured schemas)", ""]
    lines += [f"- {k}: {v}" for k, v in sorted(kinds.items())]
    lines += ["", "## Tools shown per tool/decline prompt", ""]
    lines += [f"- {k} tools: {v}" for k, v in sorted(n_tools.items())]
    lines += ["", "## v1 subset (byte-identical to v1 tasks.jsonl)", "",
              "- struct-0000..0349, tool-0000..0119, decline-0000..0029", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    tasks = generate_v2(Path(__file__).parent / "tasks_v2.jsonl")
    distribution_report(tasks, Path(__file__).parents[1]
                        / "phase1_distribution_report.md")
    fam = Counter(t["family"] for t in tasks)
    print(f"wrote {len(tasks)} tasks: {dict(fam)}")
    print("distribution report -> phase1_distribution_report.md")
