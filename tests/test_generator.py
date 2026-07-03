import json

import jsonschema

from tasks.generate_tasks import TOOLS, generate


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_generation_is_byte_reproducible(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    generate(p1)
    generate(p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_counts_and_ids(tmp_path):
    tasks = generate(tmp_path / "t.jsonl")
    by_family = {}
    for t in tasks:
        by_family.setdefault(t["family"], []).append(t)
    assert len(tasks) == 500
    assert len(by_family["structured_output"]) == 350
    assert len(by_family["tool_call"]) == 120
    assert len(by_family["tool_decline"]) == 30
    assert len({t["prompt_id"] for t in tasks}) == 500
    for t in tasks:
        assert set(t) == {"prompt_id", "family", "prompt", "schema", "gold",
                          "validator_id"}


def test_structured_schemas_are_valid_and_cover_depths(tmp_path):
    tasks = [t for t in generate(tmp_path / "t.jsonl")
             if t["family"] == "structured_output"]

    def depth(schema):
        if schema.get("type") != "object":
            return 0
        return 1 + max(depth(p) for p in schema["properties"].values())

    depths = set()
    for t in tasks:
        jsonschema.Draft202012Validator.check_schema(t["schema"])
        depths.add(depth(t["schema"]))
        assert t["schema"]["additionalProperties"] is False
    assert depths == {1, 2, 3, 4}


def test_tool_tasks_reference_shown_tools_and_verbatim_args(tmp_path):
    tasks = generate(tmp_path / "t.jsonl")
    for t in tasks:
        if t["family"] == "tool_call":
            gold = t["gold"]
            assert gold["tool"] in TOOLS
            assert f'"name": "{gold["tool"]}"' in t["prompt"]
            for v in gold["arguments"].values():
                assert str(v) in t["prompt"]  # values stated verbatim
        elif t["family"] == "tool_decline":
            assert t["gold"] == {"tool": None}
