import json
import re

import jsonschema

from tasks.generate_tasks import TOOLS
from tasks.generate_tasks_v2 import generate_v2


def _load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_v2_generation_is_byte_reproducible(tmp_path):
    p1, p2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    generate_v2(p1)
    generate_v2(p2)
    assert p1.read_bytes() == p2.read_bytes()


def test_v2_counts_and_id_ranges(tmp_path):
    tasks = generate_v2(tmp_path / "t.jsonl")
    fams = {}
    for t in tasks:
        fams.setdefault(t["family"], []).append(t["prompt_id"])
    assert len(tasks) == 1200
    assert len(fams["structured_output"]) == 700
    assert len(fams["tool_call"]) == 300
    assert len(fams["tool_decline"]) == 200
    assert len({t["prompt_id"] for t in tasks}) == 1200
    # contiguous ids
    assert fams["structured_output"] == [f"struct-{i:04d}" for i in range(700)]
    assert fams["tool_call"] == [f"tool-{i:04d}" for i in range(300)]
    assert fams["tool_decline"] == [f"decline-{i:04d}" for i in range(200)]


def test_v1_subset_byte_identical_to_v1_tasks(tmp_path):
    """The 500 v1 records inside tasks_v2.jsonl must be byte-identical to the
    committed v1 tasks.jsonl — the whole point of the tagged subset."""
    v2 = {t["prompt_id"]: t for t in generate_v2(tmp_path / "t.jsonl")}
    v1_lines = open("tasks/tasks.jsonl", encoding="utf-8").read().splitlines()
    assert len(v1_lines) == 500
    for line in v1_lines:
        pid = json.loads(line)["prompt_id"]
        v2_line = json.dumps(v2[pid], sort_keys=True, ensure_ascii=True,
                             separators=(",", ":"))
        assert v2_line == line, f"v1 record drifted: {pid}"


def test_new_schemas_valid_and_depths_covered(tmp_path):
    tasks = [t for t in generate_v2(tmp_path / "t.jsonl")
             if t["family"] == "structured_output"
             and int(t["prompt_id"].split("-")[1]) >= 350]

    def depth(s):
        if s.get("type") != "object":
            return 0
        return 1 + max(depth(p) for p in s["properties"].values())

    assert {depth(t["schema"]) for t in tasks} == {1, 2, 3, 4}
    for t in tasks:
        jsonschema.Draft202012Validator.check_schema(t["schema"])


def test_new_decline_requests_do_not_match_tools(tmp_path):
    """No new decline request may contain phrasing any pool tool satisfies."""
    tool_words = re.compile(
        r"\b(weather|email|calendar|event|currency|convert|flights?|remind\w*|"
        r"translat\w*|stocks?|ticker|restaurant|table|to.?do|task list|"
        r"directions?|tip|bill)\b", re.I)
    tasks = [t for t in generate_v2(tmp_path / "t.jsonl")
             if t["family"] == "tool_decline"
             and int(t["prompt_id"].split("-")[1]) >= 30]
    assert len(tasks) == 170
    for t in tasks:
        request = t["prompt"].split("User request: ")[1].split("\n")[0]
        assert not tool_words.search(request), f"{t['prompt_id']}: {request}"
        assert t["gold"] == {"tool": None}


def test_new_tool_tasks_verbatim_args(tmp_path):
    tasks = [t for t in generate_v2(tmp_path / "t.jsonl")
             if t["family"] == "tool_call"
             and int(t["prompt_id"].split("-")[1]) >= 120]
    assert len(tasks) == 180
    for t in tasks:
        assert t["gold"]["tool"] in TOOLS
        assert f'"name": "{t["gold"]["tool"]}"' in t["prompt"]
        for v in t["gold"]["arguments"].values():
            assert str(v) in t["prompt"]
