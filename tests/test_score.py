import json
import sys

from scorer.score import METRIC_NAMES, score_results


def _write(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _result(prompt_id, raw_output, quant="Q8_0", seed=0):
    return {"model": "m", "quant": quant, "prompt_id": prompt_id,
            "seed": seed, "raw_output": raw_output, "pipeline_version": 1}


def test_score_results(tmp_path):
    tasks_path = tmp_path / "tasks.jsonl"
    with open(tasks_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt_id": "struct-0000", "family": "structured_output",
            "prompt": "", "validator_id": "structured_output", "gold": None,
            "schema": {"type": "object",
                       "properties": {"a": {"type": "integer"}},
                       "required": ["a"], "additionalProperties": False},
        }) + "\n")
        f.write(json.dumps({
            "prompt_id": "tool-0000", "family": "tool_call",
            "prompt": "", "validator_id": "tool_call", "schema": None,
            "gold": {"tool": "get_weather",
                     "arguments": {"city": "Paris", "units": "celsius"}},
        }) + "\n")

    results = tmp_path / "results"
    results.mkdir()
    _write(results / "r1.json", _result("struct-0000", '{"a": 3}'))
    _write(results / "r2.json", _result("struct-0000", 'not json', seed=1))
    _write(results / "r3.json", _result(
        "tool-0000",
        '{"tool": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}'))

    df, n_files = score_results(results, tasks_path)

    assert n_files == 3
    def val(pid, seed, metric):
        row = df[(df.prompt_id == pid) & (df.seed == seed)
                 & (df.metric == metric)]
        return row["value"].iloc[0]

    assert val("struct-0000", 0, "json_parse_rate") == 1.0
    assert val("struct-0000", 0, "schema_compliance") == 1.0
    assert val("struct-0000", 1, "json_parse_rate") == 0.0
    assert val("struct-0000", 1, "schema_compliance") == 0.0
    assert val("tool-0000", 0, "tool_selection_acc") == 1.0
    assert val("tool-0000", 0, "argument_exact_match") == 1.0
    assert set(df.columns) == {"model", "quant", "prompt_id", "family",
                               "seed", "pipeline_version", "metric", "value"}
    assert set(df.metric).issubset(set(METRIC_NAMES.values()))


def test_scoring_never_imports_runner():
    assert not any(m == "runner" or m.startswith("runner.")
                   for m in sys.modules), \
        "scorer must not import the runner (CLAUDE.md rule 6)"
