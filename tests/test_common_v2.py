import json

from common_v2 import (cell_key, is_v1_subset, plan_cells,
                       thinking_off_messages)

TASKS = [
    {"prompt_id": "struct-0000", "family": "structured_output"},
    {"prompt_id": "struct-0400", "family": "structured_output"},  # not in v1-500
    {"prompt_id": "tool-0000", "family": "tool_call"},
    {"prompt_id": "tool-0200", "family": "tool_call"},            # not in v1-500
    {"prompt_id": "decline-0000", "family": "tool_decline"},
    {"prompt_id": "decline-0100", "family": "tool_decline"},      # not in v1-500
]


def test_v1_subset_boundaries():
    assert is_v1_subset("struct-0349") and not is_v1_subset("struct-0350")
    assert is_v1_subset("tool-0119") and not is_v1_subset("tool-0120")
    assert is_v1_subset("decline-0029") and not is_v1_subset("decline-0030")


def test_seed_knob_by_family():
    """K-quant (full 1200, not subset-restricted): 2 seeds for struct/tool,
    3 for decline."""
    cells = list(plan_cells("m", "Q4_K_M", "f16", TASKS))
    by_pid = {}
    for t, s in cells:
        by_pid.setdefault(t["prompt_id"], []).append(s)
    assert by_pid["struct-0000"] == [0, 1]
    assert by_pid["tool-0200"] == [0, 1]
    assert by_pid["decline-0100"] == [0, 1, 2]


def test_iq_quant_restricted_to_v1_subset():
    cells = list(plan_cells("m", "IQ3_M", "f16", TASKS))
    pids = {t["prompt_id"] for t, _ in cells}
    assert "struct-0000" in pids and "struct-0400" not in pids
    assert "tool-0000" in pids and "tool-0200" not in pids
    assert "decline-0000" in pids and "decline-0100" not in pids


def test_iq4xs_also_restricted():
    pids = {t["prompt_id"] for t, _ in plan_cells("m", "IQ4_XS", "f16", TASKS)}
    assert pids == {"struct-0000", "tool-0000", "decline-0000"}


def test_cell_key_includes_kv_type():
    a = cell_key("m", "Q4_K_M", "f16", "struct-0000", 0)
    b = cell_key("m", "Q4_K_M", "q4_0", "struct-0000", 0)
    assert a != b and len(a) == 40


def test_thinking_off_messages():
    assert thinking_off_messages("none", "hi") == [{"role": "user", "content": "hi"}]
    m = thinking_off_messages("no_think_system", "hi")
    assert m[0] == {"role": "system", "content": "/no_think"}
    assert m[1] == {"role": "user", "content": "hi"}


def test_manifest_mechanisms_present():
    man = json.load(open("manifest.json", encoding="utf-8"))
    for k, v in man["models"].items():
        assert v["thinking_off_mechanism"] in ("none", "no_think_system"), k
    assert man["models"]["qwen35_4b"]["thinking_off_mechanism"] == "no_think_system"
    assert man["models"]["llama32_3b"]["thinking_off_mechanism"] == "none"
