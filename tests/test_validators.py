"""Every validator: at least 3 known-good and 3 known-bad outputs."""

import pytest

from tasks.validators import VALIDATORS

# ---------------------------------------------------------------------------
# structured_output
# ---------------------------------------------------------------------------

STRUCT_TASK = {
    "schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive"]},
            "count": {"type": "integer", "minimum": 0, "maximum": 10},
            "due_date": {"type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$"},
        },
        "required": ["status", "count", "due_date"],
        "additionalProperties": False,
    },
    "gold": None,
}

STRUCT_GOOD = [
    '{"status": "active", "count": 3, "due_date": "2026-07-02"}',
    '{"status": "inactive", "count": 0, "due_date": "2025-01-31"}',
    '```json\n{"status": "active", "count": 10, "due_date": "2026-12-25"}\n```',
]

STRUCT_BAD = [
    '{"status": "active", "count": 3',                                # truncated JSON
    '{"status": "paused", "count": 3, "due_date": "2026-07-02"}',     # enum violation
    '{"status": "active", "count": 99, "due_date": "2026-07-02"}',    # range violation
    '{"status": "active", "count": 3, "due_date": "July 2, 2026"}',   # date pattern
    'Sure! Here is the JSON: {"status": "active", "count": 3, "due_date": "2026-07-02"}',
]


@pytest.mark.parametrize("output", STRUCT_GOOD)
def test_structured_good(output):
    r = VALIDATORS["structured_output"].validate(output, STRUCT_TASK)
    assert r == {"json_parses": True, "schema_complies": True}


@pytest.mark.parametrize("output", STRUCT_BAD)
def test_structured_bad(output):
    r = VALIDATORS["structured_output"].validate(output, STRUCT_TASK)
    assert not r["schema_complies"]


def test_structured_parses_but_noncompliant():
    r = VALIDATORS["structured_output"].validate('{"wrong": 1}', STRUCT_TASK)
    assert r == {"json_parses": True, "schema_complies": False}


# ---------------------------------------------------------------------------
# tool_call
# ---------------------------------------------------------------------------

TOOL_TASK = {
    "schema": None,
    "gold": {"tool": "get_weather",
             "arguments": {"city": "Paris", "units": "celsius"}},
}

TOOL_GOOD = [
    '{"tool": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}',
    '{"arguments": {"units": "celsius", "city": "Paris"}, "tool": "get_weather"}',
    '```json\n{"tool": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}\n```',
]

TOOL_BAD = [
    '{"tool": "search_flights", "arguments": {"city": "Paris", "units": "celsius"}}',  # wrong tool
    '{"tool": "get_weather", "arguments": {"city": "paris", "units": "celsius"}}',     # wrong value case
    '{"tool": "get_weather", "arguments": {"city": "Paris"}}',                         # missing arg
    '{"tool": "get_weather"}',                                                         # no arguments
    'I would call get_weather with city Paris.',                                       # not JSON
]


@pytest.mark.parametrize("output", TOOL_GOOD)
def test_tool_call_good(output):
    r = VALIDATORS["tool_call"].validate(output, TOOL_TASK)
    assert r == {"json_parses": True, "tool_selection_correct": True,
                 "arguments_exact_match": True}


@pytest.mark.parametrize("output", TOOL_BAD)
def test_tool_call_bad(output):
    r = VALIDATORS["tool_call"].validate(output, TOOL_TASK)
    assert not r["arguments_exact_match"]


def test_tool_call_selection_without_args_match():
    out = '{"tool": "get_weather", "arguments": {"city": "Lyon", "units": "celsius"}}'
    r = VALIDATORS["tool_call"].validate(out, TOOL_TASK)
    assert r["tool_selection_correct"] and not r["arguments_exact_match"]


# ---------------------------------------------------------------------------
# tool_decline
# ---------------------------------------------------------------------------

DECLINE_TASK = {"schema": None, "gold": {"tool": None}}

DECLINE_GOOD = [
    '{"tool": null}',
    '{"tool": null, "arguments": {}}',
    '```json\n{"tool": null}\n```',
]

DECLINE_BAD = [
    '{"tool": "get_weather", "arguments": {"city": "Paris", "units": "celsius"}}',
    '{"answer": "The capital of Australia is Canberra."}',   # no tool key
    'No tool applies to this request.',                      # not JSON
    '{"tool": "null"}',                                      # string, not null
]


@pytest.mark.parametrize("output", DECLINE_GOOD)
def test_decline_good(output):
    r = VALIDATORS["tool_decline"].validate(output, DECLINE_TASK)
    assert r == {"json_parses": True, "correctly_declined": True}


@pytest.mark.parametrize("output", DECLINE_BAD)
def test_decline_bad(output):
    r = VALIDATORS["tool_decline"].validate(output, DECLINE_TASK)
    assert not r["correctly_declined"]
