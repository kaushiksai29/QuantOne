"""Deterministic validators, one class per task family (CLAUDE.md rule 1).

Each validator takes (raw_output: str, task: dict) and returns a dict of
metric-name -> bool. No LLM judges, no heuristics beyond a fixed
markdown-fence strip. The scorer looks validators up in VALIDATORS by the
task's validator_id.
"""

import json
import re

import jsonschema

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)


def strip_code_fence(text):
    """Deterministically unwrap a single ```...``` fence, if the entire
    output is one fenced block. Anything else is returned unchanged."""
    m = _FENCE_RE.match(text)
    return m.group(1) if m else text


def try_parse_json(raw_output):
    """Returns (parsed_ok, value). parsed_ok is True iff the (fence-stripped)
    output is a single valid JSON document."""
    try:
        return True, json.loads(strip_code_fence(raw_output))
    except (json.JSONDecodeError, TypeError):
        return False, None


class StructuredOutputValidator:
    validator_id = "structured_output"
    metrics = ("json_parses", "schema_complies")

    def validate(self, raw_output, task):
        ok, value = try_parse_json(raw_output)
        complies = False
        if ok:
            try:
                jsonschema.Draft202012Validator(task["schema"]).validate(value)
                complies = True
            except jsonschema.ValidationError:
                complies = False
        return {"json_parses": ok, "schema_complies": complies}


class ToolCallValidator:
    validator_id = "tool_call"
    metrics = ("json_parses", "tool_selection_correct", "arguments_exact_match")

    def validate(self, raw_output, task):
        ok, value = try_parse_json(raw_output)
        gold = task["gold"]
        selected = ok and isinstance(value, dict) and value.get("tool") == gold["tool"]
        args_match = selected and value.get("arguments") == gold["arguments"]
        return {
            "json_parses": ok,
            "tool_selection_correct": bool(selected),
            "arguments_exact_match": bool(args_match),
        }


class ToolDeclineValidator:
    validator_id = "tool_decline"
    metrics = ("json_parses", "correctly_declined")

    def validate(self, raw_output, task):
        ok, value = try_parse_json(raw_output)
        declined = (ok and isinstance(value, dict)
                    and "tool" in value and value["tool"] is None)
        return {"json_parses": ok, "correctly_declined": bool(declined)}


VALIDATORS = {v.validator_id: v() for v in
              (StructuredOutputValidator, ToolCallValidator, ToolDeclineValidator)}
