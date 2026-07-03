"""Schema-driven task generation -> tasks/tasks.jsonl.

Generates exactly 500 tasks from a fixed RNG seed, byte-for-byte reproducible:
  - 350 structured-output tasks (family "structured_output"): produce a JSON
    object complying with a generated JSON Schema, nesting depth 1-4, covering
    enums, ISO dates, ranged numbers, arrays, and nested objects.
  - 120 tool-call tasks (family "tool_call"): pick the correct tool among
    2-5 distractors and produce exact arguments.
  - 30 should-not-call tasks (family "tool_decline"): no shown tool applies.

Task record: {prompt_id, family, prompt, schema, gold, validator_id}
"""

import json
import random
from pathlib import Path

GENERATION_SEED = 20260702
N_STRUCTURED = 350
N_TOOL_CALL = 120
N_DECLINE = 30

# ---------------------------------------------------------------------------
# Structured-output schema generation
# ---------------------------------------------------------------------------

FIELD_NAMES = [
    "id", "name", "status", "category", "priority", "created_at", "due_date",
    "start_date", "end_date", "quantity", "price", "rating", "score", "count",
    "weight", "height", "tags", "labels", "items", "notes", "owner", "region",
    "currency", "language", "severity", "phase", "channel", "tier", "unit",
    "code", "batch", "version", "level", "duration", "capacity", "threshold",
    "discount", "margin", "index", "rank",
]

ENUM_WORDS = [
    "active", "inactive", "pending", "archived", "draft", "approved",
    "rejected", "low", "medium", "high", "critical", "north", "south",
    "east", "west", "gold", "silver", "bronze", "daily", "weekly", "monthly",
]

DATE_SCHEMA = {
    "type": "string",
    "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
    "description": "ISO 8601 date (YYYY-MM-DD)",
}


def _leaf_schema(rng, kind):
    if kind == "enum":
        values = rng.sample(ENUM_WORDS, rng.randint(3, 5))
        return {"type": "string", "enum": values}
    if kind == "date":
        return dict(DATE_SCHEMA)
    if kind == "integer":
        lo = rng.randint(0, 50)
        return {"type": "integer", "minimum": lo, "maximum": lo + rng.randint(10, 200)}
    if kind == "number":
        lo = rng.randint(0, 20)
        return {"type": "number", "minimum": lo, "maximum": lo + rng.randint(5, 100)}
    if kind == "string":
        return {"type": "string", "minLength": 1}
    raise ValueError(kind)


def _array_schema(rng):
    item_kind = rng.choice(["enum", "integer", "string"])
    return {
        "type": "array",
        "items": _leaf_schema(rng, item_kind),
        "minItems": 1,
        "maxItems": rng.randint(3, 6),
    }


def _object_schema(rng, depth):
    """Object schema whose deepest nesting chain has exactly `depth` levels."""
    n_props = rng.randint(2, 4)
    names = rng.sample(FIELD_NAMES, n_props)
    props = {}
    for i, name in enumerate(names):
        if depth > 1 and i == 0:
            # Force one property to carry the remaining nesting depth.
            props[name] = _object_schema(rng, depth - 1)
        else:
            kind = rng.choice(["enum", "date", "integer", "number", "string", "array"])
            props[name] = _array_schema(rng) if kind == "array" else _leaf_schema(rng, kind)
    return {
        "type": "object",
        "properties": props,
        "required": sorted(props),
        "additionalProperties": False,
    }


STRUCTURED_PROMPT = (
    "Produce one example JSON object that validates against the JSON Schema "
    "below. Invent plausible values. Output only the JSON object, with no "
    "markdown fences and no other text.\n\nJSON Schema:\n{schema}"
)


def make_structured_tasks(rng):
    tasks = []
    for i in range(N_STRUCTURED):
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


# ---------------------------------------------------------------------------
# Tool-call task generation
# ---------------------------------------------------------------------------

CITIES = ["Paris", "Tokyo", "Nairobi", "Toronto", "Sydney", "Lima", "Oslo", "Seoul"]
NAMES = ["Alice Chen", "Ravi Patel", "Maria Lopez", "Tom Novak", "Aisha Bello"]
EMAILS = ["alice@example.com", "ravi@example.com", "maria@example.com", "tom@example.com"]
TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
LANGUAGES = ["French", "Japanese", "Spanish", "German", "Swahili"]
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "INR"]
RESTAURANTS = ["Blue Fig", "Casa Verde", "Umi Sushi", "The Copper Pot"]
PHRASES = ["good morning", "where is the station", "thank you very much", "see you tomorrow"]
SUBJECTS = ["Quarterly report", "Meeting follow-up", "Invoice attached", "Schedule change"]
TASK_TEXTS = ["Renew passport", "Submit expense report", "Back up the database", "Order office chairs"]


def _date(rng):
    return f"2026-{rng.randint(7, 12):02d}-{rng.randint(1, 28):02d}"


def _time(rng):
    return f"{rng.randint(6, 21):02d}:{rng.choice([0, 15, 30, 45]):02d}"


TOOLS = {
    "get_weather": {
        "description": "Get the current weather for a city.",
        "parameters": {"city": {"type": "string"},
                       "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "make": lambda rng: {"city": rng.choice(CITIES),
                             "units": rng.choice(["celsius", "fahrenheit"])},
        "template": "Get the current weather in {city}, in {units} units.",
    },
    "send_email": {
        "description": "Send an email to a recipient.",
        "parameters": {"to": {"type": "string"}, "subject": {"type": "string"}},
        "make": lambda rng: {"to": rng.choice(EMAILS), "subject": rng.choice(SUBJECTS)},
        "template": "Send an email to {to} with the subject \"{subject}\".",
    },
    "create_calendar_event": {
        "description": "Create a calendar event on a given date.",
        "parameters": {"title": {"type": "string"}, "date": {"type": "string"},
                       "duration_minutes": {"type": "integer"}},
        "make": lambda rng: {"title": rng.choice(SUBJECTS), "date": _date(rng),
                             "duration_minutes": rng.choice([15, 30, 45, 60, 90])},
        "template": "Create a calendar event titled \"{title}\" on {date} lasting {duration_minutes} minutes.",
    },
    "convert_currency": {
        "description": "Convert an amount between two currencies.",
        "parameters": {"amount": {"type": "number"},
                       "from_currency": {"type": "string"}, "to_currency": {"type": "string"}},
        "make": lambda rng: {"amount": rng.randint(10, 900),
                             "from_currency": rng.choice(CURRENCIES),
                             "to_currency": rng.choice(CURRENCIES)},
        "template": "Convert {amount} {from_currency} to {to_currency}.",
    },
    "search_flights": {
        "description": "Search for flights between two cities on a date.",
        "parameters": {"origin": {"type": "string"}, "destination": {"type": "string"},
                       "date": {"type": "string"}},
        "make": lambda rng: {"origin": rng.choice(CITIES), "destination": rng.choice(CITIES),
                             "date": _date(rng)},
        "template": "Search for flights from {origin} to {destination} on {date}.",
    },
    "set_reminder": {
        "description": "Set a reminder with a message at a given time.",
        "parameters": {"message": {"type": "string"}, "time": {"type": "string"}},
        "make": lambda rng: {"message": rng.choice(TASK_TEXTS), "time": _time(rng)},
        "template": "Set a reminder to \"{message}\" at {time}.",
    },
    "translate_text": {
        "description": "Translate text into a target language.",
        "parameters": {"text": {"type": "string"}, "target_language": {"type": "string"}},
        "make": lambda rng: {"text": rng.choice(PHRASES),
                             "target_language": rng.choice(LANGUAGES)},
        "template": "Translate \"{text}\" into {target_language}.",
    },
    "get_stock_price": {
        "description": "Get the latest stock price for a ticker symbol.",
        "parameters": {"ticker": {"type": "string"}},
        "make": lambda rng: {"ticker": rng.choice(TICKERS)},
        "template": "What is the latest stock price for {ticker}?",
    },
    "book_restaurant": {
        "description": "Book a restaurant table for a party on a date.",
        "parameters": {"name": {"type": "string"}, "date": {"type": "string"},
                       "party_size": {"type": "integer"}},
        "make": lambda rng: {"name": rng.choice(RESTAURANTS), "date": _date(rng),
                             "party_size": rng.randint(2, 8)},
        "template": "Book a table at {name} on {date} for {party_size} people.",
    },
    "create_todo": {
        "description": "Add a task to the to-do list with a priority.",
        "parameters": {"task": {"type": "string"},
                       "priority": {"type": "string", "enum": ["low", "medium", "high"]}},
        "make": lambda rng: {"task": rng.choice(TASK_TEXTS),
                             "priority": rng.choice(["low", "medium", "high"])},
        "template": "Add \"{task}\" to my to-do list with {priority} priority.",
    },
    "get_directions": {
        "description": "Get directions between two places by a travel mode.",
        "parameters": {"origin": {"type": "string"}, "destination": {"type": "string"},
                       "mode": {"type": "string", "enum": ["driving", "walking", "transit"]}},
        "make": lambda rng: {"origin": rng.choice(CITIES), "destination": rng.choice(CITIES),
                             "mode": rng.choice(["driving", "walking", "transit"])},
        "template": "Get {mode} directions from {origin} to {destination}.",
    },
    "calculate_tip": {
        "description": "Calculate the tip for a bill amount at a percentage.",
        "parameters": {"bill_amount": {"type": "number"}, "percent": {"type": "integer"}},
        "make": lambda rng: {"bill_amount": rng.randint(20, 300),
                             "percent": rng.choice([10, 15, 18, 20, 25])},
        "template": "Calculate a {percent} percent tip on a bill of {bill_amount}.",
    },
}

# Requests that none of the tools above can fulfill.
DECLINE_REQUESTS = [
    "Write a haiku about autumn leaves.",
    "Summarize the plot of Moby-Dick in two sentences.",
    "What year did the Berlin Wall fall?",
    "Explain how photosynthesis works.",
    "Recommend three science fiction novels.",
    "Tell me a joke about programmers.",
    "What is the capital of Australia?",
    "Draft a short poem about the ocean.",
    "How do I tie a bowline knot?",
    "List the planets in order from the sun.",
    "Explain the difference between TCP and UDP.",
    "What is the boiling point of water at sea level?",
    "Give me a recipe for banana bread.",
    "Who painted the Mona Lisa?",
    "Describe the rules of chess in brief.",
    "What does the acronym NASA stand for?",
    "Suggest a name for a black kitten.",
    "How many bones are in the human hand?",
    "Explain what a haiku is.",
    "What is the tallest mountain in Africa?",
    "Write a limerick about coffee.",
    "How does a refrigerator keep food cold?",
    "Name three famous Impressionist painters.",
    "What is the chemical symbol for gold?",
    "Explain the offside rule in soccer.",
    "How far is the Moon from Earth on average?",
    "Give me a fun fact about octopuses.",
    "What language has the most native speakers?",
    "Describe how a rainbow forms.",
    "Who wrote Pride and Prejudice?",
]

TOOL_PROMPT = (
    "You have access to these tools:\n\n{tools}\n\nUser request: {request}\n\n"
    "Respond with only a single JSON object and no other text. If exactly one "
    "tool can fulfill the request, respond with "
    "{{\"tool\": \"<tool_name>\", \"arguments\": {{...}}}} using argument "
    "values taken verbatim from the request. If none of the tools can fulfill "
    "the request, respond with {{\"tool\": null}}."
)


def _tool_defs_json(tool_names):
    defs = [{"name": n, "description": TOOLS[n]["description"],
             "parameters": TOOLS[n]["parameters"]} for n in tool_names]
    return json.dumps(defs, sort_keys=True, indent=2)


def make_tool_tasks(rng):
    tasks = []
    tool_names = sorted(TOOLS)
    for i in range(N_TOOL_CALL):
        correct = tool_names[i % len(tool_names)]
        n_distractors = rng.randint(2, 5)
        distractors = rng.sample([n for n in tool_names if n != correct], n_distractors)
        shown = [correct] + distractors
        rng.shuffle(shown)
        args = TOOLS[correct]["make"](rng)
        request = TOOLS[correct]["template"].format(**args)
        tasks.append({
            "prompt_id": f"tool-{i:04d}",
            "family": "tool_call",
            "prompt": TOOL_PROMPT.format(tools=_tool_defs_json(shown), request=request),
            "schema": None,
            "gold": {"tool": correct, "arguments": args},
            "validator_id": "tool_call",
        })
    return tasks


def make_decline_tasks(rng):
    tasks = []
    tool_names = sorted(TOOLS)
    for i in range(N_DECLINE):
        shown = rng.sample(tool_names, rng.randint(2, 5))
        request = DECLINE_REQUESTS[i]
        tasks.append({
            "prompt_id": f"decline-{i:04d}",
            "family": "tool_decline",
            "prompt": TOOL_PROMPT.format(tools=_tool_defs_json(shown), request=request),
            "schema": None,
            "gold": {"tool": None},
            "validator_id": "tool_decline",
        })
    return tasks


# ---------------------------------------------------------------------------


def generate(out_path):
    rng = random.Random(GENERATION_SEED)
    tasks = make_structured_tasks(rng) + make_tool_tasks(rng) + make_decline_tasks(rng)
    assert len(tasks) == 500
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for t in tasks:
            f.write(json.dumps(t, sort_keys=True, ensure_ascii=True,
                               separators=(",", ":")) + "\n")
    return tasks


if __name__ == "__main__":
    tasks = generate(Path(__file__).parent / "tasks.jsonl")
    by_family = {}
    for t in tasks:
        by_family[t["family"]] = by_family.get(t["family"], 0) + 1
    print(f"wrote {len(tasks)} tasks: {by_family}")
