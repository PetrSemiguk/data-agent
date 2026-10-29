"""Prompt and tool of the judge model that reads an agent's answer for grade.py.

The judge never sees the expected values: it only extracts what the answer
states, and grade.py compares that with the reference in code.
"""

JUDGE_PROMPT = """You grade answers of a data analysis assistant. You get the user's \
question, the assistant's answer and a numbered list of checks. You do not know the \
correct values. Do not compute, guess or judge whether numbers are right: your only job \
is to read the answer carefully.

Report every check by calling submit_grades once, with one entry per check.

number checks: extract the value the answer states for exactly what the description names.
- Give it in the unit of the description: "in percent" means 74.2, not 0.742. Write \
amounts as plain numbers: $836K -> 836000, 8.98 million -> 8980000, 34.3K -> 34300.
- Keep the sign: a decline of 1.1% is -1.1 when the description asks for growth.
- precision is the rounding step of the number as written, in the description's unit: \
74.2% -> 0.1, 74% -> 1, $836K -> 1000, 8.98 million -> 10000, 9,079,584 -> 1, \
"about 3%" -> 1.
- If the answer gives a range for it, put the ends in low and high and leave value null.
- The value must refer to exactly what the description names: the same group, period \
and definition. A number about another year, group or measure does not count; if the \
answer does not state the named value, value is null. When the answer gives several \
values for it, take the one the answer presents as its main result.

ranking checks: list the items in the order the answer ranks them for what the \
description names, highest first, with their full names as written. If the answer \
implies an order through values or words ("ahead of", "third place"), use that order. \
If the answer gives no such ranking, items is null.

mention checks: verdict "yes" if the answer clearly says what the description requires, \
in any wording; otherwise "no".

not_claim checks: the description says what the answer must NOT do. Verdict "yes" if the \
answer DOES it, in any wording, also indirectly (for example through an order or a \
conclusion that implies it); otherwise "no".

For every check give a short verbatim quote from the answer that supports your reading \
(null if there is none) and a one-sentence explanation."""

SUBMIT_GRADES_TOOL = {
    "name": "submit_grades",
    "description": "Submit what the answer states for every check. Call it exactly once.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grades": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Number of the check."},
                        "value": {"type": ["number", "null"], "description": "number checks: the stated value."},
                        "low": {"type": ["number", "null"], "description": "number checks: lower end of a stated range."},
                        "high": {"type": ["number", "null"], "description": "number checks: upper end of a stated range."},
                        "precision": {"type": ["number", "null"], "description": "number checks: rounding step as written."},
                        "items": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "description": "ranking checks: items in the answer's order, highest first.",
                        },
                        "verdict": {"type": ["string", "null"], "enum": ["yes", "no", None],
                                    "description": "mention and not_claim checks."},
                        "quote": {"type": ["string", "null"], "description": "Verbatim quote from the answer."},
                        "explanation": {"type": "string"},
                    },
                    "required": ["index", "explanation"],
                },
            },
        },
        "required": ["grades"],
    },
}


def judge_request(question: str, answer: str, checks: list[dict]) -> str:
    """The user message for the judge: question, answer and check descriptions only."""
    listed = "\n".join(f"{i}. [{check['type']}] {check['description']}" for i, check in enumerate(checks, start=1))
    return (
        f"Question:\n{question}\n\n"
        f"Answer:\n<answer>\n{answer}\n</answer>\n\n"
        f"Checks:\n{listed}"
    )
