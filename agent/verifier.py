"""A second model run that fact-checks an agent's final answer against the data."""

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import anthropic

from agent.config import VERIFY_MAX_STEPS
from agent.events import EventHandler
from agent.loop import continue_run, new_state, safe_emitter, write_log
from agent.prompts import VERIFIER_PROMPT
from agent.sandbox import Sandbox
from agent.tools import CLAIM_STATUSES, GET_SCHEMA_TOOL, REPORT_VERIFICATION_TOOL, RUN_CODE_TOOL

VERIFIER_TOOLS = [GET_SCHEMA_TOOL, RUN_CODE_TOOL, REPORT_VERIFICATION_TOOL]
VERIFIER_NAMESPACE = "verifier"

VerificationStatus = Literal["ok", "issues_found", "incomplete"]


@dataclass
class ClaimCheck:
    """One claim of the answer and what the verifier's code showed about it."""

    claim: str
    status: str
    evidence: str
    correction: str | None = None


@dataclass
class VerificationReport:
    """Outcome of one verification run.

    "incomplete" means the verifier did not submit a report (step limit, API
    error, or it answered with text instead); `claims` is then empty.
    """

    status: VerificationStatus
    claims: list[ClaimCheck]
    counts: dict[str, int]
    steps: int
    input_tokens: int
    output_tokens: int
    duration_s: float
    error_message: str | None = None
    log_path: str | None = None
    # Checks of the memory notes sent along with the answer, one per note, in
    # the same order. They do not count in `status` and `counts` above.
    note_checks: list[ClaimCheck] = field(default_factory=list)
    # input_tokens is the uncached part; the whole prompt also includes these.
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    # None when config.py has no prices for the model.
    cost_usd: float | None = 0.0


def verification_request(question: str, answer: str, notes: list[dict[str, Any]] | None = None) -> str:
    """The only message the verifier gets: the question, the answer and the
    facts the analyst wants to remember; nothing else."""
    request = (
        f"The user's question:\n{question}\n\n"
        f"The answer to verify:\n<answer>\n{answer}\n</answer>"
    )
    if notes:
        listed = "\n".join(
            f"<note>{note['text']}</note> (analyst's evidence: {note.get('evidence') or 'none'})" for note in notes
        )
        request += (
            "\n\nThe analyst also wants to save these facts about the data to memory. Check each "
            "one as a separate claim, with the note text copied exactly as the claim:\n" + listed
        )
    return request


def normalize(text: str) -> str:
    """For matching a reported claim to a note: case, spacing and a final period ignored."""
    return " ".join(text.lower().split()).rstrip(".")


def split_note_checks(claims: list[ClaimCheck], notes: list[dict[str, Any]]) -> tuple[list[ClaimCheck], list[ClaimCheck]]:
    """Separate the checks of memory notes from the checks of the answer.

    A note the verifier did not report on gets an "unverifiable" check.
    """
    answer_claims = list(claims)
    note_checks: list[ClaimCheck] = []
    for note in notes:
        wanted = normalize(note["text"])
        match = next(
            (c for c in answer_claims if normalize(c.claim) == wanted or (normalize(c.claim) and normalize(c.claim) in wanted)),
            None,
        )
        if match is None:
            note_checks.append(ClaimCheck(note["text"], "unverifiable", "The verifier did not report on this note."))
        else:
            answer_claims.remove(match)
            note_checks.append(ClaimCheck(note["text"], match.status, match.evidence, match.correction))
    return answer_claims, note_checks


def verify_answer(
    question: str,
    answer: str,
    sandbox: Sandbox,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-5-5",
    max_steps: int = VERIFY_MAX_STEPS,
    on_event: EventHandler | None = None,
    notes: list[dict[str, Any]] | None = None,
    cache: bool = True,
) -> VerificationReport:
    """Recompute the claims of `answer` with code in a fresh model context.

    Runs in the agent's kernel but in a separate namespace with its own copy of
    `df`, so it cannot see or change the agent's variables. The namespace is
    deleted afterwards. Only a log is written, no output folder.
    `notes` are facts the agent wants to remember ({"text", "evidence"}); each
    is checked as its own claim and reported in `note_checks`.
    """
    notes = notes or []
    state = new_state(
        [{"role": "user", "content": verification_request(question, answer, notes)}],
        model,
        max_steps,
        {"role": "verifier", "question": question, "answer_checked": answer},
        log_suffix="_verify",
        system=VERIFIER_PROMPT,
        tools=VERIFIER_TOOLS,
        finish_tool=REPORT_VERIFICATION_TOOL["name"],
        namespace=VERIFIER_NAMESPACE,
        cache=cache,
    )
    try:
        result = continue_run(state, sandbox, client, safe_emitter(on_event), None, pause_on_ask=False)
    finally:
        sandbox.drop_namespace(VERIFIER_NAMESPACE)

    claims: list[ClaimCheck] = []
    note_checks: list[ClaimCheck] = []
    error_message = result.error_message
    if state.finish_input is None:
        status: VerificationStatus = "incomplete"
        error_message = error_message or "The verifier finished without calling report_verification."
    else:
        claims = [
            ClaimCheck(
                claim=item["claim"].strip(),
                status=item["status"],
                evidence=item["evidence"].strip(),
                correction=(item.get("correction") or "").strip() or None,
            )
            for item in state.finish_input["claims"]
        ]
        claims, note_checks = split_note_checks(claims, notes)
        status = "issues_found" if any(c.status == "contradicted" for c in claims) else "ok"

    tally = Counter(claim.status for claim in claims)
    report = VerificationReport(
        status=status,
        claims=claims,
        counts={name: tally[name] for name in CLAIM_STATUSES},
        steps=result.steps,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        duration_s=result.duration_s,
        error_message=error_message,
        log_path=result.log_path,
        note_checks=note_checks,
        cache_write_tokens=result.cache_write_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cost_usd=result.cost_usd,
    )
    state.log["verification"] = asdict(report)
    write_log(state.log_path, state.log)
    return report
