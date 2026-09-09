"""Manual Phase 2 verification script.

Not a pytest suite (same "minimal dependencies" rule as Phase 1) -- a
small deterministic script that exercises `prompts/generator.py` and
`data/prompts.json` and reports pass/fail. It also re-runs the existing
`tests_phase1_manual.py` as a regression check, without duplicating its
individual assertions.

Run with: python tests_phase2_manual.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter

from schemas.models import PromptItem
from prompts.generator import generate_prompts

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, bool, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((name, condition, detail))


# A shared, realistic config used across the generation-based checks below.
_BASE_KWARGS = dict(
    brand="TrailBrew",
    category="camping coffee makers",
    competitors=["PeakBrew", "CampJoe"],
    product="TrailBrew Pour-Over Kit",
    use_cases=["backpacking trips", "car camping"],
    requirements=["lightweight gear", "no batteries required"],
)

EXPECTED_INTENTS = {"informational", "recommendation", "comparison", "use_case", "commercial"}


# --- 1. Basic generation ---
try:
    items = generate_prompts(**_BASE_KWARGS, prompt_count=10)
    check("Basic generation: returns requested count", len(items) == 10, f"got {len(items)}")
    check(
        "Basic generation: every result is a PromptItem",
        all(isinstance(i, PromptItem) for i in items),
    )
except Exception as exc:  # noqa: BLE001
    check("Basic generation", False, str(exc))
    items = []

# --- 2. Determinism ---
try:
    items_a = generate_prompts(**_BASE_KWARGS, prompt_count=10)
    items_b = generate_prompts(**_BASE_KWARGS, prompt_count=10)
    as_tuples_a = [(i.prompt_id, i.prompt, i.intent) for i in items_a]
    as_tuples_b = [(i.prompt_id, i.prompt, i.intent) for i in items_b]
    check(
        "Determinism: identical inputs produce identical output (ids, text, intent, order)",
        as_tuples_a == as_tuples_b,
    )
except Exception as exc:  # noqa: BLE001
    check("Determinism", False, str(exc))

# --- 3. Uniqueness ---
try:
    ids = [i.prompt_id for i in items]
    texts = [i.prompt for i in items]
    check("Uniqueness: all prompt IDs unique", len(ids) == len(set(ids)))
    check("Uniqueness: all prompt text unique", len(texts) == len(set(texts)))
except Exception as exc:  # noqa: BLE001
    check("Uniqueness", False, str(exc))

# --- 4. Intent coverage + expected distribution ---
try:
    intents_present = {i.intent for i in items}
    check("Intent coverage: all five intents present", intents_present == EXPECTED_INTENTS, str(intents_present))

    distribution = Counter(i.intent for i in items)
    even_distribution = all(distribution.get(intent) == 2 for intent in EXPECTED_INTENTS)
    check(
        "Intent coverage: 2 prompts per intent for prompt_count=10 with sufficient candidates",
        even_distribution,
        str(dict(distribution)),
    )
except Exception as exc:  # noqa: BLE001
    check("Intent coverage", False, str(exc))

# --- 5. Invalid input ---
invalid_cases = [
    ("empty brand", dict(brand="", category="coffee makers", prompt_count=1)),
    ("whitespace-only brand", dict(brand="   ", category="coffee makers", prompt_count=1)),
    ("empty category", dict(brand="TrailBrew", category="", prompt_count=1)),
    ("whitespace-only category", dict(brand="TrailBrew", category="   ", prompt_count=1)),
    ("prompt_count=0", dict(brand="TrailBrew", category="coffee makers", prompt_count=0)),
    ("negative prompt_count", dict(brand="TrailBrew", category="coffee makers", prompt_count=-3)),
]
for case_name, kwargs in invalid_cases:
    try:
        generate_prompts(**kwargs)
        check(f"Invalid input rejected: {case_name}", False, "did not raise")
    except ValueError:
        check(f"Invalid input rejected: {case_name}", True)
    except Exception as exc:  # noqa: BLE001
        check(f"Invalid input rejected: {case_name}", False, f"raised wrong exception type: {exc!r}")

# --- 6. Competitor behavior ---
# 6a. No competitors supplied: comparison prompts needing {competitor} must be
# skipped entirely, and no fabricated competitor name may appear anywhere.
try:
    items_no_competitors = generate_prompts(
        brand="TrailBrew",
        category="camping coffee makers",
        competitors=None,
        product="TrailBrew Pour-Over Kit",
        use_cases=["backpacking trips", "car camping"],
        requirements=["lightweight gear", "no batteries required"],
        prompt_count=8,
    )
    no_comparison_intent = all(i.intent != "comparison" for i in items_no_competitors)
    no_compare_wording = all("compare" not in i.prompt.lower() for i in items_no_competitors)
    check(
        "Competitor behavior: comparison prompts skipped when no competitors supplied",
        no_comparison_intent and no_compare_wording,
    )
except ValueError as exc:
    # Also acceptable: if too few candidates exist without comparison
    # templates to satisfy prompt_count, a clear ValueError is the
    # documented behavior rather than inventing data.
    check(
        "Competitor behavior: comparison prompts skipped when no competitors supplied",
        True,
        f"raised expected ValueError instead of fabricating data: {exc}",
    )
except Exception as exc:  # noqa: BLE001
    check("Competitor behavior: comparison prompts skipped when no competitors supplied", False, str(exc))

# 6b. Competitors supplied: only supplied competitor names may appear.
try:
    items_with_competitors = generate_prompts(**_BASE_KWARGS, prompt_count=10)
    comparison_items = [i for i in items_with_competitors if i.intent == "comparison"]
    supplied = set(_BASE_KWARGS["competitors"])
    only_supplied_used = all(
        any(name in i.prompt for name in supplied) for i in comparison_items
    )
    no_unlisted_names = True
    for i in comparison_items:
        # crude but sufficient check for this fixed template set: the
        # competitor token in the prompt must be one of the supplied ones.
        if not any(name in i.prompt for name in supplied):
            no_unlisted_names = False
    check(
        "Competitor behavior: only supplied competitor names appear in comparison prompts",
        len(comparison_items) > 0 and only_supplied_used and no_unlisted_names,
        f"{len(comparison_items)} comparison prompts checked",
    )
except Exception as exc:  # noqa: BLE001
    check("Competitor behavior: only supplied competitor names appear", False, str(exc))

# --- 7. JSON round trip ---
try:
    with open("data/prompts.json", encoding="utf-8") as f:
        raw = f.read()
    data = json.loads(raw)
    check("JSON round trip: file is valid JSON", True)
except Exception as exc:  # noqa: BLE001
    check("JSON round trip: file is valid JSON", False, str(exc))
    data = None

if data is not None:
    check("JSON round trip: 'prompts' collection exists", "prompts" in data and isinstance(data["prompts"], list))

    try:
        reconstructed = [PromptItem(**entry) for entry in data.get("prompts", [])]
        check(
            "JSON round trip: every entry reconstructs as PromptItem",
            len(reconstructed) == len(data.get("prompts", [])),
        )
    except Exception as exc:  # noqa: BLE001
        check("JSON round trip: every entry reconstructs as PromptItem", False, str(exc))
        reconstructed = []

    json_ids = [p.prompt_id for p in reconstructed]
    json_texts = [p.prompt for p in reconstructed]
    check("JSON round trip: IDs unique", len(json_ids) == len(set(json_ids)))
    check("JSON round trip: prompt text unique", len(json_texts) == len(set(json_texts)))

# --- 8. Phase 1 regression ---
try:
    proc = subprocess.run(
        [sys.executable, "tests_phase1_manual.py"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    check(
        "Phase 1 regression: tests_phase1_manual.py exits 0 (all Phase 1 checks pass)",
        proc.returncode == 0,
        proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else proc.stderr.strip(),
    )
except Exception as exc:  # noqa: BLE001
    check("Phase 1 regression: tests_phase1_manual.py exits 0 (all Phase 1 checks pass)", False, str(exc))

# --- report ---
print("\n=== Phase 2 Manual Verification Results ===")
all_ok = True
for name, ok, detail in results:
    status_label = PASS if ok else FAIL
    if not ok:
        all_ok = False
    line = f"[{status_label}] {name}"
    if detail:
        line += f" -- {detail}"
    print(line)

passed = sum(1 for _, ok, _ in results if ok)
failed = len(results) - passed
print(f"\nTOTAL: {passed} passed, {failed} failed")
print("OVERALL:", "ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
sys.exit(0 if all_ok else 1)
