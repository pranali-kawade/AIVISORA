"""Focused prompt-generation diversity checks. Deterministic, offline: no
LLM, no API, no network, no randomness.

Run: python3 tests_prompt_diversity_manual.py
"""

from __future__ import annotations

import sys

from prompts.generator import generate_prompts
from prompts.templates import INTENTS, TEMPLATES, get_templates
from schemas.models import PromptItem

PASS, FAIL = "PASS", "FAIL"
res: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    res.append((name, PASS if cond else FAIL, detail))


FIVE = ["informational", "recommendation", "comparison", "use_case", "commercial"]

# Realistic *pipeline* inputs: only brand + category + competitors, exactly
# what agents/nodes.py passes (no use_cases / requirements / product).
PIPE = dict(brand="SkillReach", category="rural skill development NGO",
            competitors=["GramShiksha", "BharatSkills"])
_DOMAIN_WORDS = ("SkillReach", "rural skill development NGO", "GramShiksha", "BharatSkills")


def shape(text: str) -> str:
    """Strip the substituted values so only the question structure remains."""
    for w in _DOMAIN_WORDS:
        text = text.replace(w, "")
    return " ".join(text.split())


items = generate_prompts(**PIPE, prompt_count=10)

# 1. exact requested count
check("1. exact requested count", len(items) == 10, f"got {len(items)}")

# 7. PromptItem output remains valid
check("7. every item is a valid PromptItem (id + text + intent)",
      all(isinstance(i, PromptItem) and i.prompt_id and i.prompt and i.intent for i in items))

# 2. deterministic output
again = generate_prompts(**PIPE, prompt_count=10)
check("2. deterministic (identical ids / text / intent / order)",
      [(i.prompt_id, i.prompt, i.intent) for i in items]
      == [(i.prompt_id, i.prompt, i.intent) for i in again])

# 3. all five intents remain supported
check("3. the five existing intents are unchanged and each has templates",
      INTENTS == FIVE and all(get_templates(x) for x in FIVE))
check("3b. a typical run covers every intent",
      {i.intent for i in items} == set(FIVE), str({i.intent for i in items}))

# 4. multiple prompts within an intent are genuinely different structures
for intent in ("informational", "recommendation", "use_case", "commercial"):
    got = [i.prompt for i in generate_prompts(**PIPE, prompt_count=15) if i.intent == intent]
    check(f"4. {intent}: >= 2 prompts with distinct question structures",
          len(got) >= 2 and len({shape(g) for g in got}) == len(got), str(got))
check("4b. every intent defines several distinct templates",
      all(len(v) >= 3 and len(set(v)) == len(v) for v in TEMPLATES.values()),
      str({k: len(v) for k, v in TEMPLATES.items()}))

# 5. prompts across intents are meaningfully different (no shared structure)
shapes = [shape(i.prompt) for i in items]
check("5. cross-intent prompts are meaningfully different", len(set(shapes)) == len(shapes))

# 6. brand / category / competitor substitution still works, and some
#    prompts are category-level (no brand named)
joined = " ".join(i.prompt for i in items)
comp = [i.prompt for i in items if i.intent == "comparison"]
check("6. category substituted; competitors only in comparison; category-level prompts exist",
      "rural skill development NGO" in joined
      and comp and all(("GramShiksha" in p or "BharatSkills" in p) and "SkillReach" in p for p in comp)
      and any("SkillReach" not in i.prompt for i in items)
      and "{" not in joined and "}" not in joined)

# 6b. the same templates adapt to an unrelated domain
health = generate_prompts(brand="NovaClinic", category="telehealth platforms",
                          competitors=["MediNow"], prompt_count=8)
hjoined = " ".join(i.prompt for i in health)
check("6b. same templates produce a healthcare-appropriate, filled set",
      len(health) == 8 and "telehealth platforms" in hjoined
      and "rural" not in hjoined.lower() and "{" not in hjoined)

# 8. no example brand / domain is hard-coded into the prompt layer
src = (open("prompts/templates.py", encoding="utf-8").read()
       + open("prompts/generator.py", encoding="utf-8").read()).lower()
check("8. SUREPROED / example content is NOT hard-coded",
      not any(w in src for w in ("sureproed", "skillreach", "novaclinic", "trailbrew",
                                 "rural skill", "vocational", "beneficiar")))

# extra: the real pipeline input shape no longer under-produces / errors
spread = generate_prompts(**PIPE, prompt_count=8)
check("real pipeline inputs (brand + category + competitors) yield a full, spread set",
      len(spread) == 8 and len({i.intent for i in spread}) >= 4)

print("\n=== Prompt-generation diversity ===")
p = f = 0
for n, s, d in res:
    p += s == PASS
    f += s == FAIL
    print(f"[{s}] {n}" + (f" -- {d}" if d and s == FAIL else ""))
print(f"\nTOTAL: {p} passed, {f} failed")
print("LLM / API / NETWORK / RANDOMNESS = 0")
sys.exit(0 if f == 0 else 1)
