"""AIVISORA -- Streamlit entrypoint.

Phase 12B: minimal audit-configuration UI wired to the existing Phase 11
LangGraph pipeline (``agents.graph.run_pipeline``). The UI collects inputs,
runs the pipeline once, shows the real execution stages derived from the
returned state, and retains the completed result in session state. It does
NOT render the results dashboard (metrics / gaps / actions) -- that is a
later phase.
"""

from __future__ import annotations

import streamlit as st

from agents.graph import run_pipeline
from config import settings

# User-facing stage label -> the state key that proves the stage actually ran.
_STAGES = [
    ("Generate Prompts", "prompts"),
    ("Collect AI Responses", "ai_responses"),
    ("Analyze Responses", "analyzed_responses"),
    ("Measure Visibility", "visibility_results"),
    ("Analyze Website", "website_data"),
    ("Diagnose Gaps", "gap_results"),
    ("Plan Actions", "action_plan"),
]


# --- pure, importable helpers -----------------------------------------------
def parse_competitors(raw: str) -> list[str]:
    """Comma-separated brand names -> clean list, empties dropped, duplicates
    removed (case-insensitively) while keeping first-seen spelling.
    """
    out: list[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        name = part.strip()
        key = name.casefold()
        if name and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def parse_competitor_sites(raw: str) -> tuple[dict[str, str], list[str]]:
    """One ``brand=url`` per line -> ({brand: url}, [ignored malformed lines]).
    A line is valid only with a name and an http(s) URL. Never raises.
    """
    mapping: dict[str, str] = {}
    invalid: list[str] = []
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line:
            continue
        name, sep, url = line.partition("=")
        name, url = name.strip(), url.strip()
        if sep and name and url.startswith(("http://", "https://")):
            mapping[name] = url
        else:
            invalid.append(line)
    return mapping, invalid


def validate_inputs(target_brand: str, category: str, prompt_count, provider_selected: bool) -> list[str]:
    errors: list[str] = []
    if not (target_brand or "").strip():
        errors.append("Target Brand is required.")
    if not (category or "").strip():
        errors.append("Category is required.")
    try:
        count = int(prompt_count)
    except (TypeError, ValueError):
        count = 0
    if not 1 <= count <= 20:
        errors.append("Prompt Count must be between 1 and 20.")
    if not provider_selected:
        errors.append("Select at least one AI environment.")
    return errors


def build_config(target_brand, category, target_website, competitors_raw, competitor_sites_raw, prompt_count):
    """Turn the raw UI fields into the pipeline's input state dict (parsing
    happens here). Returns (config, [ignored competitor-website lines]).
    """
    sites, invalid = parse_competitor_sites(competitor_sites_raw)
    config = {
        "target_brand": (target_brand or "").strip(),
        "category": (category or "").strip(),
        "website": ((target_website or "").strip() or None),
        "competitors": parse_competitors(competitors_raw),
        "competitor_sites": sites,
        "prompt_count": int(prompt_count),
    }
    return config, invalid


def _providers(use_gemini: bool, use_openrouter: bool) -> list:
    """The selected real provider collectors (constructed lazily, only when
    chosen). Google AI Overview contributes manual observations, not a
    collector, so it is not built here.
    """
    selected: list = []
    if use_gemini:
        from collectors.gemini import GeminiCollector

        selected.append(GeminiCollector())
    if use_openrouter:
        from collectors.openrouter import OpenRouterCollector

        selected.append(OpenRouterCollector())
    return selected


def run_audit(config: dict, *, use_gemini: bool, use_openrouter: bool, _run_pipeline=run_pipeline) -> dict:
    """Execute the existing Phase 11 pipeline for ``config`` with the selected
    providers. ``_run_pipeline`` is injectable for deterministic tests.
    """
    return _run_pipeline(config, providers=_providers(use_gemini, use_openrouter))


def _stage_summary(key: str, result: dict) -> str:
    """A short, truthful line derived only from the returned pipeline state."""
    if key == "prompts":
        return f"generated {len(result.get('prompts') or [])} prompts"
    if key == "ai_responses":
        environments = {getattr(r, "source_type", None) for r in (result.get("ai_responses") or [])}
        return f"collected responses from {len(environments)} AI environment(s)"
    if key == "analyzed_responses":
        return f"analyzed {len(result.get('analyzed_responses') or [])} responses"
    if key == "visibility_results":
        return "measured AI search visibility" if result.get("visibility_results") is not None else "not enough data to measure"
    if key == "website_data":
        return "analyzed website evidence" if result.get("website_data") is not None else "website evidence not available"
    if key == "gap_results":
        report = result.get("gap_results")
        return f"identified {len(report.findings)} gap finding(s)" if report is not None else "no gap findings"
    if key == "action_plan":
        plan = result.get("action_plan")
        return f"prepared {len(plan.actions)} optimization action(s)" if plan is not None else "no actions"
    return "done"


# --- results presentation (reads pipeline state only; computes nothing) ----
def _source_label(source_type) -> str:
    value = getattr(source_type, "value", str(source_type or "")).lower()
    if "gemini" in value:
        return "Gemini"
    if "openrouter" in value:
        return "OpenRouter"
    if "aio" in value or "overview" in value:
        return "Google AI Overview — Observed Data"
    if "mock" in value:
        return "Mock"
    return value.replace("_", " ").title() or "Unknown"


def _sources_line(result: dict) -> str:
    labels: list[str] = []
    for response in result.get("ai_responses") or []:
        label = _source_label(getattr(response, "source_type", None))
        if label not in labels:
            labels.append(label)
    return "Sources: " + (" · ".join(labels) if labels else "none")


def _counts_line(result: dict) -> str:
    parts: list[str] = []
    if result.get("prompts"):
        parts.append(f"{len(result['prompts'])} prompts")
    if result.get("ai_responses"):
        parts.append(f"{len(result['ai_responses'])} AI responses")
    if result.get("competitors"):
        parts.append(f"{len(result['competitors'])} competitors")
    return " · ".join(parts)


def _pct(ratio) -> str:
    """A visibility Ratio (or None) -> 'NN%'. A None rate means the metric
    was not measurable -> 'N/A' (never silently 0%).
    """
    rate = getattr(ratio, "rate", None)
    return "N/A" if rate is None else f"{round(rate * 100)}%"


def _num(value) -> str:
    return "N/A" if value is None else f"{value:.1f}"


def _kv(rows) -> None:
    """(label, value) pairs as aligned two-column text -- no metric cards."""
    for label, value in rows:
        left, right = st.columns([2, 1])
        left.write(label)
        right.write(value)


def _render_track(metrics) -> None:
    if metrics is None:
        st.write("Visibility could not be measured from the collected responses.")
        return
    _kv([
        ("Mention Rate", _pct(metrics.mention_rate)),
        ("Recommendation Rate", _pct(metrics.recommendation_rate)),
        ("Average Mention Position", _num(getattr(metrics.average_mention_position, "value", None))),
        ("Citation Rate", _pct(metrics.citation_rate)),
        ("Share of Voice", _pct(metrics.share_of_voice.get(metrics.target_brand))),
    ])
    if metrics.by_source:
        st.caption("Cross-model visibility (mention rate)")
        _kv([(_source_label(source), _pct(sv.mention_rate)) for source, sv in metrics.by_source.items()])
    if metrics.competitors:
        st.caption("Competitor share of voice")
        _kv([(name, _pct(metrics.share_of_voice.get(name))) for name in metrics.competitors])


def _render_diagnose(report) -> None:
    if report is None or not report.findings:
        st.write("No gap findings are available for this audit.")
        return
    for finding in report.findings:
        left, right = st.columns([2, 1])
        left.write(finding.category.value.replace("_", " ").title())
        right.write(finding.status.value)
        detail = finding.summary or ""
        if finding.competitor:
            detail = f"{detail} (competitor: {finding.competitor})".strip()
        if finding.evidence:
            detail = f"{detail}  Evidence: {', '.join(finding.evidence)}".strip()
        if detail:
            left.caption(detail)  # constrain to the left column so it never runs under the status label


def _render_optimize(plan) -> None:
    if plan is None or not plan.actions:
        st.write("No optimization actions are available for this audit.")
        return
    current_priority = None
    for position, action in enumerate(plan.actions):  # existing HIGH -> MEDIUM -> LOW ordering
        if position:
            st.divider()  # separate each recommendation; none before the first / after the last
        if action.priority.value != current_priority:
            current_priority = action.priority.value
            st.markdown(f"**{current_priority}**")
        st.markdown(f"**{action.title}**")
        # Split the continuous "WHAT: ... WHERE: ... HOW: ..." description into
        # one line per label ("**WHAT:** text"); WHY is the finding-derived reason.
        desc = action.description or ""
        marks = [(m, desc.find(m)) for m in ("WHAT:", "WHERE:", "HOW:") if desc.find(m) != -1]
        marks.sort(key=lambda mp: mp[1])
        if marks:
            for i, (marker, start) in enumerate(marks):
                end = marks[i + 1][1] if i + 1 < len(marks) else len(desc)
                st.markdown(f"**{marker}** {desc[start + len(marker):end].strip()}")
        elif desc:
            st.markdown(desc)
        st.markdown(f"**WHY:** {action.reason}")
        extra: list[str] = []
        if getattr(action.expected_impact, "value", "UNKNOWN") != "UNKNOWN":
            extra.append(f"Expected impact: {action.expected_impact.value}")
        if action.competitor:
            extra.append(f"Competitor: {action.competitor}")
        if extra:
            st.caption(" · ".join(extra))


def render_results(result: dict) -> None:
    """Presentation only -- shows the pipeline's own visibility_results,
    gap_results and action_plan. No metric or ranking is computed here.
    """
    st.caption(_sources_line(result))
    counts = _counts_line(result)
    if counts:
        st.caption(counts)

    with st.container(border=True):
        st.subheader("TRACK")
        _render_track(result.get("visibility_results"))

    with st.container(border=True):
        st.subheader("DIAGNOSE")
        _render_diagnose(result.get("gap_results"))

    with st.container(border=True):
        st.subheader("OPTIMIZE")
        _render_optimize(result.get("action_plan"))


def _new_audit() -> None:
    """Sidebar '+ New Audit': clear the current-audit inputs and result."""
    for key in ("brand", "category", "website", "competitors", "competitor_sites", "prompt_count"):
        st.session_state.pop(key, None)
    st.session_state.pop("audit_config", None)
    st.session_state.pop("audit_result", None)


# --- UI -------------------------------------------------------------------
st.set_page_config(page_title="AIVISORA", layout="centered")

# Font, plus: hide Streamlit's Deploy button, and give bordered result
# sections a subtle rounded maroon boundary. No other theme changes.
st.markdown(
    "<style>@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');"
    "html, body, .stApp, [class*=\"css\"] { font-family: 'Inter', 'Segoe UI', sans-serif; }"
    "[data-testid=\"stAppDeployButton\"], [data-testid=\"stDeployButton\"], .stDeployButton { display: none !important; }"
    "[data-testid=\"stVerticalBlockBorderWrapper\"] { border-color: rgba(128,0,32,.35); border-radius: 12px; }</style>",
    unsafe_allow_html=True,
)

st.markdown(
    "# <span style='color:#800020'>AIVISORA</span>: AI Search Visibility Intelligence & Optimization System",
    unsafe_allow_html=True,
)
st.caption("*Mapping How Brands Are Seen, Cited, and Recommended Across AI Search*")

st.divider()
st.subheader("AUDIT CONFIGURATION")

target_brand = st.text_input("Target Brand", key="brand", placeholder="Acme")
category = st.text_input("Category", key="category", placeholder="Project management software")
target_website = st.text_input("Target Website (optional)", key="website", placeholder="https://example.com")
competitors_raw = st.text_area("Competitors (comma-separated)", key="competitors", placeholder="Asana, Monday.com, ClickUp")
competitor_sites_raw = st.text_area(
    "Competitor Websites (one per line, brand=url)",
    key="competitor_sites",
    placeholder="Asana=https://asana.com\nMonday.com=https://monday.com",
)
prompt_count = st.number_input("Prompt Count", min_value=1, max_value=20, value=8, step=1, key="prompt_count")

# Sidebar: current-audit context. Written after the inputs so it reflects
# their live values; it complements the main page, never duplicates it.
with st.sidebar:
    st.markdown("### <span style='color:#800020'>AIVISORA</span>", unsafe_allow_html=True)
    st.caption("AI Search Visibility Auditor")

    st.divider()
    st.markdown("**AUDIT**")
    st.button("+ New Audit", on_click=_new_audit)
    st.caption("Current Audit")

    st.divider()
    st.markdown("**MONITORING SCOPE**")
    for _label, _value in (
        ("Target Brand", (target_brand or "").strip() or "Not configured"),
        ("Category", (category or "").strip() or "Not configured"),
        ("Website", (target_website or "").strip() or "Not provided"),
        ("Competitors", f"{len(parse_competitors(competitors_raw))} competitors"),
        ("Prompts", f"{int(prompt_count)} prompts"),
    ):
        st.markdown(f"**{_label}**  \n{_value}")

    st.divider()
    st.markdown("**AI ENVIRONMENTS**")
    for _name, _detail in (
        ("Gemini", settings.gemini_model),
        ("OpenRouter", settings.openrouter_model),
        ("Google AI Overview", "Observed Data"),
    ):
        st.markdown(f"**{_name}**  \n{_detail}")

    st.divider()
    st.markdown("**AUDIT PIPELINE**")
    st.markdown("TRACK  \n↓  \nDIAGNOSE  \n↓  \nOPTIMIZE")

st.divider()
st.subheader("AI MODELS USED")

# Every audit conceptually covers all three environments; models are read
# from the existing config -- no provider selection, no new config here.
for _col, (_name, _detail) in zip(
    st.columns(3),
    (("Gemini", settings.gemini_model),
     ("OpenRouter", settings.openrouter_model),
     ("Google AI Overview", "Observed Data")),
):
    with _col.container(border=True):
        st.markdown(f"**{_name}**")
        st.caption(_detail)

st.divider()

if st.button("Run Audit"):
    errors = validate_inputs(target_brand, category, prompt_count, True)
    if errors:
        for message in errors:
            st.error(message)
    else:
        config, invalid_site_lines = build_config(
            target_brand, category, target_website, competitors_raw, competitor_sites_raw, prompt_count
        )
        if invalid_site_lines:
            st.warning("Ignored malformed competitor website line(s): " + "; ".join(invalid_site_lines))
        st.info("Google AI Overview — Observed Data: no observations were provided, so this source is not included in the results.")

        st.divider()
        st.subheader("WORKING")
        with st.status("Running audit", expanded=True) as status:
            try:
                result = run_audit(config, use_gemini=True, use_openrouter=True)
                st.session_state["audit_config"] = config
                st.session_state["audit_result"] = result
                for label, key in _STAGES:
                    st.write(f"{label} — {_stage_summary(key, result)}")
                    if key == "prompts" and result.get("prompts"):
                        with st.expander(f"View {len(result['prompts'])} generated prompts"):
                            st.table([
                                {"#": i, "Prompt": getattr(p, "prompt", "")}
                                for i, p in enumerate(result["prompts"], 1)
                            ])
                _responses = result.get("ai_responses") or []
                st.markdown("**Collected AI responses**")
                if _responses:
                    _pidx = {getattr(p, "prompt_id", None): i for i, p in enumerate(result.get("prompts") or [], 1)}
                    st.table([
                        {
                            "Environment": _source_label(getattr(r, "source_type", None)),
                            "Prompt": _pidx.get(getattr(r, "prompt_id", None), getattr(r, "prompt_id", "") or ""),
                            "Response": ((getattr(r, "answer", None) or getattr(r, "error_message", None) or "").strip().replace("\n", " ")[:120]) or "—",
                        }
                        for r in _responses
                    ])
                else:
                    st.write("No AI responses were collected for this audit.")
                status.update(label="Audit complete", state="complete")
            except Exception:  # noqa: BLE001 - surface a concise message, never a stack trace
                st.session_state.pop("audit_result", None)
                status.update(label="Audit failed", state="error")
                st.error("The audit could not be completed. Please review your configuration and try again.")

try:
    _stored_result = st.session_state.get("audit_result")
except Exception:  # noqa: BLE001 - a context-less read must never crash the module
    _stored_result = None
if _stored_result:
    st.divider()
    st.subheader("RESULTS")
    render_results(_stored_result)
