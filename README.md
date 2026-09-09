# AEO Radar — AI Search Visibility Auditor

A final-year project that evaluates how consistently a brand appears in
AI-generated answers, calculates AI-search visibility metrics, analyzes a
brand's website and competitor evidence to identify probable content/evidence
gaps, and generates prioritized technical/content optimization actions.

## Concept: TRACK → DIAGNOSE → OPTIMIZE

- **TRACK** — measure AI-search visibility: brand mention rate, recommendation
  rate, share of voice, average mention position, citation rate (where
  available), competitor comparison, and cross-model visibility.
- **DIAGNOSE** — compare AI responses against the target website and
  competitor websites to surface *probable* content/evidence gaps (missing
  topics, evidence, product information, FAQs, structured data). The system
  never claims to know an AI model's private reasoning — findings are framed
  as "probable gap", "observed pattern", or "evidence suggests".
- **OPTIMIZE** — convert detected gaps into a prioritized, implementation-oriented
  action plan. The system never guarantees that an action will cause an AI
  system to rank or recommend a brand.

## Current status: Phase 1 (Foundation & Scaffold)

This repository currently contains **only** the foundational scaffold:
project structure, configuration handling, core Pydantic schemas, a
provider-agnostic collector interface, and a minimal Streamlit health-check
page.

**Live AI collectors (Gemini, OpenRouter), the website crawler, visibility
analysis, gap/attribution detection, the LangGraph workflow, and the action
optimizer are NOT implemented yet.** These arrive in later phases.

## Zero-budget constraint

This project is built to run without any paid API usage:

- Gemini API — free tier only.
- OpenRouter — free models only.
- Local Python processing wherever possible.
- Google AI Overview data — used only as explicitly observed/sample data,
  never via a live/paid API, and never fabricated.

Every collected AI response carries a `source_type` field
(`live_gemini`, `openrouter_free`, `google_aio_observed`, or `mock`) so that
live data, open/free-model data, and observed/sample data are never mixed
silently.

## High-level planned architecture

```
Streamlit UI
  -> Prompt Generator
  -> LangGraph workflow
  -> AI Collectors (Gemini / OpenRouter / Google AIO observed)
  -> Response Normalization
  -> Visibility Analysis
  -> Website Crawler
  -> Gap / Attribution Analysis
  -> Action Optimizer
  -> Pydantic Validation
  -> Dashboard
```

Only the schemas and interfaces needed to support this later are in place
right now; the pipeline itself is not wired up yet.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Environment variables

See `.env.example`. All are optional in Phase 1 — the application must not
crash if any of them are missing.

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini free-tier API key (used starting a later phase) |
| `OPENROUTER_API_KEY` | OpenRouter free-model API key (used starting a later phase) |
| `GEMINI_MODEL` | Gemini model identifier |
| `OPENROUTER_MODEL` | OpenRouter free model identifier |
| `REQUEST_TIMEOUT` | Timeout in seconds for outbound network requests |
| `MOCK_MODE` | When true, the app must not attempt live API calls |

## Running the app

```bash
streamlit run app.py
```

This currently shows only the project title and a configuration status
panel (which providers are configured, and whether mock mode is active).
No API keys are ever displayed.
