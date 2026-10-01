# Weather Advisory Bot

A chat bot that answers questions about outdoor activity safety using **live weather data** and **written safety policies (SOPs)** — never free-floating advice from the model's own judgment.

Built with **LangGraph**, **Google Gemini**, **FastAPI**, and the **Open-Meteo** weather API.

---

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A [Google Gemini API key](https://aistudio.google.com/apikey)

### 2. Clone and Install

```bash
git clone https://github.com/tanyaaa0070/Weather-Advisory-Bot.git
cd Weather-Advisory-Bot
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env and add your Gemini API key:
#   GOOGLE_API_KEY=your-key-here
```

### 4. Run

```bash
python -m app.server
```

Open **http://localhost:8000** in your browser and start chatting.

### 5. Run Eval Suite

```bash
python -m eval.eval_suite
```

---

## Architecture

### LangGraph Pipeline

The bot uses a real graph with **9 nodes**, **4 conditional routing functions**, and **6 distinct terminal paths**:

```
  START
    │
    ▼
  [parse_intent]  ──── LLM extracts location, activity, time
    │
    ├── (not weather query)    ──►  [handle_non_weather]  ──►  END
    ├── (no location found)    ──►  [handle_no_location]  ──►  END
    │
    ▼
  [geocode_location]  ──── Open-Meteo geocoding API
    │
    ├── (geocoding failed)     ──►  [handle_api_failure]  ──►  END
    │
    ▼
  [fetch_weather]  ──── Open-Meteo forecast API
    │
    ├── (API failed)           ──►  [handle_api_failure]  ──►  END
    │
    ▼
  [match_sops]  ──── Deterministic policy matching (NO LLM)
    │
    ├── (no SOPs matched)      ──►  [handle_no_guidance]  ──►  END
    │
    ▼
  [compose_response]  ──── LLM composes response citing SOPs
    │
    ▼
  END
```

### Where the LLM Is Used (and Where It Isn't)

The LLM is used in **exactly two places**:

| Node | LLM Role | What It Cannot Do |
|------|----------|-------------------|
| `parse_intent` | Extracts location, activity, time from natural language | Cannot decide safety thresholds |
| `compose_response` | Phrases the response naturally, citing SOPs | Cannot add advice beyond SOPs, cannot invent numbers |

**Everything else is deterministic code:**
- Geocoding → HTTP call to Open-Meteo
- Weather fetch → HTTP call to Open-Meteo
- SOP matching → Python threshold evaluation
- Failure handling → Template responses
- Routing → Conditional edge functions

### SOPs (Standard Operating Procedures)

**File:** `sops.yaml`

**Format:** YAML — chosen because:
1. Human-readable and editable by non-developers (policy team).
2. Supports inline comments documenting the reasoning behind each rule.
3. Easy to diff in version control when policies change.
4. Parsed at runtime — no code changes needed to add, modify, or remove SOPs.

**To add an 11th SOP:** Edit `sops.yaml`, add a new entry with a unique ID, restart the server. No code changes.

**Current SOPs (13 total across 4 categories):**

| ID | Name | Category | Severity |
|----|------|----------|----------|
| SOP-EX-001 | Extreme UV Outdoor Exercise Warning | outdoor_exercise | high |
| SOP-EX-002 | Extreme Heat Exercise Halt | outdoor_exercise | critical |
| SOP-EX-003 | High Wind Cycling/Two-Wheeler Safety | outdoor_exercise | high |
| SOP-EX-004 | Moderate Heat Exercise Caution | outdoor_exercise | moderate |
| SOP-TR-001 | Heavy Precipitation Travel Warning | travel | high |
| SOP-TR-002 | Moderate Rain Travel Advisory | travel | low |
| SOP-TR-003 | Strong Wind Travel Caution | travel | moderate |
| SOP-VG-001 | Heat Advisory for Children/Elderly | vulnerable_groups | high |
| SOP-VG-002 | Cold Weather Pet Safety Advisory | vulnerable_groups | moderate |
| SOP-VG-003 | Oppressive Humidity Warning | vulnerable_groups | moderate |
| SOP-GN-001 | Favorable Outdoor Leisure Conditions | general_outdoor | info |
| SOP-GN-002 | Severe Weather System Override | severe_weather | critical |
| SOP-GN-003 | Marginal Outdoor Leisure Conditions | general_outdoor | low |

### Condition Evaluation

Each SOP defines conditions as structured checks:

```yaml
conditions:
  logic: "all"  # or "any"
  checks:
    - parameter: "temperature_2m"
      operator: ">="
      value: 42
```

The evaluator supports operators: `>`, `>=`, `<`, `<=`, `==`, `between`.  
Daily parameters use dot notation: `daily.precipitation_sum`.

**The evaluator is 100% deterministic Python** — no LLM involvement in deciding whether a threshold is met.

### Multiple SOP Resolution

When multiple SOPs match the same query (e.g., high UV AND strong wind on a cycling question), **all matching SOPs are surfaced**, sorted by severity (critical first). The highest-severity SOP leads the response; lower-severity ones appear as additional considerations.

**Rationale:** For a safety-critical application, hiding a relevant warning because another warning ranks higher creates liability. The user should see the full picture.

### The Fuzzy SOP (SOP-GN-001)

For "is today good for a picnic?" — there's no single number that makes a day good for a picnic. SOP-GN-001 checks that **all** comfort factors are simultaneously in pleasant ranges:
- Temperature: 15-33°C
- Precipitation probability: <25%
- Wind: <25 km/h
- UV: <7

This is a multi-factor comfort assessment, not a single threshold.

### Session Memory

Within a single chat session, the bot remembers earlier turns. If a user asks about cycling in Mumbai and follows up with "what about walking instead?", the bot reuses the Mumbai context. Memory resets between sessions (on server restart or clearing the session).

**Implementation:** Full message history is stored in-memory per session and passed to the LangGraph pipeline on each invocation. The `parse_intent` node receives conversation history so it can resolve follow-up references.

---

## Eval Suite

Run with: `python -m eval.eval_suite`

**11 test cases covering:**

| # | Test | Category |
|---|------|----------|
| 1 | High wind + cycling → SOP-EX-003 | SOP clearly applies |
| 2 | Extreme heat → SOP-EX-002 critical | SOP clearly applies |
| 3 | "Pedalling to office" (paraphrased cycling) | Paraphrased intent |
| 4 | "Take my toddler out to play" (paraphrased children) | Paraphrased intent |
| 5 | Real API call to Chennai (live weather) | Severe live conditions |
| 6 | "Fly my drone" (no SOP covers this) | No SOP match |
| 7 | Mocked API timeout | API failure |
| 8 | Prompt injection attempt | Adversarial |
| 9 | UV + Wind simultaneous match | Multi-SOP resolution |
| 10 | Perfect picnic weather → SOP-GN-001 | Fuzzy comfort SOP |
| 11 | Follow-up question reusing location | Session memory |

### On Live Weather Tests

Test 5 uses real API data. If weather is calm when running, the test still verifies that real numbers appear in the response. The test will naturally show different results depending on when it's run — that's by design.

**For a suite that needs to keep working after a weather event passes:** Record the API response during the event and use it as a mock fixture, tagged with the capture date. The live test stays as a canary; the recorded fixture provides deterministic replay.

---

## Project Structure

```
weather-advisory-bot/
├── .env.example          # Environment config template
├── .gitignore            # Keeps API keys out of git
├── requirements.txt      # Python dependencies
├── sops.yaml             # Safety policies (edit this, not code)
├── README.md
├── app/
│   ├── __init__.py
│   ├── state.py          # LangGraph state schema
│   ├── sop_loader.py     # SOP loading + deterministic evaluation
│   ├── weather.py        # Open-Meteo API client
│   ├── graph.py          # LangGraph pipeline (9 nodes, 4 routers)
│   └── server.py         # FastAPI server + session management
├── frontend/
│   └── index.html        # Chat frontend (single file)
└── eval/
    └── eval_suite.py     # 11-case evaluation suite
```

---

## Design Decisions

### Why YAML for SOPs?
JSON is too rigid for humans to maintain (no comments, trailing comma issues). A database is overkill — we have ~15 rules, not 15,000. YAML strikes the right balance: structured enough for code to parse, readable enough for a policy team to maintain.

### Why the LLM only composes, never decides?
The LLM's role is constrained to two tasks: understanding what the user is asking (intent parsing) and phrasing the answer naturally (response composition). It never evaluates whether conditions are dangerous — that's deterministic code comparing API numbers against SOP thresholds. This means:
- The bot cannot "hallucinate" a weather value.
- The bot cannot decide a condition is safe when it isn't.
- The SOP evaluation logic is testable without an LLM.

### Why surface all matching SOPs?
We considered: pick the highest severity and answer with only that. But for a safety application, if high UV AND high wind both apply to a cycling question, hiding the wind warning because UV ranked first would be irresponsible. Surfacing all gives the user the full picture.

### Why separate failure nodes?
Each failure mode produces a meaningfully different response. A "location not found" message is different from "weather API is down" is different from "we don't have policy for that." Routing to separate nodes makes each path testable and the responses precise.
