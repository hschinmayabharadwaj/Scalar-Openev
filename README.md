---
title: SupportOps OpenEnv
sdk: docker
tags:
  - openenv
  - customer-support
  - evaluation
---

# SupportOps OpenEnv

SupportOps OpenEnv is a real-world environment that simulates customer support ticket triage.
It is designed for training and evaluating agentic systems on practical operations work:

- classify incoming ticket priority,
- route the ticket to the correct queue,
- draft a useful customer reply,
- and resolve with the right internal status code.

The environment follows an OpenEnv-style API with `reset()`, `step()`, and `state()` via HTTP.

## Why this environment is useful

Customer support operations are a common and costly workflow in real businesses. This environment
models realistic constraints and rewards agents for partial progress rather than only terminal success.

## Observation Space

`Observation` fields:

- `task_id`, `difficulty`, `objective`
- `step_count`, `max_steps`
- `ticket`: structured support ticket (customer tier, message, product area)
- `current_priority`, `current_queue`, `reply_draft`
- `last_action`, `action_history`

## Action Space

`Action` model:

- `action_type`: one of
  - `classify_priority`
  - `assign_queue`
  - `draft_reply`
  - `add_internal_note`
  - `resolve_ticket`
  - `noop`
- optional fields by action type:
  - `priority`: `low|medium|high|urgent`
  - `queue`: `billing|technical|account|trust_and_safety|general`
  - `reply_text`, `note`, `resolution_code`

## Reward Design

Each `step()` returns `Reward(score, components, reason)` where `score` is clamped to `[0.0, 1.0]`.

Dense progress signals:

- priority correctness contribution,
- queue correctness contribution,
- reply quality (keyword coverage),
- resolution correctness contribution.

Penalties discourage undesirable behavior:

- repeated identical actions,
- empty/invalid action payloads,
- no-op stalling.

## Tasks and Graders (Easy -> Medium -> Hard)

Deterministic grader score is always in `[0.0, 1.0]`:

`0.25 * priority + 0.25 * queue + 0.30 * reply_quality + 0.20 * resolution`

Included tasks:

1. `task_easy_password` (easy)
2. `task_medium_double_charge` (medium)
3. `task_hard_enterprise_outage` (hard)

Each task has fixed target labels and required reply keywords, producing reproducible outcomes.

## Local Setup

1. Install dependencies:

```bash
pip install -U pip
pip install fastapi httpx openai pydantic python-dotenv uvicorn
```

2. Run environment API:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

3. Check health:

```bash
curl http://localhost:7860/health
```

## Inference Script (Required)

The baseline script is `inference.py` in the project root and uses the OpenAI client.

Set required variables:

- `API_BASE_URL` - LLM API endpoint
- `MODEL_NAME` - model identifier
- `HF_TOKEN` - API key (fallback: `OPENAI_API_KEY`)
- `ENV_BASE_URL` - environment API URL (default `http://localhost:7860`)

Run:

```bash
python inference.py
```

Structured logs are emitted with `[START]`, `[STEP]`, and `[END]` prefixes.

## Baseline Scores

Scores depend on the chosen model and endpoint. Typical deterministic run format:

- task-level scores in `[0.0, 1.0]`
- final average score in `[0.0, 1.0]`

Example output pattern:

- `[START] ...`
- `[STEP] ...`
- `[END] average_score=0.71 task_scores={...}`

## Docker

Build and run:

```bash
docker build -t supportops-openenv .
docker run --rm -p 7860:7860 supportops-openenv
```

## Hugging Face Spaces Deployment

This repository is container-ready for Docker Spaces.

1. Create a new Docker Space.
2. Push this repository.
3. Add runtime secrets/variables:
   - `API_BASE_URL`
   - `MODEL_NAME`
   - `HF_TOKEN`
4. Ensure the Space has the `openenv` tag.

## API Summary

- `GET /health`
- `GET /tasks`
- `POST /reset` with optional `{ "task_id": "..." }`
- `POST /step` with `Action`
- `GET /state`