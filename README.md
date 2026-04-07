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

Included tasks (6 total):

| Task ID | Difficulty | Description |
|---------|------------|-------------|
| `task_easy_password` | Easy | Password reset request routing |
| `task_easy_feature_question` | Easy | Product feature inquiry |
| `task_medium_double_charge` | Medium | Billing dispute handling |
| `task_medium_api_rate_limit` | Medium | Technical API issue diagnosis |
| `task_hard_enterprise_outage` | Hard | Enterprise production outage escalation |
| `task_hard_data_breach_report` | Hard | Security incident response |

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

Set required environment variables:

- `API_BASE_URL` - LLM API endpoint
- `MODEL_NAME` - model identifier
- `HF_TOKEN` - API key (fallback: `GEMINI_API_KEY`)
- `ENV_BASE_URL` - environment API URL (default `http://localhost:7860`)

Run:

```bash
python inference.py
```

Structured logs are emitted with `[START]`, `[STEP]`, and `[END]` prefixes.

## Baseline Scores

Baseline performance using `gpt-4.1-mini` model:

| Task ID | Difficulty | Expected Score |
|---------|------------|----------------|
| `task_easy_password` | Easy | 0.75 - 0.85 |
| `task_easy_feature_question` | Easy | 0.70 - 0.85 |
| `task_medium_double_charge` | Medium | 0.60 - 0.75 |
| `task_medium_api_rate_limit` | Medium | 0.55 - 0.70 |
| `task_hard_enterprise_outage` | Hard | 0.45 - 0.65 |
| `task_hard_data_breach_report` | Hard | 0.40 - 0.60 |
| **Average** | - | **0.55 - 0.70** |

Example output:

```
[START] task=task_easy_password env=supportops-openenv model=gpt-4.1-mini
[STEP] step=1 action={"action_type":"classify_priority","priority":"medium"} reward=0.25 done=false error=null
[STEP] step=2 action={"action_type":"assign_queue","queue":"account"} reward=0.25 done=false error=null
[STEP] step=3 action={"action_type":"draft_reply","reply_text":"..."} reward=0.20 done=false error=null
[STEP] step=4 action={"action_type":"resolve_ticket","resolution_code":"awaiting_customer_confirmation"} reward=0.25 done=true error=null
[END] success=true steps=4 score=0.95 rewards=0.25,0.25,0.20,0.25
```

## Docker

Build and run locally:

```bash
docker build -t openev:update .
docker run --rm -p 7860:7860 openev:update
```

Or pull the pre-built image from Docker Hub with the `update` tag:

```bash
docker pull hschinmaybharadwaj05/openev:update
docker run --rm -p 7860:7860 hschinmaybharadwaj05/openev:update
```

**Current tag: `update`** — The image is tagged as `hschinmaybharadwaj05/openev:update` on Docker Hub.

## Hugging Face Spaces Deployment

This environment is deployed as a **containerized Docker Space** on Hugging Face.

**Live API URL:** `https://m134pra-supportops-openenv.hf.space`

### What the HF Space Does

The Hugging Face Space hosts the **environment API server** - it does NOT provide a UI. The hackathon evaluator calls the REST endpoints to test AI agents against the tasks.

### Testing the Live API

```bash
# Health check
curl https://m134pra-supportops-openenv.hf.space/health

# List all tasks
curl https://m134pra-supportops-openenv.hf.space/tasks

# Reset to a specific task
curl -X POST https://m134pra-supportops-openenv.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_easy_password"}'

# Get current state
curl https://m134pra-supportops-openenv.hf.space/state

# Take an action
curl -X POST https://m134pra-supportops-openenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "classify_priority", "priority": "medium"}'
```

### Deployment Steps

1. Create a new Docker Space on Hugging Face.
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