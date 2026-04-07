from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")

ENV_NAME = "supportops-openenv"


def log_start(task_name: str, model: str) -> None:
    """Emit [START] line per hackathon spec."""
    print(f"[START] task={task_name} env={ENV_NAME} model={model}", flush=True)


def log_step(
    step_idx: int,
    action: Dict[str, Any],
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    """Emit [STEP] line per hackathon spec."""
    action_str = json.dumps(action, separators=(",", ":"), sort_keys=True)
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step_idx} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line per hackathon spec."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def call_llm_action(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to decide the next action."""
    system_prompt = """You are an expert customer support triage agent. Your job is to:
1. Classify ticket priority (low/medium/high/urgent)
2. Assign to correct queue (billing/technical/account/trust_and_safety/general)
3. Draft a helpful customer reply with key information
4. Resolve with appropriate resolution code

IMPORTANT GUIDELINES:
- For enterprise customers with outages: priority=urgent, queue=technical
- For billing issues: queue=billing
- For security/breach reports: priority=urgent, queue=trust_and_safety
- For password/login issues: queue=account
- Include specific keywords in replies: incident details, timeframes, next steps
- Always resolve with a resolution_code that matches the situation

Return ONLY valid JSON. One action per response."""

    action_schema = """Available actions:
- {"action_type": "classify_priority", "priority": "low|medium|high|urgent"}
- {"action_type": "assign_queue", "queue": "billing|technical|account|trust_and_safety|general"}
- {"action_type": "draft_reply", "reply_text": "Your detailed response to customer"}
- {"action_type": "add_internal_note", "note": "Internal notes for team"}
- {"action_type": "resolve_ticket", "resolution_code": "code_here"}

Common resolution codes: awaiting_customer_confirmation, resolved_with_documentation, 
billing_investigation_opened, engineering_investigation, incident_escalated, security_incident_opened"""

    # Determine what actions have been taken
    history = observation.get("action_history", [])
    current_priority = observation.get("current_priority")
    current_queue = observation.get("current_queue")
    reply_draft = observation.get("reply_draft")
    
    status = []
    if current_priority:
        status.append(f"Priority set: {current_priority}")
    if current_queue:
        status.append(f"Queue assigned: {current_queue}")
    if reply_draft:
        status.append("Reply drafted")
    
    status_str = ", ".join(status) if status else "No actions taken yet"
    
    prompt = f"""{action_schema}

Current ticket:
- ID: {observation['ticket']['ticket_id']}
- Customer: {observation['ticket']['customer_name']} ({observation['ticket']['customer_tier']} tier)
- Subject: {observation['ticket']['subject']}
- Message: {observation['ticket']['message']}
- Product Area: {observation['ticket']['product_area']}

Objective: {observation['objective']}
Steps taken: {observation['step_count']}/{observation['max_steps']}
Current status: {status_str}

Decide the single best next action to maximize progress. If priority, queue, and reply are set, resolve the ticket."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"

    # Clean up response - extract JSON if wrapped in markdown
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        # Fallback action based on what's missing
        if not current_priority:
            parsed = {"action_type": "classify_priority", "priority": "medium"}
        elif not current_queue:
            parsed = {"action_type": "assign_queue", "queue": "general"}
        elif not reply_draft:
            parsed = {
                "action_type": "draft_reply",
                "reply_text": (
                    "Thank you for contacting support. We're looking into your issue "
                    "and will provide an update shortly. Please confirm if this helps."
                ),
            }
        else:
            parsed = {"action_type": "resolve_ticket", "resolution_code": "awaiting_customer_confirmation"}

    if "action_type" not in parsed:
        parsed = {"action_type": "noop"}
    return parsed


def run_task(
    http_client: httpx.Client, llm_client: OpenAI, task_id: str, max_steps: int = 8
) -> tuple[bool, int, float, List[float]]:
    """
    Run a single task episode.
    Returns (success, step_count, final_score, rewards_list).
    """
    log_start(task_id, MODEL_NAME)

    rewards: List[float] = []
    step_idx = 0
    success = False
    final_score = 0.0
    error: Optional[str] = None

    try:
        reset_resp = http_client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
        reset_resp.raise_for_status()
        observation = reset_resp.json()["observation"]

        for idx in range(1, max_steps + 1):
            step_idx = idx
            error = None

            action = call_llm_action(llm_client, observation)
            step_resp = http_client.post(f"{ENV_BASE_URL}/step", json=action)
            step_resp.raise_for_status()
            payload = step_resp.json()

            observation = payload["observation"]
            reward = float(payload["reward"]["score"])
            done = bool(payload["done"])

            # Check for any action error from the reward reason
            reason = payload.get("reward", {}).get("reason", "")
            if "penalty" in reason.lower() or "missing" in reason.lower() or "empty" in reason.lower():
                error = reason

            rewards.append(reward)
            log_step(idx, action, reward, done, error)

            if done:
                # Get the final task score from grader
                final_score = float(payload.get("info", {}).get("task_score", 0.0))
                success = final_score >= 0.5  # Consider success if score >= 0.5
                break

    except Exception as exc:
        error = str(exc)
        if step_idx == 0:
            step_idx = 1
        log_step(step_idx, {"action_type": "error"}, 0.0, True, error)
        rewards.append(0.0)

    log_end(success, step_idx, final_score, rewards)
    return success, step_idx, final_score, rewards


def main() -> None:
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY (or HF_TOKEN) before running inference.")

    llm_client = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    http_client = httpx.Client(timeout=60.0)

    try:
        tasks_resp = http_client.get(f"{ENV_BASE_URL}/tasks")
        tasks_resp.raise_for_status()
        tasks: List[Dict[str, Any]] = tasks_resp.json()["tasks"]

        for task in tasks:
            task_id = task["task_id"]
            run_task(http_client, llm_client, task_id)

    finally:
        http_client.close()


if __name__ == "__main__":
    main()
