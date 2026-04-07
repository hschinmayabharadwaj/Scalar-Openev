from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

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


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Emit [END] line per hackathon spec."""
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} rewards={rewards_str}", flush=True)


def call_llm_action(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
    """Call the LLM to decide the next action."""
    prompt = (
        "You are a customer support triage agent. Return ONLY valid JSON with keys matching this schema: "
        "{action_type, priority?, queue?, reply_text?, note?, resolution_code?}. "
        "Choose one best next action for progress.\n"
        f"Observation:\n{json.dumps(observation, indent=2)}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "Respond with JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "action_type": "draft_reply",
            "reply_text": (
                "We opened an incident and will share the next update in 15 minutes. "
                "Please confirm if this helps."
            ),
        }

    if "action_type" not in parsed:
        parsed = {"action_type": "noop"}
    return parsed


def run_task(
    http_client: httpx.Client, llm_client: OpenAI, task_id: str, max_steps: int = 8
) -> tuple[bool, int, List[float]]:
    """
    Run a single task episode.
    Returns (success, step_count, rewards_list).
    """
    log_start(task_id, MODEL_NAME)

    rewards: List[float] = []
    step_idx = 0
    success = False
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
                # Check if task was successfully resolved
                task_score = float(payload.get("info", {}).get("task_score", 0.0))
                success = task_score >= 0.5  # Consider success if score >= 0.5
                break

    except Exception as exc:
        error = str(exc)
        if step_idx == 0:
            step_idx = 1
        log_step(step_idx, {"action_type": "error"}, 0.0, True, error)
        rewards.append(0.0)

    log_end(success, step_idx, rewards)
    return success, step_idx, rewards


def main() -> None:
    if HF_TOKEN is None:
        raise ValueError("HF_TOKEN environment variable is required")

    llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
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
