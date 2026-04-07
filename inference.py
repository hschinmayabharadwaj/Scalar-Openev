from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import httpx
from openai import OpenAI


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
API_BASE_URL = os.getenv("API_BASE_URL", "https://gemini.googleapis.com/v1/gemini")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GEMINI_API_KEY")


def log_start(model: str, env_base: str, tasks: int) -> None:
    print(
        f"[START] model={model} api_base_url={API_BASE_URL} env_base_url={env_base} task_count={tasks}",
        flush=True,
    )


def log_step(task_id: str, step_idx: int, reward: float, done: bool, task_score: float, action: Dict[str, Any]) -> None:
    action_json = json.dumps(action, separators=(",", ":"), sort_keys=True)
    print(
        f"[STEP] task_id={task_id} step={step_idx} reward={reward:.4f} done={str(done).lower()} "
        f"task_score={task_score:.4f} action={action_json}",
        flush=True,
    )


def log_end(task_scores: Dict[str, float], average: float) -> None:
    ordered = json.dumps(task_scores, sort_keys=True)
    print(f"[END] average_score={average:.4f} task_scores={ordered}", flush=True)


def call_llm_action(client: OpenAI, observation: Dict[str, Any]) -> Dict[str, Any]:
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
) -> float:
    reset_resp = http_client.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    observation = reset_resp.json()["observation"]

    score = 0.0
    for idx in range(1, max_steps + 1):
        action = call_llm_action(llm_client, observation)
        step_resp = http_client.post(f"{ENV_BASE_URL}/step", json=action)
        step_resp.raise_for_status()
        payload = step_resp.json()
        observation = payload["observation"]
        reward = float(payload["reward"]["score"])
        done = bool(payload["done"])
        score = float(payload.get("info", {}).get("task_score", score))
        log_step(task_id, idx, reward, done, score, action)
        if done:
            break
    return score


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Set HF_TOKEN (or GEMINI_API_KEY) before running inference.")

    llm_client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    http_client = httpx.Client(timeout=60.0)

    try:
        tasks_resp = http_client.get(f"{ENV_BASE_URL}/tasks")
        tasks_resp.raise_for_status()
        tasks: List[Dict[str, Any]] = tasks_resp.json()["tasks"]

        log_start(MODEL_NAME, ENV_BASE_URL, len(tasks))

        scores: Dict[str, float] = {}
        for task in tasks:
            task_id = task["task_id"]
            scores[task_id] = run_task(http_client, llm_client, task_id)

        average = sum(scores.values()) / max(1, len(scores))
        log_end(scores, average)
    finally:
        http_client.close()


if __name__ == "__main__":
    main()