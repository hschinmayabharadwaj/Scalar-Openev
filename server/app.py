from __future__ import annotations

import uvicorn
from fastapi import FastAPI, HTTPException

from models import Action, ResetRequest, StepResponse
from server.environment import SupportTriageEnvironment

app = FastAPI(title="SupportOps OpenEnv", version="1.0.0")
env = SupportTriageEnvironment()


@app.get("/health")
def health() -> dict:
	return {"status": "ok"}


@app.get("/tasks")
def tasks() -> dict:
	return {"tasks": [task.model_dump() for task in env.tasks()]}


@app.post("/reset")
def reset(request: ResetRequest | None = None) -> dict:
	requested_task = request.task_id if request is not None else None
	try:
		observation = env.reset(task_id=requested_task)
	except ValueError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc

	return {"observation": observation.model_dump()}


@app.post("/step", response_model=StepResponse)
def step(action: Action) -> StepResponse:
	try:
		return env.step(action)
	except RuntimeError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state() -> dict:
	try:
		return {"state": env.state().model_dump()}
	except RuntimeError as exc:
		raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
	"""Entry point for the server."""
	uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
