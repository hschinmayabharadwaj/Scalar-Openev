from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class OpenEnvClient:
	def __init__(self, base_url: str = "http://localhost:7860") -> None:
		self.base_url = base_url.rstrip("/")
		self._client = httpx.Client(timeout=30.0)

	def close(self) -> None:
		self._client.close()

	def tasks(self) -> Dict[str, Any]:
		response = self._client.get(f"{self.base_url}/tasks")
		response.raise_for_status()
		return response.json()

	def reset(self, task_id: Optional[str] = None) -> Dict[str, Any]:
		payload = {"task_id": task_id} if task_id else {}
		response = self._client.post(f"{self.base_url}/reset", json=payload)
		response.raise_for_status()
		return response.json()

	def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
		response = self._client.post(f"{self.base_url}/step", json=action)
		response.raise_for_status()
		return response.json()

	def state(self) -> Dict[str, Any]:
		response = self._client.get(f"{self.base_url}/state")
		response.raise_for_status()
		return response.json()
