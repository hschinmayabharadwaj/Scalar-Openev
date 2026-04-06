from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from models import (
	Action,
	EnvironmentState,
	Observation,
	Reward,
	StepResponse,
	TaskDefinition,
	TaskSummary,
	TaskTarget,
	Ticket,
)


def _clamp01(value: float) -> float:
	return max(0.0, min(1.0, value))


class SupportTriageEnvironment:
	def __init__(self) -> None:
		self.max_steps = 8
		self._tasks = self._build_tasks()
		self._task_order = ["task_easy_password", "task_medium_double_charge", "task_hard_enterprise_outage"]
		self._task_index = 0
		self._state: Optional[EnvironmentState] = None

	def _build_tasks(self) -> Dict[str, TaskDefinition]:
		return {
			"task_easy_password": TaskDefinition(
				task_id="task_easy_password",
				difficulty="easy",
				title="Password Reset Triage",
				objective="Route a simple account-access request and provide an actionable response.",
				ticket=Ticket(
					ticket_id="CS-1001",
					customer_name="Ava Patel",
					customer_tier="free",
					subject="Can't log in after password reset",
					message=(
						"I reset my password but still can't sign in from my laptop. "
						"Can you help me regain access today?"
					),
					product_area="authentication",
				),
				target=TaskTarget(
					priority="medium",
					queue="account",
					resolution_code="awaiting_customer_confirmation",
					reply_keywords=["reset link", "clear cache", "confirm if resolved"],
				),
			),
			"task_medium_double_charge": TaskDefinition(
				task_id="task_medium_double_charge",
				difficulty="medium",
				title="Billing Double Charge",
				objective="Handle a billing dispute and set clear next actions for finance review.",
				ticket=Ticket(
					ticket_id="CS-2044",
					customer_name="Jordan Rivera",
					customer_tier="pro",
					subject="Charged twice for March subscription",
					message=(
						"I was charged twice this month. I already sent screenshots to your bot and "
						"need this corrected before end of week."
					),
					product_area="payments",
				),
				target=TaskTarget(
					priority="high",
					queue="billing",
					resolution_code="billing_investigation_opened",
					reply_keywords=["invoice", "refund", "24 hours"],
				),
			),
			"task_hard_enterprise_outage": TaskDefinition(
				task_id="task_hard_enterprise_outage",
				difficulty="hard",
				title="Enterprise Production Outage",
				objective=(
					"Escalate a business-critical outage for an enterprise account and communicate incident handling."
				),
				ticket=Ticket(
					ticket_id="CS-9988",
					customer_name="Nadia Stein",
					customer_tier="enterprise",
					subject="All webhook deliveries failing in production",
					message=(
						"Our payment workflows are down because webhook deliveries started failing 20 minutes ago. "
						"This is a production outage impacting checkout. We need an urgent update and incident bridge."
					),
					product_area="integrations",
				),
				target=TaskTarget(
					priority="urgent",
					queue="technical",
					resolution_code="incident_escalated",
					reply_keywords=["incident", "war room", "next update in 15 minutes"],
				),
			),
		}

	def tasks(self) -> List[TaskSummary]:
		return [
			TaskSummary(
				task_id=task.task_id,
				difficulty=task.difficulty,
				title=task.title,
				objective=task.objective,
			)
			for task in self._tasks.values()
		]

	def reset(self, task_id: Optional[str] = None) -> Observation:
		if task_id is None:
			task_id = self._task_order[self._task_index]
			self._task_index = (self._task_index + 1) % len(self._task_order)

		if task_id not in self._tasks:
			raise ValueError(f"Unknown task_id: {task_id}")

		task = deepcopy(self._tasks[task_id])
		self._state = EnvironmentState(active_task=task, step_count=0, max_steps=self.max_steps)
		return self._observation()

	def _require_state(self) -> EnvironmentState:
		if self._state is None:
			raise RuntimeError("Environment has not been reset.")
		return self._state

	def state(self) -> EnvironmentState:
		return self._require_state()

	def _observation(self) -> Observation:
		state = self._require_state()
		return Observation(
			task_id=state.active_task.task_id,
			difficulty=state.active_task.difficulty,
			objective=state.active_task.objective,
			step_count=state.step_count,
			max_steps=state.max_steps,
			ticket=state.active_task.ticket,
			current_priority=state.current_priority,
			current_queue=state.current_queue,
			reply_draft=state.reply_draft,
			last_action=None,
			action_history=list(state.action_history),
		)

	def _keyword_coverage(self, text: Optional[str], keywords: List[str]) -> float:
		if not text:
			return 0.0
		lowered = text.lower()
		matched = sum(1 for kw in keywords if kw.lower() in lowered)
		return matched / max(1, len(keywords))

	def _grader_score(self, state: EnvironmentState) -> float:
		target = state.active_task.target
		priority_score = 1.0 if state.current_priority == target.priority else 0.0
		queue_score = 1.0 if state.current_queue == target.queue else 0.0
		resolution_score = 1.0 if state.resolution_code == target.resolution_code else 0.0
		reply_score = self._keyword_coverage(state.reply_draft, target.reply_keywords)

		weighted = (
			0.25 * priority_score
			+ 0.25 * queue_score
			+ 0.30 * reply_score
			+ 0.20 * resolution_score
		)
		return _clamp01(weighted)

	def _progress_signals(self, state: EnvironmentState) -> Tuple[float, Dict[str, float]]:
		target = state.active_task.target
		priority_signal = 1.0 if state.current_priority == target.priority else 0.0
		queue_signal = 1.0 if state.current_queue == target.queue else 0.0
		reply_signal = self._keyword_coverage(state.reply_draft, target.reply_keywords)
		resolution_signal = 1.0 if state.resolution_code == target.resolution_code else 0.0

		components = {
			"priority": 0.25 * priority_signal,
			"queue": 0.25 * queue_signal,
			"reply_quality": 0.30 * reply_signal,
			"resolution": 0.20 * resolution_signal,
		}
		return sum(components.values()), components

	def step(self, action: Action) -> StepResponse:
		state = self._require_state()
		if state.resolved or state.step_count >= state.max_steps:
			score = self._grader_score(state)
			obs = self._observation()
			obs.last_action = action
			return StepResponse(
				observation=obs,
				reward=Reward(score=score, components={}, reason="Episode already complete."),
				done=True,
				info={"task_score": score, "max_steps_reached": state.step_count >= state.max_steps},
			)

		old_progress, _ = self._progress_signals(state)
		penalty = 0.0
		reason = "Action accepted."

		action_signature = action.model_dump_json(exclude_none=True)
		if state.action_history and state.action_history[-1] == action_signature:
			penalty += 0.03
			reason = "Repeated action penalty."

		if action.action_type == "classify_priority":
			if action.priority is None:
				penalty += 0.05
				reason = "Missing priority value."
			else:
				state.current_priority = action.priority
		elif action.action_type == "assign_queue":
			if action.queue is None:
				penalty += 0.05
				reason = "Missing queue value."
			else:
				state.current_queue = action.queue
		elif action.action_type == "draft_reply":
			if not action.reply_text or len(action.reply_text.strip()) < 20:
				penalty += 0.04
				reason = "Reply too short to be useful."
			else:
				state.reply_draft = action.reply_text.strip()
		elif action.action_type == "add_internal_note":
			if not action.note:
				penalty += 0.03
				reason = "Internal note is empty."
			else:
				state.internal_notes.append(action.note.strip())
		elif action.action_type == "resolve_ticket":
			state.resolved = True
			if action.resolution_code:
				state.resolution_code = action.resolution_code
			else:
				penalty += 0.05
				reason = "Resolution requires resolution_code."
		elif action.action_type == "noop":
			penalty += 0.02
			reason = "No-op penalty to discourage stalling."
		else:
			penalty += 0.08
			reason = "Unknown action penalty."

		state.step_count += 1
		state.action_history.append(action_signature)
		new_progress, components = self._progress_signals(state)
		progress_delta = new_progress - old_progress
		dense_reward = _clamp01(progress_delta + (0.05 if state.resolved else 0.0) - penalty)

		state.cumulative_penalty += penalty
		state.cumulative_reward = _clamp01(state.cumulative_reward + dense_reward)

		done = state.resolved or state.step_count >= state.max_steps
		task_score = self._grader_score(state)

		observation = self._observation()
		observation.last_action = action
		reward = Reward(score=dense_reward, components=components, reason=reason)

		info = {
			"task_score": task_score,
			"progress": new_progress,
			"penalty": penalty,
			"cumulative_reward": state.cumulative_reward,
			"done_reason": "resolved" if state.resolved else ("max_steps" if done else "ongoing"),
		}

		return StepResponse(observation=observation, reward=reward, done=done, info=info)
