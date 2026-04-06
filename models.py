from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
	"classify_priority",
	"assign_queue",
	"draft_reply",
	"add_internal_note",
	"resolve_ticket",
	"noop",
]

Priority = Literal["low", "medium", "high", "urgent"]
Queue = Literal["billing", "technical", "account", "trust_and_safety", "general"]


class Action(BaseModel):
	action_type: ActionType
	priority: Optional[Priority] = None
	queue: Optional[Queue] = None
	reply_text: Optional[str] = None
	note: Optional[str] = None
	resolution_code: Optional[str] = None


class Ticket(BaseModel):
	ticket_id: str
	customer_name: str
	customer_tier: Literal["free", "pro", "enterprise"]
	subject: str
	message: str
	product_area: str


class Observation(BaseModel):
	task_id: str
	difficulty: Literal["easy", "medium", "hard"]
	objective: str
	step_count: int
	max_steps: int
	ticket: Ticket
	current_priority: Optional[Priority] = None
	current_queue: Optional[Queue] = None
	reply_draft: Optional[str] = None
	last_action: Optional[Action] = None
	action_history: List[str] = Field(default_factory=list)


class Reward(BaseModel):
	score: float = Field(ge=0.0, le=1.0)
	components: Dict[str, float] = Field(default_factory=dict)
	reason: str


class StepResponse(BaseModel):
	observation: Observation
	reward: Reward
	done: bool
	info: Dict[str, Any] = Field(default_factory=dict)


class ResetRequest(BaseModel):
	task_id: Optional[str] = None


class TaskSummary(BaseModel):
	task_id: str
	difficulty: Literal["easy", "medium", "hard"]
	title: str
	objective: str


class TaskTarget(BaseModel):
	priority: Priority
	queue: Queue
	resolution_code: str
	reply_keywords: List[str]


class TaskDefinition(BaseModel):
	task_id: str
	difficulty: Literal["easy", "medium", "hard"]
	title: str
	objective: str
	ticket: Ticket
	target: TaskTarget


class EnvironmentState(BaseModel):
	active_task: TaskDefinition
	step_count: int
	max_steps: int
	current_priority: Optional[Priority] = None
	current_queue: Optional[Queue] = None
	reply_draft: Optional[str] = None
	internal_notes: List[str] = Field(default_factory=list)
	resolved: bool = False
	resolution_code: Optional[str] = None
	action_history: List[str] = Field(default_factory=list)
	cumulative_reward: float = 0.0
	cumulative_penalty: float = 0.0
