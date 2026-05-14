from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .decision import AgentDecision, GateDecision


@dataclass
class DecisionOutcome:
    decision_id: str
    decision: dict
    gate_result: dict

    metric_deltas: dict[str, float] | None = None
    predicted_deltas_match: dict[str, str] | None = None

    was_beneficial: bool | None = None
    rolled_back: bool = False
    timestamp: str = ""

    def _to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def _from_dict(cls, d: dict) -> "DecisionOutcome":
        return cls(**d)


class OutcomeLog:
    def __init__(self, path: Path = Path("outputs/agent_decisions.jsonl")) -> None:
        self._path = path

    def log(self, outcome: DecisionOutcome) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(outcome._to_dict()) + "\n")

    def load_all(self) -> list[DecisionOutcome]:
        if not self._path.exists():
            return []
        outcomes: list[DecisionOutcome] = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    outcomes.append(DecisionOutcome._from_dict(json.loads(line)))
        return outcomes

    def query(
        self,
        *,
        action: str | None = None,
        certainty: str | None = None,
        ground_truth_sampled: bool | None = None,
        only_committed: bool = False,
    ) -> list[DecisionOutcome]:
        results = self.load_all()
        filtered: list[DecisionOutcome] = []
        for o in results:
            if action is not None and o.decision.get("action") != action:
                continue
            if certainty is not None and o.decision.get("certainty") != certainty:
                continue
            if ground_truth_sampled is not None and o.decision.get("ground_truth_sampled") != ground_truth_sampled:
                continue
            if only_committed and o.rolled_back:
                continue
            filtered.append(o)
        return filtered
