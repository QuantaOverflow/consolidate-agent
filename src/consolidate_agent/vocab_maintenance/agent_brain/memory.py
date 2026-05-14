from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

_WORKING_MAX = 8
_HISTORY_MAX = 15
_FACTS_MAX = 30
_RECENT_OPS_MAX = 3
_RENDER_MAX_CHARS = 10_000
_HISTORY_RENDER_MAX = 8

_PATTERN_A_MIN_REVIEWS = 2
_PATTERN_B_MIN_REPEATS = 3
_PATTERN_C_WORKING_THRESHOLD = 6


@dataclass
class ToolResult:
    tool: str
    args: dict
    result: Any  # dict | list | str
    round_idx: int
    timestamp: str  # ISO 8601 UTC

    def render_full(self) -> str:
        return (
            f"[R{self.round_idx}] tool={self.tool} ts={self.timestamp}\n"
            f"  args: {self.args}\n"
            f"  result: {self.result}"
        )

    def render_brief(self) -> str:
        result_preview = str(self.result)
        if len(result_preview) > 80:
            result_preview = result_preview[:77] + "..."
        return f"[R{self.round_idx}] {self.tool}({self.args}) → {result_preview}"


@dataclass
class RoundSummary:
    round_idx: int
    tools_called: list[str]
    target: str | None
    decision_action: str | None
    decision_certainty: str | None
    outcome: str | None
    committed: bool
    rolled_back: bool

    def compact(self) -> str:
        i = self.round_idx
        target = self.target or "?"
        tools_preview = self.tools_called[:3]
        action = self.decision_action

        if action is None or action == "inspect_more":
            return f"R{i}: inspected {target} (tools: {tools_preview})"

        if action == "stop":
            reason = self.outcome or "criteria met"
            return f"R{i}: stopped — {reason}"

        if action == "split":
            if self.committed:
                return f"R{i}: split {target} | {self.outcome or ''}"
            if self.rolled_back:
                return f"R{i}: proposed split {target} but rolled back ({self.outcome or 'failed'})"
            return f"R{i}: proposed split {target} (pending)"

        # refine / merge / deprecate
        if self.committed:
            return f"R{i}: {action} {target} | {self.outcome or ''}"
        if self.rolled_back:
            return f"R{i}: proposed {action} {target} but rolled back ({self.outcome or 'failed'})"
        return f"R{i}: proposed {action} {target} (pending)"


@dataclass
class NetworkState:
    tag_sizes: dict[str, int]
    recent_operations: dict[str, list[str]]
    total_records: int
    total_edges: int
    triage_report: Any = None  # TriageReport | None — typed Any to avoid import cycle

    def render(self) -> str:
        lines = [
            f"total_records={self.total_records} total_edges={self.total_edges}",
            f"tags({len(self.tag_sizes)}): "
            + ", ".join(f"{k}:{v}" for k, v in list(self.tag_sizes.items())[:10]),
        ]
        if len(self.tag_sizes) > 10:
            lines[-1] += f" ... (+{len(self.tag_sizes) - 10} more)"
        if self.recent_operations:
            ops_preview = {k: v for k, v in list(self.recent_operations.items())[:5]}
            lines.append(f"recent_ops: {ops_preview}")
        return "\n".join(lines)


def _detect_critical_patterns(memory: "AgentMemory") -> str | None:
    """Returns a CRITICAL alert string if patterns detected, else None."""
    alerts: list[str] = []

    # Pattern A: same target+action proposed and REVIEW-gated in >= 2 of last 3 rounds
    last3 = list(memory.history)[-3:]
    if last3:
        # Count (action, target) pairs that ended in pending (rolled_back=False, committed=False)
        review_counts: dict[tuple[str, str], int] = {}
        for rs in last3:
            if (
                rs.decision_action not in (None, "stop", "inspect_more")
                and rs.target
                and not rs.committed
                and not rs.rolled_back
                and rs.outcome
                and "gate=REVIEW" in (rs.outcome or "")
            ):
                key = (rs.decision_action, rs.target)
                review_counts[key] = review_counts.get(key, 0) + 1

        for (action, target), count in review_counts.items():
            if count >= _PATTERN_A_MIN_REVIEWS:
                gate_name = "high_cert_no_preview"
                alerts.append(
                    f"Pattern A: You have proposed `{action} {target}` {count} times in last "
                    f"{len(last3)} rounds. All BLOCKED by gate `{gate_name}`.\n"
                    f"  DO NOT propose the same action+target again.\n"
                    f"  Options:\n"
                    f"    (a) Call `propose_{action}_preview` first (if not yet called)\n"
                    f"    (b) Pick a DIFFERENT target\n"
                    f"    (c) action='inspect_more' if you need more info\n"
                    f"    (d) action='stop' if network is good enough"
                )

    # Pattern B: same tool+args called >= 3 times across last 3 rounds working results
    # Scan history tool calls (from round summaries) — but we only have working list for current round
    # and history has tools_called list (names only). Use working list + synthesize from history.
    tool_call_counts: dict[str, int] = {}
    for rs in list(memory.history)[-3:]:
        for tool_name in rs.tools_called:
            # history only stores names; track by name
            key = json.dumps({"tool": tool_name}, sort_keys=True)
            tool_call_counts[key] = tool_call_counts.get(key, 0) + 1

    # Also check working (current round) for exact tool+args duplicates
    working_key_counts: dict[str, int] = {}
    for tr in memory.working:
        key = json.dumps({"tool": tr.tool, "args": tr.args}, sort_keys=True, default=str)
        working_key_counts[key] = working_key_counts.get(key, 0) + 1

    for key, count in working_key_counts.items():
        if count >= _PATTERN_B_MIN_REPEATS:
            parsed = json.loads(key)
            alerts.append(
                f"Pattern B: You called `{parsed['tool']}({parsed.get('args', {})})` "
                f"{count} times already. Result is the same.\n"
                f"  Stop calling it; use a DIFFERENT tool or different args."
            )

    # Check same tool+args across history rounds (using tools_called names, count by name)
    for key_str, count in tool_call_counts.items():
        if count >= _PATTERN_B_MIN_REPEATS:
            parsed = json.loads(key_str)
            tool_name = parsed["tool"]
            # Avoid duplicate alert if already caught in working
            already_alerted = any(
                f"`{tool_name}(" in a for a in alerts
            )
            if not already_alerted:
                alerts.append(
                    f"Pattern B: You called `{tool_name}` {count} times across last "
                    f"{len(list(memory.history)[-3:])} rounds. Result is the same.\n"
                    f"  Stop calling it; use a DIFFERENT tool or different args."
                )

    # Pattern C: worker memory at >= threshold but no decision yet
    if len(memory.working) >= _PATTERN_C_WORKING_THRESHOLD:
        alerts.append(
            f"Pattern C: You've used {len(memory.working)}/{_WORKING_MAX} tools this round. "
            f"Either decide now or explicitly action='inspect_more' to extend exploration."
        )

    if not alerts:
        return None

    body = "\n".join(alerts)
    return (
        "⚠⚠⚠ CRITICAL PATTERNS DETECTED — READ FIRST ⚠⚠⚠\n\n"
        + body
        + "\n\n⚠⚠⚠ END CRITICAL PATTERNS ⚠⚠⚠"
    )


class AgentMemory:
    def __init__(self, initial_state: NetworkState) -> None:
        self.working: list[ToolResult] = []
        self.history: list[RoundSummary] = []
        self.facts: list[str] = []
        self.state: NetworkState = initial_state
        self.current_round_idx: int = 0

    def record_tool_call(self, tool: str, args: dict, result: Any) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        tr = ToolResult(
            tool=tool,
            args=args,
            result=result,
            round_idx=self.current_round_idx,
            timestamp=ts,
        )
        self.working.append(tr)
        if len(self.working) > _WORKING_MAX:
            self.working.pop(0)

    def add_fact(self, fact: str) -> None:
        self.facts.append(fact)
        if len(self.facts) > _FACTS_MAX:
            self.facts.pop(0)

    def archive_round(
        self,
        round_summary: RoundSummary,
        applied_changes: dict | None = None,
    ) -> None:
        self.history.append(round_summary)
        if len(self.history) > _HISTORY_MAX:
            self.history.pop(0)
        self.working.clear()
        self.current_round_idx += 1

        if applied_changes:
            for tag in applied_changes.get("added_tags", []):
                self.state.tag_sizes.setdefault(tag, 0)
                ops = self.state.recent_operations.setdefault(tag, [])
                ops.append("added")
                if len(ops) > _RECENT_OPS_MAX:
                    ops.pop(0)

            for tag in applied_changes.get("removed_tags", []):
                self.state.tag_sizes.pop(tag, None)
                ops = self.state.recent_operations.setdefault(tag, [])
                ops.append("removed")
                if len(ops) > _RECENT_OPS_MAX:
                    ops.pop(0)

            for tag in applied_changes.get("modified_tags", []):
                ops = self.state.recent_operations.setdefault(tag, [])
                ops.append("modified")
                if len(ops) > _RECENT_OPS_MAX:
                    ops.pop(0)

            reassign = applied_changes.get("record_reassignments", 0)
            if reassign:
                self.state.total_records = max(0, self.state.total_records + reassign)

    def render_for_llm(self) -> str:
        sections: list[str] = []

        critical = _detect_critical_patterns(self)
        if critical:
            sections.append(critical)

        if self.state.triage_report is not None:
            sections.append(self.state.triage_report.render())

        sections.append("=== Current network state ===\n" + self.state.render())

        if self.facts:
            facts_text = "\n".join(f"- {f}" for f in self.facts)
            sections.append("=== Run facts ===\n" + facts_text)
        else:
            sections.append("=== Run facts ===\n(none)")

        recent = list(self.history)[-_HISTORY_RENDER_MAX:]
        if recent:
            history_text = "\n".join(r.compact() for r in recent)
            sections.append("=== Recent rounds (last 8) ===\n" + history_text)
        else:
            sections.append("=== Recent rounds (last 8) ===\n(none)")

        if self.working:
            working_text = "\n".join(tr.render_full() for tr in self.working)
            sections.append("=== This round so far ===\n" + working_text)
        else:
            sections.append("=== This round so far ===\n(none)")

        full = "\n\n".join(sections)
        if len(full) <= _RENDER_MAX_CHARS:
            return full

        # Truncate history to fit; find sections by content rather than fixed index
        state_sec = next(s for s in sections if s.startswith("=== Current network state ==="))
        facts_sec = next(s for s in sections if s.startswith("=== Run facts ==="))
        working_sec = next(s for s in sections if s.startswith("=== This round so far ==="))
        critical_secs = [s for s in sections if s.startswith("⚠⚠⚠ CRITICAL")]
        triage_secs = [s for s in sections if s.startswith("=== Network triage ===")]
        prefix_parts = critical_secs + triage_secs + [state_sec, facts_sec, "(history truncated)", working_sec]
        base = "\n\n".join(prefix_parts)
        budget = _RENDER_MAX_CHARS - len(base) - 4  # 4 for "\n\n"

        if recent and budget > 0:
            truncated_history = "=== Recent rounds (last 8) ===\n"
            added = ""
            for r in reversed(recent):
                line = r.compact() + "\n"
                if len(truncated_history) + len(added) + len(line) > budget:
                    break
                added = line + added
            truncated_history += added.rstrip() or "(truncated)"
            result = "\n\n".join(critical_secs + triage_secs + [state_sec, facts_sec, truncated_history, working_sec])
        else:
            result = base

        return result[:_RENDER_MAX_CHARS]

    def context_size_estimate(self) -> int:
        return len(self.render_for_llm()) // 4
