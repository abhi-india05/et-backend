import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from backend.config.settings import settings
from backend.utils.helpers import build_agent_response, now_iso
from backend.utils.logger import get_logger, record_audit

logger = get_logger("failure_recovery")


class FailureRecoveryEngine:
    def __init__(self):
        self.max_retries = settings.max_retries
        self.retry_delay = settings.retry_delay
        self.escalation_threshold = 2
        self.failed_tasks: List[Dict[str, Any]] = []
        self.recovered_tasks: List[Dict[str, Any]] = []

    def execute_with_recovery(
        self,
        task_name: str,
        task_fn: Callable,
        task_args: Dict[str, Any],
        session_id: str,
        fallback_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
       
        last_error: Optional[str] = None
        last_result: Optional[Dict[str, Any]] = None

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.info(
                        "task_retry",
                        task=task_name,
                        attempt=attempt + 1,
                        delay=delay,
                        session_id=session_id,
                    )
                    self._safe_sleep(delay)

                result = task_fn(**task_args)
                last_result = result

                if isinstance(result, dict) and result.get("status") == "failure":
                    last_error = result.get("error") or result.get("reasoning", "Agent returned failure")
                    raise RuntimeError(last_error)

                if attempt > 0:
                    self.recovered_tasks.append({
                        "task": task_name,
                        "session_id": session_id,
                        "attempts": attempt + 1,
                        "recovered_at": now_iso(),
                    })
                    logger.info("task_recovered", task=task_name, attempts=attempt + 1)

                return result

            except RuntimeError as e:
                last_error = str(e)
            except Exception as e:
                last_error = str(e)
                logger.warning(
                    "task_attempt_failed",
                    task=task_name,
                    attempt=attempt + 1,
                    error=last_error,
                )

        should_escalate = self.max_retries >= self.escalation_threshold - 1
        self.failed_tasks.append({
            "task": task_name,
            "session_id": session_id,
            "attempts": self.max_retries + 1,
            "last_error": last_error,
            "failed_at": now_iso(),
            "escalated": should_escalate,
        })

        if fallback_fn is not None:
            try:
                logger.info("fallback_attempt", task=task_name)
                fallback_result = fallback_fn(**task_args)
                fallback_result["_recovered_via_fallback"] = True
                record_audit(
                    session_id=session_id,
                    agent_name="failure_recovery",
                    action=f"fallback_{task_name}",
                    input_summary=f"Task: {task_name} failed after {self.max_retries + 1} attempts",
                    output_summary="Fallback succeeded",
                    status="success",
                )
                return fallback_result
            except Exception as fe:
                logger.error("fallback_failed", task=task_name, error=str(fe))

        record_audit(
            session_id=session_id,
            agent_name="failure_recovery",
            action=f"exhausted_{task_name}",
            input_summary=f"Task: {task_name}",
            output_summary=f"Failed after {self.max_retries + 1} attempts. Error: {str(last_error)[:120]}",
            status="failure",
        )

        return build_agent_response(
            status="escalated" if should_escalate else "failure",
            data={
                "task": task_name,
                "retry_count": self.max_retries + 1,
                "last_error": last_error,
                "escalated": should_escalate,
                "escalation_reason": (
                    f"Task '{task_name}' failed {self.max_retries + 1} times. "
                    "Manual intervention required."
                ) if should_escalate else "",
            },
            reasoning=(
                f"Task '{task_name}' exhausted all {self.max_retries + 1} attempts. "
                f"Last error: {last_error}. "
                f"{'ESCALATED for human review.' if should_escalate else 'Marked as failed.'}"
            ),
            confidence=0.0,
            agent_name="failure_recovery",
            error=last_error,
        )

    @staticmethod
    def _safe_sleep(seconds: float) -> None:
        """
        Sleep without blocking the asyncio event loop.
        When called from a thread (via run_in_executor), time.sleep is correct.
        When called directly in a sync context, also fine.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                time.sleep(seconds) 
        except RuntimeError:
            time.sleep(seconds)

    def get_recovery_report(self) -> Dict[str, Any]:
        total_failed = len(self.failed_tasks)
        total_recovered = len(self.recovered_tasks)
        return {
            "total_failed": total_failed,
            "total_recovered": total_recovered,
            "escalated": len([t for t in self.failed_tasks if t.get("escalated")]),
            "recovery_rate": round(
                total_recovered / max(total_recovered + total_failed, 1), 3
            ),
            "failed_tasks": self.failed_tasks[-20:],    # Cap list size
            "recovered_tasks": self.recovered_tasks[-20:],
            "report_time": now_iso(),
        }

    def reset(self) -> None:
        self.failed_tasks = []
        self.recovered_tasks = []


def run_failure_recovery(
    failed_agent: str,
    session_id: str,
    agent_outputs: Dict[str, Any],
) -> Dict[str, Any]:
    logger.info("failure_recovery_triggered", agent=failed_agent, session_id=session_id)

    playbook: Dict[str, Dict[str, List[str]]] = {
        "prospecting": {
            "actions": ["Fallback to basic enrichment without LLM scoring"],
            "recommendations": ["Retry with simplified company profile", "Use manual LinkedIn search"],
        },
        "outreach": {
            "actions": ["Generate single-email template sequence"],
            "recommendations": ["Assign to human SDR for manual outreach"],
        },
        "deal_intelligence": {
            "actions": ["Apply rule-based inactivity thresholds without LLM analysis"],
            "recommendations": ["Alert sales manager via Slack", "Flag for manual review"],
        },
        "churn": {
            "actions": ["Return raw weighted churn scores without LLM retention strategies"],
            "recommendations": ["Alert Customer Success team", "Export at-risk list to CSV"],
        },
        "action": {
            "actions": ["Queue emails for manual send", "Log CRM updates for manual entry"],
            "recommendations": ["Store failed actions in escalation queue"],
        },
        "crm_auditor": {
            "actions": ["Return raw anomaly data without LLM recommendations"],
            "recommendations": ["Manually review pipeline report"],
        },
    }

    matched_key = next(
        (k for k in playbook if k in failed_agent.lower()),
        None,
    )
    entry = playbook.get(matched_key, {
        "actions": ["Log error for engineering investigation"],
        "recommendations": ["Contact engineering team"],
    })

    successful_agents = [
        name for name, out in agent_outputs.items()
        if isinstance(out, dict) and out.get("status") == "success"
    ]

    result = build_agent_response(
        status="success",
        data={
            "failed_agent": failed_agent,
            "recovery_actions": entry["actions"],
            "recommendations": entry["recommendations"],
            "successful_agents": successful_agents,
            "escalated": True,
            "escalation_message": (
                f"Agent '{failed_agent}' failed and requires attention. "
                "Recovery actions logged."
            ),
            "timestamp": now_iso(),
        },
        reasoning=(
            f"Recovery playbook applied for '{failed_agent}'. "
            f"{len(entry['actions'])} action(s), {len(entry['recommendations'])} recommendation(s). "
            "Escalated for human review."
        ),
        confidence=0.70,
        agent_name="failure_recovery",
    )

    record_audit(
        session_id=session_id,
        agent_name="failure_recovery",
        action="recovery_plan_generated",
        input_summary=f"Failed agent: {failed_agent}",
        output_summary="Recovery plan generated and escalated",
        status="success",
        reasoning=result["reasoning"],
        confidence=0.70,
    )
    return result


_engine_instance: Optional[FailureRecoveryEngine] = None


def get_recovery_engine() -> FailureRecoveryEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = FailureRecoveryEngine()
    return _engine_instance