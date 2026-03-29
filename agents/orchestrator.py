from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

from backend.agents.action_agent import run_action_agent
from backend.agents.churn_agent import run_churn_agent
from backend.agents.crm_auditor_agent import run_crm_auditor_agent
from backend.agents.deal_intelligence_agent import run_deal_intelligence_agent
from backend.agents.digital_twin_agent import run_digital_twin_agent
from backend.agents.explainability_agent import run_explainability_agent
from backend.agents.failure_recovery import get_recovery_engine, run_failure_recovery
from backend.agents.guardrails import validate_tools_used
from backend.agents.outreach_agent import run_outreach_agent
from backend.agents.prospecting_agent import run_prospecting_agent
from backend.config.settings import settings
from backend.models.schemas import (
    DealRisk,
    DigitalTwinProfileOutput,
    EmailSequenceResult,
    ExecutionPlan,
    ExplainabilityOutput,
    ProspectingOutput,
    WorkflowValidation,
)
from backend.utils.helpers import build_agent_response, generate_session_id, now_iso
from backend.utils.logger import get_logger, record_audit

logger = get_logger("orchestrator")


def _terminal_agent_status(agent_name: str, output: Dict[str, Any]) -> None:
    status = str(output.get("status", "unknown")).lower()
    tag = "SUCCESS" if status == "success" else "FAILURE"
    print(f"[BACKEND][{agent_name}][{tag}] status={status}")


def _record_agent_started(session_id: str, agent_name: str, input_summary: str) -> None:
    record_audit(
        session_id=session_id,
        agent_name=agent_name,
        action="agent_started",
        input_summary=input_summary,
        output_summary=f"{agent_name} execution started.",
        status="pending",
        reasoning=f"Starting {agent_name}.",
        confidence=0.0,
    )


def _pause_between_agents(session_id: str, previous_agent: str, next_agent: str) -> None:
    delay_seconds = float(settings.agent_inter_call_delay_seconds or 0.0)
    if delay_seconds <= 0.0:
        return

    print(
        f"[BACKEND][orchestrator][INFO] cooldown={delay_seconds:.2f}s previous={previous_agent} next={next_agent}"
    )
    record_audit(
        session_id=session_id,
        agent_name="orchestrator",
        action="agent_cooldown",
        input_summary=f"{previous_agent} -> {next_agent}",
        output_summary=f"Cooldown {delay_seconds:.2f}s before running {next_agent}.",
        status="pending",
        reasoning="Applying configured delay between sequential agent executions to reduce burst rate-limit failures.",
        confidence=0.0,
    )
    time.sleep(delay_seconds)

ALLOWED_TOOLS_BY_TASK: Dict[str, List[str]] = {
    "cold_outreach": ["scraping_tool", "vector_memory", "llm", "email_tool", "crm_tool"],
    "risk_detection": ["crm_tool", "scraping_tool", "vector_memory", "llm", "email_tool"],
    "churn_prediction": ["crm_tool", "llm", "email_tool"],
}


def _build_plan(task_type: str, input_data: Dict[str, Any]) -> ExecutionPlan:
    product_context = input_data.get("product_context") or {}
    if task_type == "cold_outreach":
        steps = ["prospecting_agent", "digital_twin_agent", "outreach_agent", "action_agent", "crm_auditor_agent", "validator"]
    elif task_type == "risk_detection":
        steps = ["deal_intelligence_agent", "crm_auditor_agent", "action_agent", "validator"]
    elif task_type == "churn_prediction":
        steps = ["churn_agent", "action_agent", "validator"]
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    return ExecutionPlan(
        task_type=task_type,
        allowed_tools=ALLOWED_TOOLS_BY_TASK[task_type],
        steps=steps,
        fallback_strategy="Retry each agent, then degrade gracefully to deterministic fallback output.",
        product_context=product_context,
    )


def _generic_failure(agent_name: str, message: str) -> Dict[str, Any]:
    return build_agent_response(
        status="failure",
        data={},
        reasoning=message,
        confidence=0.0,
        agent_name=agent_name,
        error=message,
    )


def _execute_agent(task_name: str, task_fn, task_args: Dict[str, Any], fallback_message: str) -> Dict[str, Any]:
    return get_recovery_engine().execute_with_recovery(
        task_name=task_name,
        task_fn=task_fn,
        task_args=task_args,
        session_id=str(task_args.get("session_id", "")),
        fallback_fn=lambda **_: _generic_failure(task_name, fallback_message),
    )


def _validate_outputs(plan: ExecutionPlan, agent_outputs: Dict[str, Any]) -> WorkflowValidation:
    validation = validate_tools_used(allowed_tools=plan.allowed_tools, agent_outputs=agent_outputs)
    try:
        if "prospecting_agent" in agent_outputs and agent_outputs["prospecting_agent"].get("status") == "success":
            ProspectingOutput.model_validate(agent_outputs["prospecting_agent"]["data"])
        if "digital_twin_agent" in agent_outputs and agent_outputs["digital_twin_agent"].get("status") == "success":
            for item in agent_outputs["digital_twin_agent"]["data"].get("twin_profiles", []):
                DigitalTwinProfileOutput.model_validate(item)
        if "outreach_agent" in agent_outputs and agent_outputs["outreach_agent"].get("status") == "success":
            for item in agent_outputs["outreach_agent"]["data"].get("sequences", []):
                EmailSequenceResult.model_validate(item)
        if "deal_intelligence_agent" in agent_outputs and agent_outputs["deal_intelligence_agent"].get("status") == "success":
            for item in agent_outputs["deal_intelligence_agent"]["data"].get("risks", []):
                DealRisk.model_validate(item)
        if "explainability_agent" in agent_outputs and agent_outputs["explainability_agent"].get("status") == "success":
            ExplainabilityOutput.model_validate(agent_outputs["explainability_agent"]["data"])
    except Exception as exc:
        validation.valid = False
        validation.errors.append(str(exc))

    for agent_name, output in agent_outputs.items():
        if not isinstance(output, dict):
            continue
        if output.get("status") == "failure":
            validation.warnings.append(f"{agent_name} failed and relied on workflow recovery.")
        if output.get("status") == "escalated":
            validation.warnings.append(f"{agent_name} escalated for human review.")
    return validation


def _compute_impact_metrics(agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if "prospecting_agent" in agent_outputs:
        metrics["leads_identified"] = len(agent_outputs["prospecting_agent"].get("data", {}).get("leads", []))
    if "outreach_agent" in agent_outputs:
        sequences = agent_outputs["outreach_agent"].get("data", {}).get("sequences", [])
        metrics["email_sequences_created"] = len(sequences)
        metrics["total_emails_crafted"] = sum(len(item.get("emails", [])) for item in sequences)
    if "deal_intelligence_agent" in agent_outputs:
        metrics["deals_at_risk"] = agent_outputs["deal_intelligence_agent"].get("data", {}).get("total_at_risk", 0)
    if "churn_agent" in agent_outputs:
        metrics["churn_risks_identified"] = len(agent_outputs["churn_agent"].get("data", {}).get("top_churn_risks", []))
        metrics["arr_at_risk"] = agent_outputs["churn_agent"].get("data", {}).get("total_arr_at_risk", 0)
    if "action_agent" in agent_outputs:
        metrics["emails_sent"] = agent_outputs["action_agent"].get("data", {}).get("emails_sent", 0)
        metrics["crm_updates"] = agent_outputs["action_agent"].get("data", {}).get("crm_updates", 0)
    return metrics


def _execute_workflow(task_type: str, input_data: Dict[str, Any], session_id: str, user_id: Optional[str]) -> Dict[str, Any]:
    plan = _build_plan(task_type, input_data)
    agent_outputs: Dict[str, Any] = {}
    completed_agents: List[str] = []
    failed_agents: List[str] = []
    last_completed_agent: Optional[str] = None

    record_audit(
        session_id=session_id,
        agent_name="orchestrator",
        action="plan_workflow",
        input_summary=f"Task: {task_type}",
        output_summary=f"Planned steps: {', '.join(plan.steps)}",
        status="success",
        reasoning=f"Built planner output for {task_type}.",
        confidence=1.0,
        extra={"plan": plan.model_dump()},
    )

    if task_type == "cold_outreach":
        _record_agent_started(session_id, "prospecting_agent", f"Company: {input_data.get('company', '')}")
        prospecting = _execute_agent(
            "prospecting_agent",
            run_prospecting_agent,
            {
                "company": input_data.get("company", ""),
                "product_context": plan.product_context,
                "industry": input_data.get("industry", ""),
                "company_size": input_data.get("size", ""),
                "session_id": session_id,
                "notes": input_data.get("notes", ""),
                "user_id": user_id,
            },
            "Prospecting failed and no deterministic fallback could be generated.",
        )
        agent_outputs["prospecting_agent"] = prospecting
        _terminal_agent_status("prospecting_agent", prospecting)
        (completed_agents if prospecting.get("status") == "success" else failed_agents).append("prospecting_agent")
        last_completed_agent = "prospecting_agent"

        _pause_between_agents(session_id, "prospecting_agent", "digital_twin_agent")

        _record_agent_started(session_id, "digital_twin_agent", f"Company: {input_data.get('company', '')}")
        twin = _execute_agent(
            "digital_twin_agent",
            run_digital_twin_agent,
            {
                "leads": prospecting.get("data", {}).get("leads", []),
                "company": input_data.get("company", ""),
                "product_context": plan.product_context,
                "industry": input_data.get("industry", ""),
                "session_id": session_id,
                "user_id": user_id,
            },
            "Digital twin simulation failed.",
        )
        agent_outputs["digital_twin_agent"] = twin
        _terminal_agent_status("digital_twin_agent", twin)
        (completed_agents if twin.get("status") == "success" else failed_agents).append("digital_twin_agent")
        last_completed_agent = "digital_twin_agent"

        _pause_between_agents(session_id, "digital_twin_agent", "outreach_agent")

        _record_agent_started(session_id, "outreach_agent", f"Company: {input_data.get('company', '')}")
        outreach = _execute_agent(
            "outreach_agent",
            run_outreach_agent,
            {
                "leads": prospecting.get("data", {}).get("leads", []),
                "twin_profiles": twin.get("data", {}).get("twin_profiles", []),
                "company": input_data.get("company", ""),
                "product_context": plan.product_context,
                "session_id": session_id,
                "user_id": user_id,
            },
            "Outreach generation failed.",
        )
        agent_outputs["outreach_agent"] = outreach
        (completed_agents if outreach.get("status") == "success" else failed_agents).append("outreach_agent")
        last_completed_agent = "outreach_agent"

        _pause_between_agents(session_id, "outreach_agent", "action_agent")

        _record_agent_started(session_id, "action_agent", f"auto_send={bool(input_data.get('auto_send', False))}")
        if not input_data.get("auto_send", False):
            action = build_agent_response(
                status="success",
                data={
                    "action_type": "send_sequences",
                    "executed_actions": [],
                    "total_actions": 0,
                    "emails_sent": 0,
                    "crm_updates": 0,
                    "auto_send": False,
                    "message": "Auto-send disabled. Sequences are ready for human review.",
                    "timestamp": now_iso(),
                },
                reasoning="Skipped automatic email sending because auto_send=false.",
                confidence=1.0,
                agent_name="action_agent",
                tools_used=[],
            )
            record_audit(
                session_id=session_id,
                agent_name="action_agent",
                action="skip_send_sequences",
                input_summary="Auto-send disabled.",
                output_summary="Skipped sequence sending and returned drafts for human review.",
                status="success",
                reasoning="Auto-send disabled by request.",
                confidence=1.0,
            )
        else:
            action = _execute_agent(
                "action_agent",
                run_action_agent,
                {
                    "action_type": "send_sequences",
                    "payload": {"sequences": outreach.get("data", {}).get("sequences", [])},
                    "session_id": session_id,
                    "user_id": user_id,
                },
                "Action execution failed.",
            )
        agent_outputs["action_agent"] = action
        (completed_agents if action.get("status") == "success" else failed_agents).append("action_agent")
        last_completed_agent = "action_agent"

        _pause_between_agents(session_id, "action_agent", "crm_auditor_agent")

        _record_agent_started(session_id, "crm_auditor_agent", "Evaluating CRM health after outreach generation.")
        crm_audit = _execute_agent(
            "crm_auditor_agent",
            run_crm_auditor_agent,
            {"session_id": session_id, "user_id": user_id},
            "CRM auditor failed.",
        )
        agent_outputs["crm_auditor_agent"] = crm_audit
        (completed_agents if crm_audit.get("status") == "success" else failed_agents).append("crm_auditor_agent")
        last_completed_agent = "crm_auditor_agent"

    elif task_type == "risk_detection":
        _record_agent_started(session_id, "deal_intelligence_agent", "Analyzing risk signals for requested deals.")
        deal_intel = _execute_agent(
            "deal_intelligence_agent",
            run_deal_intelligence_agent,
            {
                "deal_ids": input_data.get("deal_ids"),
                "inactivity_threshold": input_data.get("inactivity_threshold_days", 10),
                "customer_engagement_signals": input_data.get("customer_engagement_signals", []),
                "session_id": session_id,
                "user_id": user_id,
            },
            "Deal intelligence failed.",
        )
        agent_outputs["deal_intelligence_agent"] = deal_intel
        (completed_agents if deal_intel.get("status") == "success" else failed_agents).append("deal_intelligence_agent")
        last_completed_agent = "deal_intelligence_agent"

        _pause_between_agents(session_id, "deal_intelligence_agent", "crm_auditor_agent")

        _record_agent_started(session_id, "crm_auditor_agent", "Running CRM audit for detected risk context.")
        crm_audit = _execute_agent(
            "crm_auditor_agent",
            run_crm_auditor_agent,
            {"session_id": session_id, "user_id": user_id},
            "CRM auditor failed.",
        )
        agent_outputs["crm_auditor_agent"] = crm_audit
        (completed_agents if crm_audit.get("status") == "success" else failed_agents).append("crm_auditor_agent")
        last_completed_agent = "crm_auditor_agent"

        high_risks = [
            risk
            for risk in deal_intel.get("data", {}).get("risks", [])
            if risk.get("risk_level") in {"critical", "high"}
        ][:5]
        _pause_between_agents(session_id, "crm_auditor_agent", "action_agent")

        _record_agent_started(session_id, "action_agent", f"Preparing follow-up actions for {len(high_risks)} high-risk deal(s).")
        action = _execute_agent(
            "action_agent",
            run_action_agent,
            {"action_type": "risk_followup", "payload": {"risks": high_risks}, "session_id": session_id, "user_id": user_id},
            "Risk follow-up actions failed.",
        )
        agent_outputs["action_agent"] = action
        (completed_agents if action.get("status") == "success" else failed_agents).append("action_agent")
        last_completed_agent = "action_agent"

    elif task_type == "churn_prediction":
        _record_agent_started(session_id, "churn_agent", "Computing churn risk and retention strategy.")
        churn = _execute_agent(
            "churn_agent",
            run_churn_agent,
            {
                "account_ids": input_data.get("account_ids"),
                "top_n": input_data.get("top_n", 3),
                "customer_engagement_signals": input_data.get("customer_engagement_signals", []),
                "session_id": session_id,
                "user_id": user_id,
            },
            "Churn analysis failed.",
        )
        agent_outputs["churn_agent"] = churn
        (completed_agents if churn.get("status") == "success" else failed_agents).append("churn_agent")
        last_completed_agent = "churn_agent"

        _pause_between_agents(session_id, "churn_agent", "action_agent")

        _record_agent_started(session_id, "action_agent", "Executing retention outreach actions.")
        action = _execute_agent(
            "action_agent",
            run_action_agent,
            {
                "action_type": "retention_outreach",
                "payload": {"churn_risks": churn.get("data", {}).get("top_churn_risks", [])},
                "session_id": session_id,
                "user_id": user_id,
            },
            "Retention outreach actions failed.",
        )
        agent_outputs["action_agent"] = action
        (completed_agents if action.get("status") == "success" else failed_agents).append("action_agent")
        last_completed_agent = "action_agent"

    escalated = False
    if failed_agents:
        recovery = run_failure_recovery(
            failed_agent=failed_agents[-1],
            session_id=session_id,
            agent_outputs=agent_outputs,
        )
        agent_outputs["failure_recovery"] = recovery
        escalated = bool(recovery.get("data", {}).get("escalated", False))
        last_completed_agent = "failure_recovery"

    if last_completed_agent:
        _pause_between_agents(session_id, last_completed_agent, "explainability_agent")

    _record_agent_started(session_id, "explainability_agent", f"Summarizing workflow decisions for task={task_type}.")
    explainability = _execute_agent(
        "explainability_agent",
        run_explainability_agent,
        {"session_id": session_id, "agent_outputs": agent_outputs, "task_type": task_type},
        "Explainability generation failed.",
    )
    agent_outputs["explainability_agent"] = explainability
    if explainability.get("status") == "success":
        completed_agents.append("explainability_agent")
    else:
        failed_agents.append("explainability_agent")

    validation = _validate_outputs(plan, agent_outputs)
    final_status = "completed"
    if failed_agents or not validation.valid:
        final_status = "completed_with_errors"

    final_output = {
        "session_id": session_id,
        "task_type": task_type,
        "status": final_status,
        "completed_agents": completed_agents,
        "failed_agents": failed_agents,
        "escalated": escalated,
        "agent_outputs": agent_outputs,
        "plan": plan.model_dump(),
        "validation": validation.model_dump(),
        "explanation": explainability.get("data", {}),
        "actions_taken": agent_outputs.get("action_agent", {}).get("data", {}),
        "impact_metrics": _compute_impact_metrics(agent_outputs),
        "completed_at": now_iso(),
    }
    record_audit(
        session_id=session_id,
        agent_name="orchestrator",
        action="finalize",
        input_summary=f"Task: {task_type}",
        output_summary=f"Completed with status={final_status}. {len(completed_agents)} agents succeeded, {len(failed_agents)} failed.",
        status="success" if final_status == "completed" else "failure",
        reasoning=f"Workflow '{task_type}' complete.",
        confidence=0.95 if final_status == "completed" else 0.7,
        extra={"validation": validation.model_dump()},
    )
    return final_output


async def run_orchestrator(
    task_type: str,
    input_data: Dict[str, Any],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    if not session_id:
        session_id = generate_session_id()
    logger.info("orchestrator_start", task_type=task_type, session_id=session_id)
    return await asyncio.to_thread(_execute_workflow, task_type, input_data, session_id, user_id)
