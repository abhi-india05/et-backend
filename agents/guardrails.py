from __future__ import annotations

from typing import Any, Dict, Iterable, Type, TypeVar

from pydantic import BaseModel, ValidationError

from backend.models.schemas import WorkflowValidation
from backend.utils.helpers import extract_json_from_text, safe_json_loads

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_llm_json(raw_text: str, model_type: Type[ModelT]) -> ModelT:
    parsed = safe_json_loads(raw_text) or extract_json_from_text(raw_text)
    if parsed is None:
        raise ValueError("LLM response did not contain valid JSON")
    try:
        return model_type.model_validate(parsed)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def validate_tools_used(*, allowed_tools: Iterable[str], agent_outputs: Dict[str, Any]) -> WorkflowValidation:
    allowed = {tool for tool in allowed_tools}
    validation = WorkflowValidation(valid=True)
    for agent_name, output in agent_outputs.items():
        if not isinstance(output, dict):
            continue
        used_tools = output.get("tools_used", []) or []
        invalid_tools = sorted(set(used_tools) - allowed)
        if invalid_tools:
            validation.valid = False
            validation.errors.append(f"{agent_name} attempted disallowed tools: {', '.join(invalid_tools)}")
    return validation
