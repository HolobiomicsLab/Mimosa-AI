"""
State schema for Mimosa workflow assembly.
The code is not imported directly. It is loaded by the workflow factory and used in a Python environment as part of the crafted workflow.
"""

from typing import TypedDict, List, Union, Any
from pydantic import BaseModel, model_validator
import ast
import json

class Answer(BaseModel):
    status: str
    message: str = ""
    retry_advice: str = ""
    # FIX (2026-10-06): Optional with None default — agents emit "error": null
    # when nothing went wrong (per workflow_v15.md's "error details if any"),
    # and pydantic 2.13 rejects None for str fields. Same latent risk for
    # message/retry_advice — handled by the _null_to_empty validator below.
    error: Union[str, None] = None

    @model_validator(mode="before")
    @classmethod
    def _null_to_empty(cls, data: Any) -> Any:
        """Convert explicit None values to empty strings for str fields.

        Agents following the completion protocol emit JSON null for optional
        fields when nothing went wrong; pydantic rejects None for str.
        """
        if isinstance(data, dict):
            return {k: ("" if v is None and k != "status" else v)
                    for k, v in data.items()}
        return data

    @classmethod
    def validate(cls, data: Union[str, dict, Any]) -> 'Answer':
        """
        Robust parsing that handles malformed JSON from agents
        """
        if isinstance(data, str):
            cleaned_data = cls._clean_malformed_json(data)
            try:
                return cls.model_validate_json(cleaned_data)
            except (json.JSONDecodeError, ValueError):
                try:
                    parsed = ast.literal_eval(data)
                    if isinstance(parsed, dict):
                        return cls.model_validate(parsed)
                    else:
                        raise ValueError("Parsed string is not a dict")
                except Exception as e:
                    return cls._extract_from_malformed(data)
        elif isinstance(data, dict):
            # FIX: fall back to from_raw on validation failure — matching
            # the class docstring's "Robust parsing" promise (the str branch
            # already has graceful fallbacks; this was the unprotected path).
            try:
                return cls.model_validate(data)
            except Exception:
                return cls.from_raw(data)
        else:
            return cls(status="ERROR", message=f"Invalid data type: {type(data)}")
    
    @staticmethod
    def _clean_malformed_json(json_str: str) -> str:
        """Fix common JSON malformation issues"""
        import re
        
        pattern = r'"error":\s*"(\{[^}]*\})"'
        
        def escape_nested_json(match):
            nested_json = match.group(1)
            escaped = nested_json.replace('"', '\\"')
            return f'"error": "{escaped}"'
        
        return re.sub(pattern, escape_nested_json, json_str)
    
    @staticmethod
    def _extract_from_malformed(data: str) -> 'Answer':
        """Extract fields from completely malformed JSON"""
        import re
        
        status_match = re.search(r'"status":\s*"([^"]*)"', data)
        message_match = re.search(r'"message":\s*"([^"]*)"', data)
        
        return Answer(
            status=status_match.group(1) if status_match else "ERROR",
            message=message_match.group(1) if message_match else "Failed to parse response",
            retry_advice="Agent returned malformed JSON - review prompt instructions"
        )

    @classmethod
    def from_raw(cls, raw_data: Any) -> 'Answer':
        """Safe factory method that never fails"""
        try:
            if isinstance(raw_data, str):
                return cls.model_validate_json(raw_data)
            elif isinstance(raw_data, dict):
                return cls.model_validate(raw_data)
            else:
                return cls(
                    status="INVALID_TYPE",
                    message=f"Expected JSON string or dict, got {type(raw_data)}",
                    retry_advice="Ensure agent returns valid JSON"
                )
        except Exception as e:
            return cls(
                status="PARSE_ERROR",
                message=f"Failed to parse: {str(raw_data)[:200]}{'...' if len(str(raw_data)) > 200 else ''}",
                retry_advice=f"JSON parsing failed: {str(e)}"
            )

class Action(TypedDict):
    tool: str

class Observation(TypedDict):
    data: str

class WorkflowState(TypedDict):
    workflow_uuid: str
    model_id: str
    goal: str
    step_name: List[str]
    task_prompt: List[str]
    actions: List[Action]
    observations: List[Observation]
    answers: List[str]
    success: List[bool]

MAX_CONSECUTIVE_FALLBACKS = 3
MAX_CONSECUTIVE_RETRY = 5

def master_router(state: WorkflowState) -> str:
    raw_answer = state["answers"][-1]
    try:
        last_answer = Answer.validate(raw_answer)
    except Exception as e:
        print(f"❌ Failed to validate answer format of\n: {raw_answer}\n")
        last_answer = Answer.from_raw(raw_answer)

    current_agent = state["step_name"][-1]

    if "SUCCESS" in last_answer.status or "SUCCESS" in last_answer.message:
        print(f"✅ Success from '{current_agent}'. Proceeding.")
        return "next_node"
    elif "FALLBACK" in last_answer.status or "FALLBACK" in last_answer.message:
        retry_count = state["step_name"][-5:].count(current_agent)
        if retry_count >= MAX_CONSECUTIVE_FALLBACKS:
            print(f"❌ Detected fallback infinite loop: {retry_count} out of {MAX_CONSECUTIVE_FALLBACKS}. Aborting.")
            return END
        print(f"⏪ Fallback from '{current_agent}' to previous agent..")
        return "fallback_node"
    elif "RETRY" in last_answer.status or "RETRY" in last_answer.message:
        retry_count = state["step_name"][-5:].count(current_agent)
        if retry_count >= MAX_CONSECUTIVE_RETRY:
            print(f"❌ Detected retry infinite loop: {retry_count} out of {MAX_CONSECUTIVE_RETRY}. Aborting.")
            return END
        return "retry_node"
    elif "FAILURE" in last_answer.status:
        print(f"❌ Failure from '{current_agent}'. Aborting.")
        return END
    else :
        print(f"⛔ Protocol violation from '{current_agent}'. Agent must specify SUCCESS/RETRY/FALLBACK/FAILURE. Terminating.")
        return END

def debate_router(state: WorkflowState) -> str:
    recent_answers = state["answers"][-3:]  # Last 3 agents
    votes = [json.loads(a).get("verdict") for a in recent_answers]
    
    if votes.count("APPROVE") >= 2:
        return "next_node"
    elif len(recent_answers) < 9:  # Max 3 debate rounds
        return "another_round"
    else:
        return "fallback_node"