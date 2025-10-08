"""
State schema for Mimosa workflow assembly.
The code is not imported directly. It is loaded by the workflow factory and used in a Python environment as part of the crafted workflow.
"""

from typing import TypedDict, List, Union, Any
from pydantic import BaseModel
import ast
import json

class Answer(BaseModel):
    status: str
    message: str = ""
    retry_advice: str = ""
    error: str = ""
    
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
            return cls.model_validate(data)
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

def master_router(state: WorkflowState) -> str:
    raw_answer = state["answers"][-1]
    try:
        last_answer = Answer.validate(raw_answer)
    except Exception as e:
        print(f"❌ Failed to validate answer format of\n: {raw_answer}\n")
        last_answer = Answer.from_raw(raw_answer)

    current_agent = state["step_name"][-1]

    if "SUCCESS" in last_answer.status:
        print(f"✅ Success from '{current_agent}'. Proceeding.")
        return "next_node"
    elif "FALLBACK" in last_answer.status:
         print(f"⏪ Insufficient data from '{current_agent}'. Retrying previous step.")
         return "fallback_node"
    elif "RETRY" in last_answer.status:
        return "retry_node"
    elif "FAILURE" in last_answer.status:
        print(f"❌ Failure from '{current_agent}'. Aborting.")
        return END
    else :
        print(f"⛔ Protocol violation from '{current_agent}'. Agent must specify SUCCESS/RETRY/FALLBACK/FAILURE. Terminating.")
        return END