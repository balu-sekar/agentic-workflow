from typing import List, Dict, Any, TypedDict

class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    task: str
    plan: List[str]
    current_step: int
    results: List[Dict[str, Any]]
    done: bool
