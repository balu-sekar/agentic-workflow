from agent.nodes import planner, executor, reviewer, should_continue
from agent.state import AgentState
from langgraph.graph import StateGraph, END


class Workflow():
    def __init__(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("reviewer", reviewer)
        workflow.add_node("output", lambda x: x)  # Pass-through node for output

        workflow.add_edge("planner", "executor")
        workflow.add_conditional_edges(
            "executor",
            should_continue,
            {
                "reviewer": "reviewer",
                "executor": "executor",
                "output": "output"
            }
        )

        workflow.add_edge("reviewer", "output")
        workflow.add_edge("output", END)

        workflow.set_entry_point("planner")

        self.workflow = workflow.compile()



