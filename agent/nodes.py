import json
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
load_dotenv()


from agent.state import AgentState

llm = AzureChatOpenAI(
            openai_api_version=os.environ.get("AZURE_OPENAI_VERSION"),
            azure_deployment=os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_KEY"),
            model="gpt-4"
        )


def planner(state: AgentState) -> AgentState:
	"""Create a plan based on the task"""
	messages = state["messages"]
	task = state["task"]

	planner_prompt = f"""
    You are a planning agent. Given a task, create a step-by-step plan to accomplish it.

    TASK: {task}

    Respond with a JSON array of steps. Each step should be a string describing one action.
    Maximum of 3 steps.
    """

	response = llm.invoke([HumanMessage(content=planner_prompt)])
	plan_text = response.content

	# Extract the JSON from the response
	try:
		# Find JSON array in the response if it's not just pure JSON
		start_idx = plan_text.find('[')
		end_idx = plan_text.rfind(']') + 1
		if start_idx != -1 and end_idx != 0:
			plan_text = plan_text[start_idx:end_idx]

		plan = json.loads(plan_text)
		print("Plan:", plan)
		print("=====================================")
	except:
		# Fallback if JSON parsing fails
		plan = ["Analyze the task", "Execute the task", "Verify results"]

	# Update the state with the plan
	new_messages = messages + [
		{"role": "system", "content": "Planning phase completed"},
		{"role": "assistant", "content": f"Created plan: {plan}"}
	]

	return {
		**state,
		"messages": new_messages,
		"plan": plan,
		"current_step": 0
	}


def executor(state: AgentState) -> AgentState:
	"""Execute the current step in the plan"""
	messages = state["messages"]
	plan = state["plan"]
	current_step = state["current_step"]
	results = state["results"]

	if current_step >= len(plan):
		return {**state, "done": True}

	current_task = plan[current_step]

	executor_prompt = f"""
    You are an execution agent. Execute the following step and provide the result.

    STEP: {current_task}

    Respond with a detailed explanation of how you executed this step and what the outcome was.
    """

	response = llm.invoke([HumanMessage(content=executor_prompt)])
	execution_result = response.content

	# Update the state with the execution result
	new_messages = messages + [
		{"role": "system", "content": f"Executing step {current_step + 1}: {current_task}"},
		{"role": "assistant", "content": execution_result}
	]

	new_results = results + [{"step": current_task, "result": execution_result}]

	print("executor", new_results)
	print("=====================================")

	return {
		**state,
		"messages": new_messages,
		"current_step": current_step + 1,
		"results": new_results
	}


def reviewer(state: AgentState) -> AgentState:
	"""Review the results of execution"""
	messages = state["messages"]
	results = state["results"]
	plan = state["plan"]
	current_step = state["current_step"]

	if current_step < len(plan):
		# Not all steps have been executed yet
		return state

	review_prompt = f"""
    You are a review agent. Review the following execution results and provide a summary.

    RESULTS:
    {json.dumps(results, indent=2)}

    Respond with a concise summary of what was accomplished and if the task was completed successfully.
    """

	response = llm.invoke([HumanMessage(content=review_prompt)])
	review_result = response.content

	# Update the state with the review
	new_messages = messages + [
		{"role": "system", "content": "Review phase"},
		{"role": "assistant", "content": review_result}
	]

	print("reviewer", review_result)

	return {
		**state,
		"messages": new_messages,
		"done": True
	}


# Define the edge conditions
def should_continue(state: AgentState) -> str:
	"""Determine the next node based on the current state"""
	if state["done"]:
		return "output"

	current_step = state["current_step"]
	plan = state["plan"]

	if current_step >= len(plan):
		return "reviewer"
	else:
		return "executor"
