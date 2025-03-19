from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import Workflow
from agent.state import AgentState

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React app's address
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task: str

@app.get("/")
def init():
    return {"Hello": "World"}


@app.post("/workflow")
def run_workflow(request: TaskRequest):
    # Initialize the workflow
    agent = Workflow()

    # Create initial state
    initial_state = AgentState(
        messages=[],
        task=request.task,
        plan=[],
        current_step=0,
        results=[],
        done=False
    )

    # Execute the workflow
    result = agent.workflow.invoke(initial_state)

    return result