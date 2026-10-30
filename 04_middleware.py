"""Example 4: middleware - add guardrails without rewriting the agent.

The agent can draft and send emails, but HumanInTheLoopMiddleware pauses
before send_email so you can approve or reject it in the terminal.

Run:  python 04_middleware.py
"""

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware, SummarizationMiddleware
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from common import SMALL_MODEL, banner, get_model


@tool
def search_docs(query: str) -> str:
    """Search the internal knowledge base."""
    print(f"   [tool] search_docs({query!r})")
    return "The Q4 offsite is on Dec 12 in Charlotte. RSVP by Nov 20."


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    print(f"   [tool] send_email(to={to!r}, subject={subject!r})")
    return f"Email sent to {to}."


agent = create_agent(
    model=get_model(),
    tools=[search_docs, send_email],
    system_prompt="You are an assistant. Look things up before writing emails.",
    middleware=[
        # Summarize older messages once the chat grows past 20 messages.
        SummarizationMiddleware(model=get_model(SMALL_MODEL), trigger=("messages", 20)),
        # Pause before any email goes out; other tools run freely.
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),
    ],
    checkpointer=InMemorySaver(),  # required so a paused run can resume
)

config = {"configurable": {"thread_id": "demo-1"}}
request = "Email sam@example.com the date and RSVP deadline for the Q4 offsite."
banner(f"User: {request}")
result = agent.invoke({"messages": [{"role": "user", "content": request}]}, config)

# The run pauses here if the model wants to call send_email.
while result.get("__interrupt__"):
    pending = result["__interrupt__"][0].value["action_requests"]
    decisions = []
    for action in pending:
        banner(f"APPROVAL NEEDED: {action['name']}")
        for k, v in action["args"].items():
            print(f"{k}: {v}")
        choice = input("\nApprove? [y/n]: ").strip().lower()
        if choice == "y":
            decisions.append({"type": "approve"})
        else:
            decisions.append({"type": "reject", "message": "The user rejected this email."})
    result = agent.invoke(Command(resume={"decisions": decisions}), config)

banner("Final answer")
print(result["messages"][-1].text)
