"""Example 6: durable execution - pause for a human, resume later.

State is saved in a SQLite file, so you can stop the program between steps
(or restart your laptop) and the run continues where it left off.
No API key needed.

Run it in three steps:
    python 06_pause_resume.py start 240      # starts, then pauses for approval
    python 06_pause_resume.py status         # shows the saved, paused state
    python 06_pause_resume.py resume yes     # or: resume no
"""

import sqlite3
import sys
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class RefundState(TypedDict, total=False):
    ticket: str
    amount: float
    approved: bool
    result: str


def prepare_refund(state: RefundState):
    print(f"   [node] prepare_refund: ticket {state['ticket']}, ${state['amount']}")
    return {}


def approve_refund(state: RefundState):
    print("   [node] approve_refund: waiting for a human...")
    decision = interrupt({"question": "Approve this refund?", "amount": state["amount"]})
    return {"approved": decision == "yes"}


def issue_refund(state: RefundState):
    print("   [node] issue_refund")
    return {"result": f"Refunded ${state['amount']}"}


def deny_refund(state: RefundState):
    print("   [node] deny_refund")
    return {"result": "Refund denied"}


g = StateGraph(RefundState)
g.add_node("prepare_refund", prepare_refund)
g.add_node("approve_refund", approve_refund)
g.add_node("issue_refund", issue_refund)
g.add_node("deny_refund", deny_refund)
g.add_edge(START, "prepare_refund")
g.add_edge("prepare_refund", "approve_refund")
g.add_conditional_edges(
    "approve_refund", lambda s: "issue_refund" if s["approved"] else "deny_refund"
)
g.add_edge("issue_refund", END)
g.add_edge("deny_refund", END)

conn = sqlite3.connect("refunds.sqlite", check_same_thread=False)
app = g.compile(checkpointer=SqliteSaver(conn))
config = {"configurable": {"thread_id": "ticket-881"}}

cmd = sys.argv[1] if len(sys.argv) > 1 else "start"

if cmd == "start":
    amount = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    result = app.invoke({"ticket": "ticket-881", "amount": amount}, config)
    if "__interrupt__" in result:
        print("\nPAUSED:", result["__interrupt__"][0].value)
        print("State saved to refunds.sqlite. Close this program, then run:")
        print("   python 06_pause_resume.py resume yes")
elif cmd == "status":
    snap = app.get_state(config)
    print("Saved values:", snap.values)
    print("Next node:   ", snap.next or "(finished)")
elif cmd == "resume":
    answer = sys.argv[2] if len(sys.argv) > 2 else "yes"
    result = app.invoke(Command(resume=answer), config)
    print("\nDONE:", result.get("result"))
else:
    print(__doc__)
