"""Example 5: the agent loop written by hand as a LangGraph graph.

Run:  python 05_graph_by_hand.py "What's the weather in Charlotte and Austin?"

It also prints the graph as Mermaid text - paste it into https://mermaid.live
to show the loop visually.
"""

import sys

from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from common import banner, get_model


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    print(f"   [tool] get_weather({city!r})")
    fake = {"charlotte": "72°F and sunny", "austin": "85°F and humid"}
    return fake.get(city.lower(), "60°F and cloudy")


tools = [get_weather]
llm = get_model().bind_tools(tools)


def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}


g = StateGraph(MessagesState)
g.add_node("agent", agent)
g.add_node("tools", ToolNode(tools))
g.add_edge(START, "agent")
g.add_conditional_edges("agent", tools_condition)  # tool call? -> tools, else END
g.add_edge("tools", "agent")  # the loop
app = g.compile(checkpointer=InMemorySaver())

banner("Graph (Mermaid)")
print(app.get_graph().draw_mermaid())

question = sys.argv[1] if len(sys.argv) > 1 else "What's the weather in Charlotte and Austin?"
config = {"configurable": {"thread_id": "weather-1"}}

banner(f"User: {question}  (streaming each step)")
for step in app.stream({"messages": [("user", question)]}, config, stream_mode="updates"):
    for node, update in step.items():
        last = update["messages"][-1]
        if getattr(last, "tool_calls", None):
            print(f"-> {node}: wants {[c['name'] for c in last.tool_calls]}")
        else:
            print(f"-> {node}: {last.text[:200]}")

# Same thread_id = the graph remembers the conversation (checkpointer).
banner("Follow-up on the same thread")
out = app.invoke({"messages": [("user", "Which of those two cities is warmer?")]}, config)
print(out["messages"][-1].text)
