"""Example 3: a support agent built with create_agent (LangChain 1.0).

Run:  python 03_support_agent.py "Where is order 1042?"
Try:  "Where are orders 1042 and 2077?"   (the agent calls the tool twice)
      "Cancel order 2077"                 (uses a second tool)
"""

import sys

from langchain.agents import create_agent
from langchain.tools import tool

from common import banner, get_model

# A fake database so the demo runs anywhere.
ORDERS = {
    "1042": {"status": "shipped", "carrier": "UPS", "eta": "Nov 3"},
    "2077": {"status": "processing", "carrier": None, "eta": "Nov 6"},
    "3150": {"status": "delivered", "carrier": "FedEx", "eta": "Oct 28"},
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up an order's shipping status by its order ID."""
    order = ORDERS.get(order_id)
    print(f"   [tool] get_order_status({order_id!r})")
    return str(order) if order else f"No order found with ID {order_id}."


@tool
def cancel_order(order_id: str) -> str:
    """Cancel an order. Only orders that have not shipped can be cancelled."""
    print(f"   [tool] cancel_order({order_id!r})")
    order = ORDERS.get(order_id)
    if not order:
        return f"No order found with ID {order_id}."
    if order["status"] != "processing":
        return f"Order {order_id} is already {order['status']} and can't be cancelled."
    order["status"] = "cancelled"
    return f"Order {order_id} cancelled."


agent = create_agent(
    model=get_model(),
    tools=[get_order_status, cancel_order],
    system_prompt="You are a concise, helpful customer support agent for an online store.",
)

question = sys.argv[1] if len(sys.argv) > 1 else "Where is order 1042?"
banner(f"User: {question}")
result = agent.invoke({"messages": [{"role": "user", "content": question}]})

banner("Agent steps")
for msg in result["messages"]:
    if msg.type == "ai" and msg.tool_calls:
        for call in msg.tool_calls:
            print(f"AI decided to call {call['name']}({call['args']})")
    elif msg.type == "tool":
        print(f"Tool returned: {msg.content}")

banner("Final answer")
print(result["messages"][-1].text)
