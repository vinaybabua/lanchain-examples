"""Example 1: the same task with the raw SDK and with a LangChain chain.

Run:  python 01_simple_chain.py "What is RAG?"
"""

import sys

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common import MODEL, banner, get_model

question = sys.argv[1] if len(sys.argv) > 1 else "What is RAG?"
prompt_text = f"You are a friendly teacher. Explain simply in 3 sentences: {question}"

# --- Without LangChain: provider-specific client and response shape ---------
if MODEL.startswith("anthropic:"):
    import anthropic

    banner("Raw Anthropic SDK")
    client = anthropic.Anthropic()
    r = client.messages.create(
        model=MODEL.split(":", 1)[1],
        max_tokens=300,
        messages=[{"role": "user", "content": prompt_text}],
    )
    print(r.content[0].text)  # OpenAI would be r.choices[0].message.content

# --- With LangChain: one interface for every provider -----------------------
banner(f"LangChain chain  ({MODEL})")
prompt = ChatPromptTemplate.from_template(
    "You are a friendly teacher. Explain simply in 3 sentences: {question}"
)
llm = get_model()  # swap provider by changing MODEL in .env
chain = prompt | llm | StrOutputParser()

print(chain.invoke({"question": question}))

# Bonus: the same chain streams token by token with no extra code.
banner("Same chain, streamed")
for chunk in chain.stream({"question": "What is an AI agent?"}):
    print(chunk, end="", flush=True)
print()
