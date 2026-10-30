# LangChain & LangGraph examples

Six runnable demos that match the "LangChain & LangGraph" deck (Examples 1–6).
Tested with langchain 1.4, langchain-core 1.7, langgraph 1.2 on Python 3.13.

## Setup (once)

Requires Python 3.10 or newer.

```bash
cd langchain-examples
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` and paste your `ANTHROPIC_API_KEY`. To use another provider, change
the `MODEL` line, for example `MODEL=openai:gpt-4o` with `OPENAI_API_KEY` set.

## Run the demos

| # | Slide | Command | What to point out |
|---|-------|---------|-------------------|
| 1 | Same task, two ways | `python 01_simple_chain.py "What is RAG?"` | Raw SDK vs `prompt \| llm \| parser`; streaming for free |
| 2 | RAG over a document | `python 02_rag.py "How many vacation days do new employees get?"` | The retrieved chunks print before the answer |
| 3 | create_agent | `python 03_support_agent.py "Where are orders 1042 and 2077?"` | The model chooses which tool to call, and how often |
| 4 | Middleware | `python 04_middleware.py` | The run pauses and asks you to approve the email |
| 5 | Graph by hand | `python 05_graph_by_hand.py` | Mermaid graph shows the loop; follow-up question remembers context |
| 6 | Pause & resume | see below | Stop the program between steps; nothing is lost |

**Example 6** needs no API key. Run it in three steps:

```bash
python 06_pause_resume.py start 240     # pauses for approval, saves state to refunds.sqlite
python 06_pause_resume.py status        # shows the saved, paused state
python 06_pause_resume.py resume yes    # or: resume no
```

Delete `refunds.sqlite` to reset it.

## Notes for presenting

- **Example 2** uses the sample `data/handbook.md`. Point it at your own file with `--file path/to/doc.pdf`.
  It uses OpenAI embeddings when `OPENAI_API_KEY` is set; otherwise it falls back to
  keyword (BM25) search, so it still runs with only an Anthropic key.
- **Example 3**: also try `"Cancel order 2077"` (works) and `"Cancel order 1042"` (already shipped, so the tool refuses).
- **Example 4**: answer `n` to show the agent handling a rejected action.
- **Example 5**: paste the printed Mermaid text into https://mermaid.live to show the graph.
- Tools use fake in-memory data (orders, weather, email), so nothing real is sent.
- `langchain-community` prints a deprecation warning on import; it is harmless.
