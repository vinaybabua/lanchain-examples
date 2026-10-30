"""Example 2: RAG - answer questions from your own document.

Run:  python 02_rag.py "How many vacation days do new employees get?"
      python 02_rag.py "..." --file path/to/your.pdf

Uses OpenAI embeddings if OPENAI_API_KEY is set; otherwise falls back to a
keyword (BM25) retriever so the demo still runs with only an Anthropic key.
"""

import argparse
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from common import banner, get_model

parser = argparse.ArgumentParser()
parser.add_argument("question", nargs="?", default="How many vacation days do new employees get?")
parser.add_argument("--file", default="data/handbook.md")
args = parser.parse_args()

# 1. Load
if args.file.lower().endswith(".pdf"):
    from langchain_community.document_loaders import PyPDFLoader

    docs = PyPDFLoader(args.file).load()
else:
    from langchain_community.document_loaders import TextLoader

    docs = TextLoader(args.file, encoding="utf-8").load()

# 2. Split into overlapping chunks
chunks = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
).split_documents(docs)
print(f"Loaded {len(docs)} document(s), split into {len(chunks)} chunks")

# 3. Embed & store (or keyword index as a fallback)
if os.getenv("OPENAI_API_KEY"):
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_openai import OpenAIEmbeddings

    store = InMemoryVectorStore(OpenAIEmbeddings(model="text-embedding-3-small"))
    store.add_documents(chunks)
    retriever = store.as_retriever(search_kwargs={"k": 3})
    print("Retriever: vector search (OpenAI embeddings)")
else:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(chunks, k=3)
    print("Retriever: BM25 keyword search (set OPENAI_API_KEY for embeddings)")

# 4. Retrieve, then answer
llm = get_model()


def answer(question: str) -> str:
    hits = retriever.invoke(question)
    banner("Retrieved chunks")
    for i, d in enumerate(hits, 1):
        print(f"[{i}] {d.page_content[:150].strip()}...\n")
    context = "\n\n".join(d.page_content for d in hits)
    reply = llm.invoke(
        "Answer using ONLY this context. If it isn't there, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return reply.text


result = answer(args.question)
banner(f"Q: {args.question}")
print(result)
