"""Shared setup for the examples: loads .env and builds the chat model."""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Any "provider:model" string that init_chat_model understands.
MODEL = os.getenv("MODEL", "anthropic:claude-sonnet-5-5")
SMALL_MODEL = os.getenv("SMALL_MODEL", "anthropic:claude-haiku-5-5")


def require_key():
    provider = MODEL.split(":", 1)[0]
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}.get(provider)
    if key and not os.getenv(key):
        sys.exit(f"Set {key} in your .env file first (see .env.example).")


def get_model(model: str = MODEL):
    """Return a chat model. Change MODEL in .env to switch providers."""
    from langchain.chat_models import init_chat_model

    require_key()
    return init_chat_model(model, temperature=0)


def banner(title: str):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60)
