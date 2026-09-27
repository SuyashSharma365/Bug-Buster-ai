"""Configurable local Hugging Face and API-backed chat models."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


def _local_huggingface_llm() -> BaseChatModel:
    """Create a local chat model backed by a Transformers text-generation pipeline."""
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    model_name = os.getenv("DOCMIND_HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=int(os.getenv("DOCMIND_MAX_NEW_TOKENS", "512")),
        do_sample=False,
        return_full_text=False,
    )
    return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=generator))


def get_llm(provider: str = "groq") -> BaseChatModel:
    """Return the configured chat model for Groq, HF local, or Anthropic."""
    selected = provider.strip().lower()
    if selected in {"hf", "huggingface", "local"}:
        return _local_huggingface_llm()
    if selected == "groq":
        from langchain_groq import ChatGroq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required when DOCMIND_LLM_PROVIDER=groq")
        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            api_key=api_key,
            temperature=0,
        )
    if selected == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required when DOCMIND_LLM_PROVIDER=anthropic")
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            api_key=api_key,
            temperature=0,
        )
    raise ValueError("Unsupported LLM provider. Choose 'groq', 'hf', or 'anthropic'.")
