"""Run a one-off DocMind question from the command line."""

from __future__ import annotations

import argparse
import asyncio

from agent.graph import run_agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--provider", choices=["hf", "openai", "anthropic"])
    args = parser.parse_args()
    result = asyncio.run(run_agent(args.question, args.provider))
    print(result.answer)
    if result.tool_calls:
        print(f"Tools used: {', '.join(result.tool_calls)}")


if __name__ == "__main__":
    main()
