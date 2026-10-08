"""Ask a question from the terminal.

python -m bilingual_rag "Can I change my flight date?"
python -m bilingual_rag "كم يستغرق استرداد المبلغ؟" --search-only
"""

from __future__ import annotations

import argparse

from .answer import render_markdown
from .assistant import RETRIEVERS, AssistantConfig, build_assistant
from .llm import LLMClient, has_llm_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--search-only", action="store_true", help="show retrieved passages, no model call")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    parser.add_argument("--model", choices=["main", "cheap"], default="main")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    use_llm = not args.search_only and has_llm_config()
    llm = LLMClient.from_env(args.model) if use_llm else None
    assistant = build_assistant(AssistantConfig(retriever=args.retriever, k=args.k), llm=llm)

    if llm is None:
        if not args.search_only:
            print("(No OPENROUTER_API_KEY / model set: showing search results only.)\n")
        for hit in assistant.search(args.question):
            snippet = hit.chunk.text[:200].replace("\n", " ")
            print(f"{hit.rank}. [{hit.score:.2f}] {hit.chunk.label}\n   {snippet}\n")
        return
    print(render_markdown(assistant.ask(args.question)))


if __name__ == "__main__":
    main()
