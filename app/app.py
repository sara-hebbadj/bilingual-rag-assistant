"""Gradio demo: ask in Arabic or English, get an answer with its sources.

    python app/app.py        # then open http://127.0.0.1:7860

Without an API key the demo still works in "Search only" mode: it shows the
passages BM25 retrieves, which is useful for checking retrieval by eye.
On a Hugging Face Space: add OPENROUTER_API_KEY as a secret and MODEL_MAIN as a variable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

# Lets `python app/app.py` work even if the package was not pip-installed (e.g. on a Space).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bilingual_rag.answer import render_markdown  # noqa: E402
from bilingual_rag.assistant import AssistantConfig, build_assistant  # noqa: E402
from bilingual_rag.ingest import REPO_ROOT  # noqa: E402
from bilingual_rag.llm import LLMClient, has_llm_config  # noqa: E402

ANSWER_MODE = "Answer with sources (LLM)"
SEARCH_MODE = "Search only (no LLM)"

EXAMPLES = [
    "How long does a refund take after I cancel?",
    "Can I give my ticket to someone else?",
    "كم رسوم تعديل موعد الرحلة؟",
    "حقيبتي لم تصل معي إلى المطار، ماذا أفعل؟",
    "Can I pay for an Umrah package in instalments?",  # answer only exists in Arabic
    "Can I bring my cat on the plane?",  # not covered: should say "I don't know"
]

INTRO = """### Sarab Travel help assistant (Arabic / English)
Answers come **only** from the help pages, with a citation for each fact, and the assistant says
"I don't know" when the pages do not cover a question.
_Sarab Travel is a **fictional** agency: every policy here is made up for this demo._"""

DEMO_MODE_NOTE = (
    "**Demo mode — live AI is off; add OPENROUTER_API_KEY in Space settings to enable** (plus a "
    "`MODEL_MAIN` variable). Until then only **Search only** mode works: it shows the help-page "
    "passages the BM25 search finds, without a generated answer."
)


def render_search(hits) -> str:
    if not hits:
        return "No matching passages. / لا توجد مقاطع مطابقة."
    lines = ["**Top passages (BM25 search, no LLM)**", ""]
    for hit in hits:
        snippet = hit.chunk.text.replace("\n", " ")
        lines.append(f"{hit.rank}. **{hit.chunk.label}** · score {hit.score:.2f}  \n{snippet[:300]}")
    return "\n\n".join(lines)


def make_responder(assistant):
    """Return the chat handler: (message, history, mode) -> (new history, cleared textbox)."""

    def respond(message: str, history: list, mode: str):
        if not message.strip():
            return history, ""
        if mode == ANSWER_MODE and assistant.llm is not None:
            reply = render_markdown(assistant.ask(message))
        else:
            reply = render_search(assistant.search(message))
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
        return history, ""

    return respond


def build_demo(llm=None) -> gr.Blocks:
    if llm is None and has_llm_config():
        role = "main" if os.getenv("MODEL_MAIN") else "cheap"  # has_llm_config() accepts either model
        llm = LLMClient.from_env(role, trace_path=REPO_ROOT / "evals" / "app_traces.jsonl")
    assistant = build_assistant(AssistantConfig(), llm=llm)
    modes = [ANSWER_MODE, SEARCH_MODE] if llm else [SEARCH_MODE]
    respond = make_responder(assistant)

    with gr.Blocks(title="Bilingual RAG assistant") as demo:
        gr.Markdown(INTRO)
        if llm is None:
            gr.Markdown(DEMO_MODE_NOTE)
        mode = gr.Radio(modes, value=modes[0], label="Mode")
        chatbot = gr.Chatbot(height=460, label="Conversation")
        textbox = gr.Textbox(placeholder="Ask a question / اكتب سؤالك", label="Question", lines=1)
        gr.Examples(EXAMPLES, inputs=textbox)
        textbox.submit(respond, [textbox, chatbot, mode], [chatbot, textbox])
        gr.Button("Send / إرسال", variant="primary").click(respond, [textbox, chatbot, mode], [chatbot, textbox])
    return demo


if __name__ == "__main__":
    build_demo().launch()
