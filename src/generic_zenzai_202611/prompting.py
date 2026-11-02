"""Prompt format and length limits shared by training and evaluation."""

MAX_TOKENS = 512
CONTEXT_CHARS = 40
PROMPT_TEMPLATE = "左文脈：{context}\n読み：{reading}\n変換：\n"


def make_prompt(reading, left_context):
    context = "" if left_context is None else left_context[-CONTEXT_CHARS:]
    return PROMPT_TEMPLATE.format(context=context, reading=reading)
