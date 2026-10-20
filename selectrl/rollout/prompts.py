"""Policy prompt (locked) and chat-template rendering to token ids."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List

SOLVE_PROMPT_V1 = (
    "Solve the following competitive programming problem in Python 3.\n\n"
    "[PROBLEM]\n{question}\n[END OF PROBLEM]\n\n"
    "Your program must read the input from standard input and write the answer to standard output. It will be run on "
    "hidden tests with a time limit, so make it efficient. Give the complete program in a single ```python code block."
)


def solve_messages(question: str) -> List[Dict[str, str]]:
    return [{"role": "user", "content": SOLVE_PROMPT_V1.format(question=question.strip())}]


class ChatRenderer:
    """Renders chat messages with the model's own chat template (thinking switch included) into token ids."""

    def __init__(self, model_path: str, enable_thinking: bool = False):
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.enable_thinking = enable_thinking

    def text(self, messages: List[Dict[str, str]]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=self.enable_thinking)

    def __call__(self, messages: List[Dict[str, str]]) -> List[int]:
        return list(self.tok(self.text(messages), add_special_tokens=False)["input_ids"])

    @lru_cache(maxsize=None)
    def solve_prompt_ids(self, question: str) -> tuple:
        return tuple(self(solve_messages(question)))

    @property
    def pad_id(self) -> int:
        pid = self.tok.pad_token_id
        return int(pid if pid is not None else self.tok.eos_token_id)
