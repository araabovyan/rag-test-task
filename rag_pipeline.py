"""RAG pipeline: generates pandas code from questions, executes it, synthesizes answers."""

import builtins
import datetime
import re
import traceback
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from groq import Groq

from data_loader import load_dataframes, get_schema_description

# Builtins that are safe for generated code to use.
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "frozenset", "getattr", "hasattr", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "print", "range",
        "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
        "type", "zip",
        # Exceptions the LLM might raise/catch
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "ZeroDivisionError", "RuntimeError",
        "True", "False", "None",
    )
    if hasattr(builtins, name)
}

# Regex that matches standalone import lines (import ... / from ... import ...)
_IMPORT_RE = re.compile(r"^\s*(import |from \S+ import )", re.MULTILINE)

load_dotenv()

# --- Prompt templates ---

CODE_GEN_SYSTEM = """\
You are a data analyst. Given the schema below, write Python/pandas code that \
answers the user's question.

{schema}

The DataFrames `clients`, `invoices`, and `line_items` are already loaded. \
pandas is available as `pd`, numpy as `np`, and the `datetime` module is \
available as `datetime`. Do NOT import anything or read files.

Output ONLY valid Python code (no markdown, no explanations). Store the final \
answer in a variable called `result` (DataFrame, Series, scalar, or string).

Key rules:
- Use pd.Timestamp for date comparisons.
- Line total including tax = quantity * unit_price * (1 + tax_rate).
- Do NOT use print().
- If the question asks to "list" something, make result a DataFrame.

If conversation history is provided and the user refers to a previous answer \
("which of those", "from them", etc.), use the prior context to understand \
what they mean and write code accordingly.
"""

ANSWER_GEN_SYSTEM = """\
You are a business data analyst. A code-based retrieval step already ran \
that filtered and computed the relevant data for the user's question. \
The data below is the result -- trust that it already reflects any filters \
(date ranges, regions, etc.) implied by the question.

Answer using ONLY this data. Use markdown tables for tabular data. \
Never invent numbers. If the data is empty, say so.

If there's conversation history, use it to understand follow-up references.
"""


class RAGPipeline:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.dfs = load_dataframes()
        self.schema = get_schema_description(self.dfs)

    def _generate_code(self, question: str, history: list[dict] | None = None) -> str:
        system = CODE_GEN_SYSTEM.format(schema=self.schema)
        messages = [{"role": "system", "content": system}]

        for turn in history or []:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["code"]})
            preview = turn.get("data", "")
            if preview:
                # FIXME: 2000 chars is arbitrary, might cut off important context
                if len(preview) > 2000:
                    preview = preview[:2000] + "\n... (truncated)"
                messages.append({
                    "role": "user",
                    "content": f"[Code returned:\n```\n{preview}\n```]",
                })

        messages.append({"role": "user", "content": question})

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=2048,
        )
        code = resp.choices[0].message.content.strip()
        if code.startswith("```"):
            code = code.split("\n", 1)[1]
            code = code.rsplit("```", 1)[0]
        return code

    @staticmethod
    def _sanitize_code(code: str) -> str:
        """Remove import statements from generated code.

        All required libraries (pd, np, datetime, etc.) are already
        provided in the execution namespace, so imports are unnecessary
        and would fail in the sandboxed exec.
        """
        return _IMPORT_RE.sub("", code)

    def _execute_code(self, code: str) -> tuple[object, str | None]:
        """Run generated code in a restricted namespace. Returns (result, error)."""
        code = self._sanitize_code(code)
        ns = {
            "__builtins__": _SAFE_BUILTINS,
            "pd": pd, "np": np, "datetime": datetime,
            "clients": self.dfs["clients"].copy(),
            "invoices": self.dfs["invoices"].copy(),
            "line_items": self.dfs["line_items"].copy(),
        }
        try:
            # TODO: swap exec for something like RestrictedPython in production
            exec(code, ns)
            return ns.get("result", "No `result` variable was set."), None
        except Exception:
            return None, traceback.format_exc()

    def _generate_answer(self, question: str, data_str: str,
                         history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": ANSWER_GEN_SYSTEM}]

        for turn in history or []:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({
            "role": "user",
            "content": f"**Question:** {question}\n\n**Retrieved data:**\n```\n{data_str}\n```",
        })

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=2048,
        )
        return resp.choices[0].message.content.strip()

    def ask(self, question: str, history: list[dict] | None = None,
            max_retries: int = 2) -> dict:
        """Run the full pipeline: question -> code -> execute -> answer."""
        history = history or []
        last_error = None
        code = data_str = ""

        for _ in range(max_retries):
            if last_error is None:
                code = self._generate_code(question, history=history)
            else:
                fix_prompt = (
                    f"The previous code raised an error:\n{last_error}\n\n"
                    f"Original question: {question}\n\nPlease fix the code."
                )
                code = self._generate_code(fix_prompt, history=history)

            result, error = self._execute_code(code)
            if error is not None:
                last_error = error
                continue

            if isinstance(result, pd.DataFrame):
                data_str = result.to_string(index=False)
            elif isinstance(result, pd.Series):
                data_str = result.to_string()
            else:
                data_str = str(result)

            answer = self._generate_answer(question, data_str, history=history)
            return {"answer": answer, "code": code, "data": data_str, "error": None}

        return {
            "answer": "Sorry, I couldn't retrieve the data. Try rephrasing your question.",
            "code": code, "data": "", "error": last_error,
        }
