from __future__ import annotations

import contextlib
import io
import traceback
from dataclasses import dataclass
from typing import Any, Callable


_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "format": format,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "open": open,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "__import__": __import__,
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "RuntimeError": RuntimeError,
}


@dataclass
class ReplExecution:
    stdout: str
    stderr: str
    final_answer: str | None
    child_calls: list[dict[str, Any]]


class RecursiveLocalRepl:
    def __init__(
        self,
        *,
        context_payload: str,
        llm_query_fn: Callable[[str, str | None], dict[str, Any]],
        rlm_query_fn: Callable[[str, str | None, int | None], dict[str, Any]],
    ):
        self._llm_query_fn = llm_query_fn
        self._rlm_query_fn = rlm_query_fn
        self._child_calls: list[dict[str, Any]] = []
        self._last_final_answer: str | None = None

        self.globals: dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS,
            "FINAL": self._final,
            "FINAL_VAR": self._final_var,
            "SHOW_VARS": self._show_vars,
            "llm_query": self._llm_query,
            "rlm_query": self._rlm_query,
        }
        self.locals: dict[str, Any] = {"context": context_payload}

    def _final(self, value: Any) -> str:
        answer = str(value)
        self._last_final_answer = answer
        return answer

    def _final_var(self, variable_name: str | Any) -> str:
        if not isinstance(variable_name, str):
            return self._final(variable_name)
        key = variable_name.strip().strip("\"'")
        if key not in self.locals:
            available = sorted(name for name in self.locals if not name.startswith("_"))
            return f"Variable '{key}' not found. Available variables: {available}"
        return self._final(self.locals[key])

    def _show_vars(self) -> dict[str, str]:
        return {name: type(value).__name__ for name, value in self.locals.items() if not name.startswith("_")}

    def _record_call(self, payload: dict[str, Any]) -> str:
        self._child_calls.append(payload)
        return payload["response"]

    def _llm_query(self, prompt: str, model: str | None = None) -> str:
        return self._record_call(self._llm_query_fn(prompt, model))

    def _rlm_query(self, prompt: str, model: str | None = None, max_depth: int | None = None) -> str:
        return self._record_call(self._rlm_query_fn(prompt, model, max_depth))

    def execute(self, code: str) -> ReplExecution:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        self._child_calls = []
        self._last_final_answer = None

        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                exec(code, self.globals, self.locals)
            except Exception:
                traceback.print_exc(file=stderr_buffer)

        return ReplExecution(
            stdout=stdout_buffer.getvalue(),
            stderr=stderr_buffer.getvalue(),
            final_answer=self._last_final_answer,
            child_calls=list(self._child_calls),
        )