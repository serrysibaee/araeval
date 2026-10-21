"""Small helpers shared by the loader and the preprocessors (kept separate to avoid import cycles)."""
import ast
import json
from typing import Any


def parse_value(v: Any) -> Any:
    """Decode list/dict values that the Hub stores as strings."""
    if isinstance(v, str) and v[:1] in "[{":
        for loader in (json.loads, ast.literal_eval):
            try:
                return loader(v)
            except Exception:
                pass
    return v
