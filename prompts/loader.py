from __future__ import annotations

import os

import pystache

_here = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_here)
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")

_renderer = pystache.Renderer(
    string_encoding="utf-8",
    escape=lambda u: u if isinstance(u, str) else ("" if u is None else str(u)),
)


def load_prompt(name: str) -> str:
    path = os.path.join(PROMPTS_DIR, f"{name}.mustache")
    with open(path, encoding="utf-8") as f:
        return f.read().rstrip()


def render_prompt(name: str, **kwargs: str | int | float | bool | None) -> str:
    template = load_prompt(name)
    ctx = {k: ("" if v is None else str(v)) for k, v in kwargs.items()}
    return _renderer.render(template, ctx).rstrip()
