"""Helpers for emitting LaTeX snippets from Python data structures."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence


def booktabs_table(
    rows: Sequence[Sequence[str]],
    *,
    header: Sequence[str],
    caption: str,
    label: str,
    col_fmt: str | None = None,
) -> str:
    """Return a LaTeX booktabs table as a string.

    Parameters
    ----------
    rows:     Data rows, each a sequence of cell strings.
    header:   Column header strings.
    caption:  \\caption{} text.
    label:    \\label{} key.
    col_fmt:  tabular column format string, e.g. 'lrr'. Auto-generated if None.
    """
    n_cols = len(header)
    if col_fmt is None:
        col_fmt = "l" + "r" * (n_cols - 1)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{h}}}" for h in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(c) for c in row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def newcommand(name: str, value: str) -> str:
    r"""Return a single \newcommand line, e.g. \newcommand{\scrNaiveHoldGapMm}{46}."""
    return rf"\newcommand{{\{name}}}{{{value}}}"


def write_newcommands(path: Path, commands: dict[str, str]) -> None:
    r"""Write a .tex file of \newcommand definitions, one per key/value pair."""
    lines = [newcommand(k, v) for k, v in commands.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(path: Path, **kwargs) -> None:
    """Write a booktabs_table() result to a .tex file."""
    path.write_text(booktabs_table(**kwargs), encoding="utf-8")
