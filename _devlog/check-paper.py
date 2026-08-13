"""
Structural check for paper/*.tex, because there is no LaTeX toolchain here.

It does not typeset anything. It catches the four things that actually break a
first compile on someone else's machine: unbalanced braces, an odd number of
math delimiters, mismatched environments, and cross-references that point at
nothing -- an \input without a file, a \ref without a \label, a \cite without a
bib entry.

    python _devlog/check-paper.py

Exits non-zero on the first section that fails. Lives in the bitacora rather
than in the repo: it is a writing aid, not part of the benchmark.
"""
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"

# Verbatim-ish contexts where a stray $ or brace is not ours to balance.
COMMENT = re.compile(r"(?<!\\)%.*$")


def strip_comments(text):
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def check_braces(name, text, problems):
    depth = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        for match in re.finditer(r"(?<!\\)[{}]", line):
            depth += 1 if match.group() == "{" else -1
            if depth < 0:
                problems.append(f"{name}:{line_no}: closing brace with none open")
                return
    if depth:
        problems.append(f"{name}: {depth} brace(s) left open")


def check_math(name, text, problems):
    for line_no, line in enumerate(text.splitlines(), 1):
        if len(re.findall(r"(?<!\\)\$", line)) % 2:
            problems.append(f"{name}:{line_no}: odd number of $")


def check_environments(name, text, problems):
    stack = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for kind, env in re.findall(r"\\(begin|end)\{(\w+\*?)\}", line):
            if kind == "begin":
                stack.append((env, line_no))
            elif not stack:
                problems.append(f"{name}:{line_no}: \\end{{{env}}} with no \\begin")
            elif stack[-1][0] != env:
                open_env, open_line = stack.pop()
                problems.append(f"{name}:{line_no}: \\end{{{env}}} closes "
                                f"\\begin{{{open_env}}} from line {open_line}")
            else:
                stack.pop()
    for env, line_no in stack:
        problems.append(f"{name}:{line_no}: \\begin{{{env}}} never closed")


def main():
    # Includes tables/: they are generated, but a generator that emits an
    # unbalanced \textbf{ breaks the compile just as thoroughly as a typo.
    sections = sorted(PAPER.glob("**/*.tex"))
    if not sections:
        raise SystemExit(f"no .tex under {PAPER}")

    # Keyed by path relative to paper/, not by name: there is more than one
    # main.tex once a shortened version lives in a subdirectory, and keying by
    # name silently drops one of them.
    bodies = {p.relative_to(PAPER).as_posix():
              strip_comments(p.read_text(encoding="utf-8"))
              for p in sections}
    everything = "\n".join(bodies.values())

    labels = set(re.findall(r"\\label\{([^}]*)\}", everything))
    bib = PAPER / "refs.bib"
    keys = set(re.findall(r"@\w+\{([^,]+),",
                          bib.read_text(encoding="utf-8"))) if bib.exists() else set()

    problems = []
    for name, text in bodies.items():
        check_braces(name, text, problems)
        check_math(name, text, problems)
        check_environments(name, text, problems)

        # Paths resolve against the directory of the file that writes them,
        # the way LaTeX resolves them -- a document in paper/workshop/ says
        # ../tables/tab_axis and is right to.
        here = (PAPER / name).parent

        # A path may be written through a macro, because Overleaf compiles from
        # the project root while a local run compiles from the file's own
        # directory, and one source has to satisfy both. Try every expansion
        # the document defines for it and accept the path if any of them lands.
        macros = dict(re.findall(r"\\def\\(\w+)\{([^}]*)\}", text))

        def expansions(target):
            found = re.match(r"\\(\w+)\s*(.*)", target)
            if not found or found.group(1) not in macros:
                return [target]
            rest = found.group(2)
            return [macros[found.group(1)] + rest, rest, "../" + rest]

        for target in re.findall(r"\\input\{([^}]*)\}", text):
            if not any((here / f"{option}.tex").exists()
                       or (PAPER / f"{option}.tex").exists()
                       for option in expansions(target)):
                problems.append(f"{name}: \\input{{{target}}} has no file")
        # \includegraphics is normally written without an extension so LaTeX
        # can pick pdf over png; accept the file under any of them. Search the
        # file's own directory and every \graphicspath it declares.
        roots = [here] + [here / p for group in
                          re.findall(r"\\graphicspath\{(.*?)\}\s*$",
                                     text, re.MULTILINE)
                          for p in re.findall(r"\{([^}]*)\}", group)]
        for target in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}",
                                 text):
            stems = [root / target for root in roots]
            if not any(stem.exists() or any(stem.with_suffix(s).exists()
                                            for s in (".pdf", ".png", ".jpg"))
                       for stem in stems):
                problems.append(f"{name}: \\includegraphics{{{target}}} "
                                "has no file")
        for ref in re.findall(r"\\(?:eq)?ref\{([^}]*)\}", text):
            if ref not in labels:
                problems.append(f"{name}: \\ref{{{ref}}} has no \\label")
        for group in re.findall(r"\\cite[tp]?\{([^}]*)\}", text):
            for key in (k.strip() for k in group.split(",")):
                if key not in keys:
                    problems.append(f"{name}: \\cite{{{key}}} is not in refs.bib")

    for problem in problems:
        print(problem)
    print(f"\n{len(sections)} section(s), {len(labels)} label(s), "
          f"{len(keys)} bib entr(ies): "
          + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
