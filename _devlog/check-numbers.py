"""
Every number written in the paper's prose, and whether a generated table backs it.

The rule this project runs on is that no figure in the paper is typed by hand:
tables come from `export_tables.py`, and loose numbers in the text are quoted
from those tables. Nothing enforced the second half. This does, as far as a
text scan can: it lists each numeric literal in the prose and says whether the
same literal appears in some generated table.

    python _devlog/check-numbers.py            # only what needs checking
    python _devlog/check-numbers.py --all      # everything, with its verdict

Three verdicts:

  TABLE       the literal appears in a generated table. Safe as long as the
              tables were regenerated after the last run.
  PROTOCOL    a design or protocol constant (three families, five seeds, 5000
              updates). Checked against the protocol table, not the results.
  CHECK       neither. Either it comes from `summarize_results.py` output that
              no table carries -- effect sizes, p-values, seed listings, ratios
              computed in the text -- or it is stale. **These are the ones a
              numbers pass has to verify by hand.**

What it cannot do: notice a number that is wrong but happens to match a
different cell of a different table, or read a sentence's claim about a number.
It narrows the pass from "reread everything" to a list, and that is all.
"""
import argparse
import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent.parent / "paper"

COMMENT = re.compile(r"(?<!\\)%.*$")
# A number, optionally signed and decimal. Not part of a LaTeX control word
# (\section2) and not glued to a letter (v4, 64x64 handled by the boundary).
NUMBER = re.compile(r"(?<![\w\\.])-?\d+(?:\.\d+)?(?![\w.])")

# Design and protocol constants, which are checked against tab_protocol and
# tab_tasks rather than against a measurement. Listing them here is a claim
# that they are structural; if one of these ever becomes a measured quantity,
# take it out.
STRUCTURAL = {
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "15", "20", "25", "32",
    "45", "50", "64", "100", "150", "200", "225", "512", "1000", "5000",
    "10000", "0.5", "1.0", "1.1", "9.8", "9.81", "4.0", "7.0", "3.0",
}


def strip_comments(text):
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def numbers_in(text):
    return {m.group() for m in NUMBER.finditer(text)}


def normalise(token):
    """`-6.21` and `6.21` are the same measurement wearing a different sign
    convention between a table column and a sentence."""
    return token.lstrip("-").rstrip("0").rstrip(".") or "0"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="also list the numbers that are already backed")
    args = parser.parse_args()

    tables = PAPER / "tables"
    if not tables.is_dir():
        raise SystemExit(f"no generated tables under {tables}")
    backed = set()
    for path in sorted(tables.glob("*.tex")):
        backed |= {normalise(n) for n in numbers_in(strip_comments(
            path.read_text(encoding="utf-8")))}

    counts = {"TABLE": 0, "PROTOCOL": 0, "CHECK": 0}
    findings = []
    for path in sorted(PAPER.glob("*.tex")):
        text = strip_comments(path.read_text(encoding="utf-8"))
        for line_no, line in enumerate(text.splitlines(), 1):
            for token in sorted(numbers_in(line)):
                if normalise(token) in backed:
                    verdict = "TABLE"
                elif token.lstrip("-") in STRUCTURAL:
                    verdict = "PROTOCOL"
                else:
                    verdict = "CHECK"
                counts[verdict] += 1
                if verdict == "CHECK" or args.all:
                    findings.append((verdict, path.name, line_no, token,
                                     line.strip()[:88]))

    for verdict, name, line_no, token, context in findings:
        print(f"{verdict:9} {name}:{line_no}  {token}")
        print(f"          {context}")
    print(f"\n{counts['TABLE']} backed by a table, {counts['PROTOCOL']} "
          f"structural, {counts['CHECK']} to verify by hand.")
    if counts["CHECK"]:
        print("Verify the CHECK list against:  python experiments/"
              "summarize_results.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
