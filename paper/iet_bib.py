"""Write refs_iet.tex: the cited entries of refs.bib in IET reference style.

AIAM (IET Conference Proceedings) asks for the referencing style of its sample
article: numbered in order of first citation, square brackets, and entries such as
  Smith, T., Jones, M.: 'Title', Journal, 2007, 1, (2), pp. 1-7
  Jones, L., Brown, D.: 'Title'. Proc. Int. Conf. X, 2006, pp. 1-7
with the first three authors followed by 'et al.' when there are more than three.
No IET .bst is published for conference papers, so this script formats the entries.

Usage: python iet_bib.py   (run from paper/; rerun whenever refs.bib or \\cite keys change)
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_bib(text: str) -> dict:
    entries = {}
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        kind, key = m.group(1).lower(), m.group(2)
        i, depth = m.end(), 1
        while depth and i < len(text):
            depth += {"{": 1, "}": -1}.get(text[i], 0)
            i += 1
        body = text[m.end():i - 1]
        fields = {}
        for f in re.finditer(r"(\w+)\s*=\s*\{", body):
            j, d = f.end(), 1
            while d and j < len(body):
                d += {"{": 1, "}": -1}.get(body[j], 0)
                j += 1
            fields[f.group(1).lower()] = " ".join(body[f.end():j - 1].split())
        entries[key] = dict(kind=kind, **fields)
    return entries


def cite_order(tex: str) -> list:
    tex = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())
    order = []
    for m in re.finditer(r"\\cite[pt]?\*?(?:\[[^\]]*\])?\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in order:
                order.append(k)
    return order


def initials(given: str) -> str:
    out = []
    for part in given.split():
        sub = [s for s in part.split("-") if s]
        out.append("-".join(s.lstrip("{\\'\"")[0] + "." for s in sub))
    return "".join(out)


def one_author(name: str) -> str:
    name = name.strip()
    if "," in name:
        last, given = [s.strip() for s in name.split(",", 1)]
    else:
        parts = name.split()
        last, given = parts[-1], " ".join(parts[:-1])
    return f"{last}, {initials(given)}" if given else last


def authors(field: str) -> str:
    names = [one_author(a) for a in re.split(r"\s+and\s+", field)]
    if len(names) > 3:
        return ", ".join(names[:3]) + ", et al."
    return ", ".join(names)


def strip_braces(s: str) -> str:
    # drop protective braces {PIE} -> PIE, keep accent groups like {\"a}
    return re.sub(r"\{([^{}\\]*)\}", r"\1", s)


def fmt(e: dict) -> str:
    who = authors(e["author"])
    title = strip_braces(e["title"])
    pages = e.get("pages", "").replace("--", "-")
    year = e.get("year", "")
    if e["kind"] == "article":
        parts = [e["journal"], year]
        if e.get("volume"):
            parts.append(e["volume"])
        if e.get("number"):
            parts.append(f"({e['number']})")
        if pages:
            parts.append(f"pp. {pages}")
        return f"{who}: `{title}', {', '.join(parts)}"
    venue = e.get("booktitle", "")
    tail = ", ".join(p for p in [venue, year, f"pp. {pages}" if pages else ""] if p)
    return f"{who}: `{title}'. {tail}"


def main() -> None:
    entries = parse_bib((HERE / "refs.bib").read_text(encoding="utf-8"))
    order = cite_order((HERE / "main.tex").read_text(encoding="utf-8"))
    missing = [k for k in order if k not in entries]
    if missing:
        raise SystemExit(f"cited but not in refs.bib: {missing}")
    lines = ["\\begin{thebibliography}{99}"]
    lines += [f"\\bibitem{{{k}}} {fmt(entries[k])}" for k in order]
    lines.append("\\end{thebibliography}")
    (HERE / "refs_iet.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"refs_iet.tex: {len(order)} cited entries; uncited in refs.bib: "
          f"{sorted(set(entries) - set(order))}")


if __name__ == "__main__":
    main()
