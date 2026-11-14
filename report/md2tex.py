"""Convert report/draft.md to report/report.tex (A4, single column, Arial 10 pt): cover page,
contents page, then the body numbered from page 1 (abstract first), references and appendix.

    python report/md2tex.py
    cd report && xelatex report.tex && xelatex report.tex     # twice, for the contents page

draft.md is the source: edit it and convert again, do not edit report.tex. The "Draft notes" block
quote at the top of draft.md is left out. The log prints REFSTART with the page on which the
references begin; the body is within the 10-page limit while that is 11 or less."""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
src = (HERE / "draft.md").read_text(encoding="utf-8")
figdir, margin, dst = "figures", "2.54cm", HERE / "report.tex"
BS, NL = chr(92), chr(10)
SUB = {"⁻⁴": r"$^{-4}$", "⁻⁵": r"$^{-5}$", "→": r"$\rightarrow$", "≥": r"$\geq$", "≤": r"$\leq$",
       "−": r"$-$", "×": r"$\times$", "±": r"$\pm$", "λ": r"$\lambda$", "…": r"\ldots{}"}
FIG_WIDTH = {"fig1": "1.0", "fig2": "0.92", "fig4": "1.0", "fig3": "0.96", "fig5": "0.96", "fig6": "0.96", "fig7": "0.96", "fig8": "0.96"}


def esc_code(c: str) -> str:
    c = c.replace(BS, "/")
    c = re.sub(r"([%&#_${}])", lambda m: BS + m.group(1), c)
    return r"{\small\texttt{" + c.replace("/", "/" + BS + "allowbreak{}") + "}}"


def inline(t: str) -> str:
    keep = []

    def stash(kind):
        def f(m):
            keep.append((kind, m.group(1)))
            return f"@@K{len(keep) - 1}@@"
        return f
    t = re.sub(r"`([^`]*)`", stash("code"), t)
    t = re.sub(r"(https?://[^\s)]*[^\s).,;])", stash("url"), t)
    t = re.sub(r"\$([^$]+)\$", stash("math"), t)
    t = re.sub(r"([%&#_])", lambda m: BS + m.group(1), t)
    t = t.replace("~", r"\textasciitilde{}").replace("^", r"\^{}")
    t = re.sub(r'"([^"]*)"', lambda m: "``" + m.group(1) + "''", t)
    t = re.sub(r"\*\*(.+?)\*\*", lambda m: r"\textbf{" + m.group(1) + "}", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*", lambda m: r"\textit{" + m.group(1) + "}", t)
    t = re.sub(r"osu! (?=[a-z(])", lambda m: "osu!" + BS + " ", t)
    for k, v in SUB.items():
        t = t.replace(k, v)

    def back(m):
        kind, body = keep[int(m.group(1))]
        if kind == "url":
            return BS + "url{" + body + "}"
        return esc_code(body) if kind == "code" else "$" + body + "$"
    return re.sub(r"@@K(\d+)@@", back, t)


lines = src.splitlines()
title = next(l[2:] for l in lines if l.startswith("# "))
head_lines = []                                   # the lines between the title and the first rule
i = next(k for k, l in enumerate(lines) if l.startswith("# ")) + 1
while not lines[i].startswith(">") and lines[i].strip() != "---":
    if lines[i].strip():
        head_lines.append(lines[i].strip())
    i += 1

out = []
START = re.compile(r"(#|\||- |\d+\. |```|!\[|>|---|\$\$)")
section = ""
while i < len(lines):
    l = lines[i]
    if l.startswith(">") or l.strip() == "---":
        i += 1
        continue
    if l.startswith("```"):
        j, body = i + 1, []
        while not lines[j].startswith("```"):
            body.append(lines[j])
            j += 1
        out.append(r"{\footnotesize\begin{verbatim}" + NL + NL.join(body) + NL + r"\end{verbatim}}")
        i = j + 1
        continue
    if l.startswith("$$"):
        body = l.strip().strip("$")
        i += 1
        out.append(r"\[" + body + r"\]")
        continue
    m = re.match(r"(#+) (.*)", l)
    if m:
        lvl, text = len(m.group(1)), m.group(2)
        if text == "Abstract":
            section, abstract_at = "abstract", len(out)
            out.append(r"\section*{Abstract}")
        elif text.startswith("1. "):
            section = "body"
            abstract = out[abstract_at:]          # the abstract opens the body, so it counts in the 10 pages
            del out[abstract_at:]
            out.append(r"\newpage\tableofcontents\newpage\pagenumbering{arabic}\setcounter{page}{1}")
            out += abstract
            out.append(r"\section{" + inline(text[3:]) + "}")
        elif text in ("References", "Appendix"):
            section = text
            out.append(r"\newpage\typeout{" + ("REFSTART" if text == "References" else "APPSTART") + r" \thepage}")
            out.append(r"\section*{" + text + r"}\addcontentsline{toc}{section}{" + text + "}")
        elif lvl == 2:
            out.append(r"\section{" + inline(re.sub(r"^\d+\. ", "", text)) + "}")
        elif re.match(r"\d+\.\d+ ", text):
            out.append(r"\subsection{" + inline(re.sub(r"^\d+\.\d+ ", "", text)) + "}")
        else:
            out.append(r"\subsection*{" + inline(text) + r"}\addcontentsline{toc}{subsection}{" + inline(text) + "}")
        i += 1
        continue
    m = re.match(r"!\[.*?\]\((.*?)\)", l)
    if m:
        name = pathlib.Path(m.group(1)).name
        w = FIG_WIDTH.get(name.split("_")[0], "1.0")
        out.append("@@FIG@@" + r"\includegraphics[width=" + w + r"\linewidth]{" + figdir + "/" + name + "}")
        i += 1
        continue
    if l.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
            i += 1
        rows = [r for r in rows if not all(re.fullmatch(r":?-+:?", c) for c in r)]
        n = len(rows[0])
        longest = [max(len(r[k]) for r in rows) for k in range(n)]
        end = " " + BS + BS + NL
        body = end.join(" & ".join(inline(c) for c in r) for r in rows[1:])
        headrow = " & ".join(inline(c) for c in rows[0])
        if sum(longest) + 3 * n < 92:            # a small table: its natural width, centred
            env, spec, close = r"\begin{tabular}{", "l" * n, r"\end{tabular}"
        else:
            # every column wraps; widths in proportion to the longest cell (capped), so short
            # columns stay narrow and the table always fits the text width
            # sized by the body cells: a long header wraps, a number such as "96.1 ± 0.9" does not
            w = [min(max(max(len(r[k]) for r in rows[1:]), 7), 36 if n > 4 else 46) + 3 for k in range(n)]
            cols = [r">{\hsize=" + f"{n * x / sum(w):.3f}" + r"\hsize\raggedright\arraybackslash}X" for x in w]
            env, spec, close = r"\begin{tabularx}{\linewidth}{", "".join(cols), r"\end{tabularx}"
        out.append("@@TAB@@" + r"\small\setlength{\tabcolsep}{5pt}\renewcommand{\arraystretch}{1.12}" + env + spec
                   + r"}\toprule" + NL + headrow + " " + BS + BS + r"\midrule" + NL + body + " " + BS + BS
                   + r"\bottomrule" + close)
        continue
    if re.match(r"(- |\d+\. )", l):
        env = "itemize" if l.startswith("- ") else "enumerate"
        items = []
        while i < len(lines) and (re.match(r"(- |\d+\. )", lines[i]) or (lines[i].startswith("  ") and lines[i].strip())):
            if re.match(r"(- |\d+\. )", lines[i]):
                items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
            else:
                items[-1] += " " + lines[i].strip()
            i += 1
        out.append(r"\begin{" + env + r"}\setlength\itemsep{1.5pt}\setlength\parskip{0pt}\setlength\topsep{2pt}" + NL
                   + NL.join(r"\item " + inline(t) for t in items) + NL + r"\end{" + env + "}")
        continue
    if not l.strip():
        i += 1
        continue
    para = []
    while i < len(lines) and lines[i].strip() and not START.match(lines[i]):
        para.append(lines[i])
        i += 1
    text = " ".join(para)
    if text.startswith("**Figure") or text.startswith("**Table"):     # captions: smaller, set off from the text
        out.append(("@@CAPT@@" if text.startswith("**Table") else "@@CAPF@@") + r"{\small " + inline(text) + r"\par}")
    elif section == "References":
        out.append(r"{\hangindent=1.6em " + inline(text) + r"\par}")
    else:
        out.append(inline(text) + NL)

# figures and tables float with their captions, as they would in the final LaTeX
merged, k, where = [], 0, "[!htb]"
while k < len(out):
    o = out[k]
    nxt = out[k + 1] if k + 1 < len(out) else ""
    if "APPSTART" in o:
        where = "[!h]"                            # appendix tables stay near where they are written
    if o.startswith("@@FIG@@") and nxt.startswith("@@CAPF@@"):
        merged.append(r"\begin{figure}" + where + r"\centering " + o[7:] + r"\par\vspace{3pt}\raggedright " + nxt[8:] + r"\end{figure}")
        k += 2
    elif o.startswith("@@CAPT@@") and nxt.startswith("@@TAB@@") and where == "[!h]":
        # appendix tables do not float, so they stay under their own heading
        merged.append(r"\par\medskip\noindent\begin{minipage}{\linewidth}\raggedright " + o[8:] + r"\vspace{3pt}\centering "
                      + nxt[7:] + r"\end{minipage}\par\medskip")
        k += 2
    elif o.startswith("@@CAPT@@") and nxt.startswith("@@TAB@@"):
        merged.append(r"\begin{table}" + where + r"\raggedright " + o[8:] + r"\vspace{3pt}\centering " + nxt[7:] + r"\end{table}")
        k += 2
    else:
        for tag in ("@@FIG@@", "@@TAB@@", "@@CAPT@@", "@@CAPF@@"):
            if o.startswith(tag):
                o = r"\begin{center}" + o[len(tag):] + r"\end{center}"
        merged.append(o)
        k += 1
out = merged

cover = (r"\begin{titlepage}\thispagestyle{empty}\vspace*{2.2cm}\begin{center}"
         + r"{\fontsize{19}{24}\selectfont\bfseries " + inline(title) + r"\par}\vspace{1.3cm}"
         + NL.join(r"{\large " + inline(h) + r"\par}\vspace{0.35cm}" for h in head_lines)
         + r"\end{center}\vspace{1.0cm}")
k = next(n for n, o in enumerate(out) if "tableofcontents" in o)
front, rest = out[:k], out[k:]
head = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=MARGIN]{geometry}
\usepackage{fontspec}\setmainfont{Arial}\setmonofont{Consolas}[Scale=0.85]
\usepackage{amsmath,amssymb,graphicx,tabularx,booktabs,array,titlesec}
\usepackage[hidelinks]{hyperref}\urlstyle{same}
\titleformat{\section}{\fontsize{13}{16}\selectfont\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\fontsize{11}{13.5}\selectfont\bfseries}{\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{12pt}{4pt}\titlespacing*{\subsection}{0pt}{9pt}{3pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{4.4pt}\frenchspacing\emergencystretch=3em
\setlength{\abovedisplayskip}{4pt}\setlength{\belowdisplayskip}{4pt}
\renewcommand{\topfraction}{0.92}\renewcommand{\bottomfraction}{0.85}\renewcommand{\textfraction}{0.06}
\renewcommand{\floatpagefraction}{0.85}\setcounter{topnumber}{3}\setcounter{bottomnumber}{2}\setcounter{totalnumber}{5}
\setlength{\textfloatsep}{8pt plus 2pt}\setlength{\intextsep}{7pt plus 2pt}\setlength{\floatsep}{7pt plus 2pt}
\begin{document}\pagenumbering{roman}
""".replace("MARGIN", margin)
doc = (head + cover + (NL + NL).join(front) + r"\end{titlepage}" + NL + (NL + NL).join(rest) + NL
       + r"\typeout{ENDPAGE \thepage}" + NL + r"\end{document}" + NL)
pathlib.Path(dst).write_text(doc, encoding="utf-8")
