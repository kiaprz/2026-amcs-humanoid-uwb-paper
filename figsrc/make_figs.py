"""Generate standalone TikZ/pgfplots sources for the n10 figures (PL and EN)."""
import argparse, math, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent

PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{arrows.meta}
\begin{document}
"""

LANG = {
 "PL": dict(
  a="(a) poprawa zagwarantowana: $\\mathrm{LB}>f^{+}(q_0)+\\varepsilon_g$",
  b="(b) poprawa większa niż $\\varepsilon_g$ wykluczona: $\\mathrm{UB}\\le f^{-}(q_0)+\\varepsilon_g$",
  c="(c) wynik nierozstrzygnięty: nie zachodzi ani (a), ani (b)",
  axis="wartość funkcji celu $f-\\kappa$ (dla $\\beta=0$: wskaźnik $f$)",
  q0="$[f^{-}(q_0),f^{+}(q_0)]$",
  ga="(a) przekrój płaszczyzną łącza 1",
  gb="(b) obwiednie kalibracyjne",
  gc="(c) wskaźniki i ich ograniczenia",
  xs="wzdłuż łącza 1 od znacznika [m]",
  ys="wzdłuż $u_\\perp$ [m]",
  tag="znacznik",
  anchor="do kotwicy 1",
  pivot="oś obrotu",
  xc="prześwit $c$ [m]",
  yc="waga znormalizowana",
  xq="$q$ [rad]",
  yq="wartość wskaźnika",
 ),
 "EN": dict(
  a="(a) guaranteed improvement: $\\mathrm{LB}>f^{+}(q_0)+\\varepsilon_g$",
  b="(b) improvement larger than $\\varepsilon_g$ excluded: $\\mathrm{UB}\\le f^{-}(q_0)+\\varepsilon_g$",
  c="(c) inconclusive: neither (a) nor (b) holds",
  axis="objective value $f-\\kappa$ (for $\\beta=0$: the index $f$)",
  q0="$[f^{-}(q_0),f^{+}(q_0)]$",
  ga="(a) cross-section in the plane of link 1",
  gb="(b) calibration envelopes",
  gc="(c) indices and their bounds",
  xs="along link 1 from the tag [m]",
  ys="along $u_\\perp$ [m]",
  tag="tag",
  anchor="to anchor 1",
  pivot="rotation axis",
  xc="clearance $c$ [m]",
  yc="normalized weight",
  xq="$q$ [rad]",
  yq="index value",
 ),
}


def decision(L):
    # scale: value v -> x = (v-0.2)*20 cm  (0.2..0.9 -> 0..14 cm)
    def X(v):
        return (v - 0.2) * 20.0
    rows = []
    # (label, LB, UB) ; f-(q0)=0.40, f+(q0)=0.46, eps_g=0.10 -> thresholds 0.50, 0.56
    cases = [(L["a"], 0.62, 0.80), (L["b"], 0.30, 0.48), (L["c"], 0.53, 0.72)]
    fm, fp, eg = 0.40, 0.46, 0.10
    s = []
    for k, (lab, lb, ub) in enumerate(cases):
        y = -1.25 * k
        s.append(f"\\node[anchor=west,font=\\footnotesize] at (0,{y+0.42:.2f}) {{{lab}}};")
        s.append(f"\\draw[gray!70,line width=0.4pt] (0,{y:.2f}) -- (14,{y:.2f});")
        s.append(f"\\fill[gray!35] ({X(fm):.2f},{y-0.12:.2f}) rectangle ({X(fp):.2f},{y+0.12:.2f});")
        s.append(f"\\draw ({X(fm):.2f},{y-0.12:.2f}) rectangle ({X(fp):.2f},{y+0.12:.2f});")
        for t in (fm + eg, fp + eg):
            s.append(f"\\draw[densely dotted,line width=0.6pt] ({X(t):.2f},{y-0.28:.2f}) -- ({X(t):.2f},{y+0.24:.2f});")
        s.append(f"\\draw[line width=1.6pt] ({X(lb):.2f},{y-0.02:.2f}) -- ({X(ub):.2f},{y-0.02:.2f});")
        s.append(f"\\draw[line width=0.8pt] ({X(lb):.2f},{y-0.16:.2f}) -- ({X(lb):.2f},{y+0.12:.2f});")
        s.append(f"\\draw[line width=0.8pt] ({X(ub):.2f},{y-0.16:.2f}) -- ({X(ub):.2f},{y+0.12:.2f});")
        s.append(f"\\node[font=\\scriptsize,below] at ({X(lb):.2f},{y-0.14:.2f}) {{$\\mathrm{{LB}}$}};")
        s.append(f"\\node[font=\\scriptsize,below] at ({X(ub):.2f},{y-0.14:.2f}) {{$\\mathrm{{UB}}$}};")
    y = -1.25 * 3 + 0.35
    s.append(f"\\draw[-{{Stealth[length=2mm]}}] (0,{y:.2f}) -- (14.3,{y:.2f});")
    for v in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        s.append(f"\\draw ({X(v):.2f},{y:.2f}) -- ({X(v):.2f},{y-0.08:.2f}) node[below,font=\\scriptsize] {{{v:.1f}}};")
    s.append(f"\\node[font=\\footnotesize] at (7,{y-0.62:.2f}) {{{L['axis']}}};")
    return PRE + "\\begin{tikzpicture}[x=0.6cm,y=1cm]\n" + "\n".join(s) + "\n\\end{tikzpicture}\n\\end{document}\n"


def instance(L):
    RO, K0, RK = 0.30, 0.55, 0.05
    qA = math.asin(1/6)
    qs = [(0.0, "$q=0$"), (qA, "$q=q_A$"), (math.pi/6, "$q=\\pi/6$")]
    geo = []
    geo.append(r"\begin{axis}[width=6.3cm,height=5.0cm,axis equal image,xmin=-0.04,xmax=0.64,ymin=-0.09,ymax=0.22,"
               r"xlabel={" + L["xs"] + r"},ylabel={" + L["ys"] + r"},title={" + L["ga"] + r"},"
               r"title style={font=\footnotesize},label style={font=\footnotesize},tick label style={font=\scriptsize},"
               r"xtick={0,0.2,0.4,0.6},ytick={0,0.1,0.2}]")
    geo.append(r"\addplot[line width=1.0pt] coordinates {(0,0) (0.64,0)};")
    geo.append(r"\node[font=\scriptsize,anchor=north east] at (axis cs:0.64,-0.006) {" + L["anchor"] + r" $\rightarrow$};")
    geo.append(r"\addplot[only marks,mark=square*,mark size=1.6pt] coordinates {(0,0)};")
    geo.append(r"\node[font=\scriptsize,anchor=south west] at (axis cs:-0.03,0.010) {" + L["tag"] + r"};")
    geo.append(r"\addplot[only marks,mark=+,mark size=2.2pt] coordinates {(0.55,0)};")
    geo.append(r"\node[font=\scriptsize,anchor=south] at (axis cs:0.55,0.012) {" + L["pivot"] + r"};")
    shades = ["gray!70", "gray!45", "gray!20"]
    for (t, lab), sh in zip(qs, shades):
        cx, cy = K0 - RO*math.cos(t), RO*math.sin(t)
        geo.append(f"\\draw[fill={sh},line width=0.5pt] (axis cs:{cx:.5f},{cy:.5f}) circle[radius=0.05];")
        if t > 0:
            geo.append(f"\\node[font=\\scriptsize,anchor=west] at (axis cs:{cx+0.052:.5f},{cy+0.02:.5f}) {{{lab}}};")
        else:
            geo.append(f"\\node[font=\\scriptsize,anchor=east] at (axis cs:{cx-0.052:.5f},{cy-0.035:.5f}) {{{lab}}};")
    # path of the centre, drawn last so that it stays visible over the spheres
    pts = " ".join(f"({K0-RO*math.cos(t):.5f},{RO*math.sin(t):.5f})" for t in [i*(math.pi/6)/60 for i in range(61)])
    geo.append(r"\draw[densely dashed,line width=0.6pt] plot coordinates {" + pts + "};")
    geo.append(r"\end{axis}")

    env = []
    env.append(r"\begin{axis}[width=5.6cm,height=5.0cm,xmin=-0.06,xmax=0.10,ymin=0,ymax=1.08,"
               r"xlabel={" + L["xc"] + r"},ylabel={" + L["yc"] + r"},title={" + L["gb"] + r"},"
               r"title style={font=\footnotesize},label style={font=\footnotesize},tick label style={font=\scriptsize},"
               r"scaled x ticks=false,xtick={-0.04,0,0.04,0.08},xticklabels={$-0.04$,$0$,$0.04$,$0.08$},"
               r"legend style={font=\scriptsize,at={(0.97,0.03)},anchor=south east,draw=none,fill=none}]")
    env.append(r"\addplot[line width=1.0pt] table[x=c,y=u] {plot_env.dat};")
    env.append(r"\addlegendentry{$\psi^{+}$}")
    env.append(r"\addplot[line width=1.0pt,densely dashed] table[x=c,y=l] {plot_env.dat};")
    env.append(r"\addlegendentry{$\psi^{-}$}")
    env.append(r"\end{axis}")

    ind = []
    ind.append(r"\begin{axis}[width=5.6cm,height=5.0cm,xmin=0,xmax=0.5236,ymin=0.2,ymax=1.42,"
               r"xlabel={" + L["xq"] + r"},ylabel={" + L["yq"] + r"},title={" + L["gc"] + r"},"
               r"title style={font=\footnotesize},label style={font=\footnotesize},tick label style={font=\scriptsize},"
               r"xtick={0,0.1,0.2,0.3,0.4,0.5},"
               r"legend style={font=\scriptsize,at={(0.98,0.03)},anchor=south east,draw=none,fill=white}]")
    ind.append(r"\addplot[line width=0.5pt,gray] table[x=q,y=fpA] {plot_phi.dat};")
    ind.append(r"\addlegendentry{$f^{+}$}")
    ind.append(r"\addplot[line width=1.0pt] table[x=q,y=phip] {plot_phi.dat};")
    ind.append(r"\addlegendentry{$\Phi^{+}$}")
    ind.append(r"\addplot[line width=1.0pt,densely dashed] table[x=q,y=phim] {plot_phi.dat};")
    ind.append(r"\addlegendentry{$\Phi^{-}$}")
    ind.append(r"\addplot[line width=0.5pt,gray,densely dashed] table[x=q,y=fmA] {plot_phi.dat};")
    ind.append(r"\addlegendentry{$f^{-}$}")
    ind.append(r"\end{axis}")

    body = (r"\begin{tikzpicture}" + "\n".join(geo) + r"\end{tikzpicture}\hspace{2mm}"
            + r"\begin{tikzpicture}" + "\n".join(env) + r"\end{tikzpicture}\hspace{2mm}"
            + r"\begin{tikzpicture}" + "\n".join(ind) + r"\end{tikzpicture}")
    return PRE + body + "\n\\end{document}\n"


def plot_data():
    """Regenerate plot_env.dat and plot_phi.dat (Fig. 1b,c: envelopes, nominal
    envelope indices and budget-A bounds) from poc/paper_example.py."""
    sys.path.insert(0, str(HERE.parent / "poc"))
    from paper_example import figure_data
    for name, text in zip(("plot_env.dat", "plot_phi.dat"), figure_data()):
        (HERE / name).write_text(text, encoding="ascii", newline="\n")
        print(name, "written")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true",
                        help="regenerate the plot data files and skip LaTeX")
    args = parser.parse_args()
    plot_data()
    if args.data_only:
        sys.exit(0)
    for lang, L in LANG.items():
        for name, fn in (("fig_decision", decision), ("fig_instance", instance)):
            tex = HERE / f"{name}_n10_{lang}.tex"
            tex.write_text(fn(L), encoding="utf-8")
            r = subprocess.run(["pdflatex", "-disable-installer", "-disable-write18", "-interaction=nonstopmode", "-halt-on-error", tex.name],
                               cwd=HERE, capture_output=True, text=True)
            ok = (HERE / f"{name}_n10_{lang}.pdf").exists() and r.returncode == 0
            print(name, lang, "OK" if ok else "FAIL")
            if not ok:
                print(r.stdout[-3000:])
