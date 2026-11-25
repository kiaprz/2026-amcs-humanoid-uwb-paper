"""Vector grayscale redraw of the article's G1 map figure from stored PoC data.

Inputs (read only): poc/results/hard/map.npz, poc/results/hard/result.json,
poc/results/bnb/result.json.  No posture is re-evaluated; the script only
re-plots stored numbers, exactly as poc/plot_map.py and poc/plot_bnb.py do.
"""
import json, pathlib, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RES = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "poc" / "results"

PRE = r"""\documentclass[border=1pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{mathptmx}
\usepackage{amsmath}
\usepackage{tikz}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\usetikzlibrary{patterns}
\begin{document}
"""
L = {
 "EN": dict(xa="shoulder pitch angle [rad]", ya="elbow angle [rad]", ta="(a) sampled lower bound $f^-$",
            cb="sampled $f^-$", tb="(b) branch-and-bound history", xb="subdivisions", yb="bound value",
            thr="exclusion threshold", cap="calibration ceiling $f^{\\mathrm{cap}}$"),
 "PL": dict(xa="kąt barku [rad]", ya="kąt łokcia [rad]", ta="(a) próbkowane dolne ograniczenie $f^-$",
            cb="próbkowane $f^-$", tb="(b) przebieg metody podziału i ograniczeń", xb="liczba podziałów", yb="wartość ograniczenia",
            thr="próg wykluczenia", cap="pułap kalibracyjny $f^{\\mathrm{cap}}$"),
}

m = np.load(RES / "hard" / "map.npz")
x, y, z = m["shoulder_pitch"], m["elbow"], m["f_minus"]
hard = json.loads((RES / "hard" / "result.json").read_text())
thr_imp = hard["improvement_threshold"]
q0 = hard["reference"]["q"]; qb = hard["selected"]["q"]
# filled region f^- > threshold, exactly as in poc/plot_map.py (contourf), and its boundary
fig, ax = plt.subplots()
cf = ax.contourf(x, y, z, levels=[thr_imp, z.max() + 1])
cl = ax.contour(x, y, z, levels=[thr_imp])
polys = [p for p in cf.allsegs[0]]
lines = [p for p in cl.allsegs[0]]
plt.close(fig)

bnb = json.loads((RES / "bnb" / "result.json").read_text())
h = bnb["history"]
# keep points where UB or LB changes (step plot), plus endpoints
pts = []
for k, v in enumerate(h):
    if k == 0 or k == len(h) - 1 or v["UB"] != h[k - 1]["UB"] or v["LB"] != h[k - 1]["LB"]:
        pts.append((v["splits"], v["LB"], v["UB"]))
thr_exc = bnb["exclusion_threshold"]; fcap = bnb["geometry"]["f_cap"]

def coords(arr):
    return " ".join(f"({a:.5f},{b:.5f})" for a, b in arr)

def build(lang):
    T = L[lang]
    rows = "\n".join(f"{x[j]:.6f} {y[i]:.6f} {z[i, j]:.6f}" for i in range(len(y)) for j in range(len(x)))
    s = []
    s.append(r"\begin{tikzpicture}")
    s.append(r"\begin{axis}[name=A,width=6.4cm,height=5.6cm,enlargelimits=false,"
             r"xmin=%.4f,xmax=%.4f,ymin=%.4f,ymax=%.4f," % (x[0] - (x[1]-x[0])/2, x[-1] + (x[1]-x[0])/2, y[0] - (y[1]-y[0])/2, y[-1] + (y[1]-y[0])/2) +
             r"colormap={bw}{gray(0cm)=(0.25); gray(1cm)=(0.95)},colorbar,"
             r"colorbar style={width=2.5mm,ylabel={%s},ylabel style={font=\footnotesize},yticklabel style={font=\scriptsize,/pgf/number format/fixed,/pgf/number format/precision=2}}," % T["cb"] +
             r"xlabel={%s},ylabel={%s},title={%s},title style={font=\footnotesize},label style={font=\footnotesize},tick label style={font=\scriptsize},"
             % (T["xa"], T["ya"], T["ta"]) +
             r"xtick={-0.6,-0.5,-0.4,-0.3},ytick={1.0,1.1,1.2,1.3,1.4},xticklabel style={/pgf/number format/fixed},clip=true]")
    s.append(r"\addplot[matrix plot*,mesh/cols=%d,point meta=explicit,shader=flat corner] table[meta index=2] {" % len(x))
    s.append(rows)
    s.append(r"};")
    for p in polys:
        s.append(r"\fill[pattern=north east lines,pattern color=black] " + " -- ".join(f"(axis cs:{a:.5f},{b:.5f})" for a, b in p) + " -- cycle;")
    for p in lines:
        s.append(r"\addplot[line width=1.4pt,white] coordinates {" + coords(p) + "};")
        s.append(r"\addplot[line width=0.5pt,black,densely dashed] coordinates {" + coords(p) + "};")
    s.append(r"\addplot[only marks,mark=*,mark size=2.6pt,fill=white,draw=black,clip mode=individual] coordinates {(%.4f,%.4f)};" % tuple(q0))
    s.append(r"\addplot[only marks,mark=star,mark size=3.6pt,line width=0.9pt,black,clip mode=individual] coordinates {(%.4f,%.4f)};" % tuple(qb))
    s.append(r"\node[font=\footnotesize,anchor=south west,fill=white,inner sep=1pt] at (axis cs:%.4f,%.4f) {$q_0$};" % (q0[0] + 0.012, q0[1] + 0.02))
    s.append(r"\node[font=\footnotesize,anchor=north east,fill=white,inner sep=1pt] at (axis cs:%.4f,%.4f) {$q_b$};" % (qb[0] - 0.012, qb[1] - 0.02))
    s.append(r"\end{axis}")
    s.append(r"\begin{axis}[at={(A.east)},anchor=west,xshift=3.5cm,width=6.0cm,height=5.6cm,xmin=0,xmax=%d,ymin=0.20,ymax=0.60," % (pts[-1][0] + 20) +
             r"xlabel={%s},ylabel={%s},title={%s},title style={font=\footnotesize},label style={font=\footnotesize},tick label style={font=\scriptsize}," % (T["xb"], T["yb"], T["tb"]) +
             r"legend cell align=left,legend style={font=\scriptsize,at={(0.97,0.14)},anchor=south east,draw=none,fill=white},scaled x ticks=false,xticklabel style={/pgf/number format/1000 sep={}}]")
    s.append(r"\addplot[line width=0.5pt,gray,densely dotted,domain=0:%d] {%.6f};" % (pts[-1][0] + 20, fcap))
    s.append(r"\addlegendentry{%s}" % T["cap"])
    s.append(r"\addplot[line width=1.2pt,const plot] coordinates {" + coords([(a, u) for a, _, u in pts]) + "};")
    s.append(r"\addlegendentry{$\mathrm{UB}$}")
    s.append(r"\addplot[line width=0.7pt,densely dashed,domain=0:%d] {%.6f};" % (pts[-1][0] + 20, thr_exc))
    s.append(r"\addlegendentry{%s}" % T["thr"])
    s.append(r"\addplot[line width=0.8pt,gray,const plot] coordinates {" + coords([(a, l) for a, l, _ in pts]) + "};")
    s.append(r"\addlegendentry{$\mathrm{LB}$}")
    s.append(r"\end{axis}")
    s.append(r"\end{tikzpicture}")
    return PRE + "\n".join(s) + "\n\\end{document}\n"

for lang in ("EN", "PL"):
    f = HERE / f"fig_g1map_n10_{lang}.tex"
    f.write_text(build(lang), encoding="utf-8")
    r = subprocess.run(["pdflatex", "-disable-installer", "-disable-write18", "-interaction=nonstopmode", f.name], cwd=HERE, capture_output=True, text=True)
    print(lang, "OK" if r.returncode == 0 else "FAIL", "threshold", thr_imp, "exclusion", thr_exc, "fcap", fcap, "npts", len(pts))
