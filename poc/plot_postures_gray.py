"""Four-panel grayscale posture figure for the AMCS article (package n10).

Panels: (a) reference and (b) selected posture of the immobilized baseline
(results/result.json); (c) reference and (d) selected posture of the
full-rank configuration (results/hard/result.json).  Same geometry and
witnesses as plot_clearance.py; only the presentation differs: shades of
grey, link 1 dashed, numbered links, orthographic view without axes and a
0.2 m scale bar, Times-metric lettering (Liberation Serif) converted to
vector outlines and a PDF with rasterized solids and vector lines.  The
final PDF has no fonts or transparency, as required by AMCS. The figure
contains no language-dependent words, so
the same file serves the English and the Polish version.

Usage (from the repository root):
    python poc/plot_postures_gray.py poc/results/result.json \
        poc/results/hard/result.json fig_g1postures_n10.pdf

Ghostscript (gs, gswin64c or mgs on PATH, or the executable named by the
AMCS_GHOSTSCRIPT environment variable) outlines the lettering; without it
the script keeps the Matplotlib PDF with embedded fonts and says so.
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', os.path.join(__import__('tempfile').gettempdir(), 'amcs-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import mujoco
from scipy.spatial import ConvexHull
from g1_geometry import G1Geometry

matplotlib.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                            'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Serif',
                            'mathtext.it': 'Liberation Serif:italic', 'font.size': 7.5, 'pdf.fonttype': 42})
XLIM, YLIM, ZLIM = (-.30, .45), (-.40, .40), (.66, 1.36)
SCALE = (0.0, 0.40, 0.63)
ZOOM = 1.55
DARK, MID, LIGHT = '#4a4a4a', '#9a9a9a', '#d4d4d4'


def solids(g):
    """Collision solids (group 3) and the visual waist meshes, as triangle arrays."""
    m = g.robot.model
    meshes, waist = {}, set()
    for gid in range(m.ngeom):
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid]))
        waist_context = m.geom_group[gid] == 2 and body in {'waist_yaw_link', 'waist_roll_link'}
        if m.geom_group[gid] != 3 and not waist_context:
            continue
        if any(k in body for k in ('hip', 'knee', 'ankle')):
            continue  # legs are outside the close-up
        if waist_context:
            waist.add(gid)
        if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[gid]
            start, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            pts = np.array(m.mesh_vert[start:start+n])
            if waist_context:
                first, count = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                meshes[gid] = pts[m.mesh_face[first:first+count]]
            else:
                meshes[gid] = pts[ConvexHull(pts).simplices]
        elif m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, z = m.geom_size[gid, :2]
            t = np.linspace(0, 2*np.pi, 33)[:-1]
            pts = np.array([[r*np.cos(a), r*np.sin(a), h] for h in [-z, z] for a in t])
            meshes[gid] = pts[ConvexHull(pts).simplices]
    return meshes, waist


def exit_parameter(p, a):
    """Largest s in [0,1] such that p+s(a-p) stays in the plotted box."""
    s_max = 1.0
    for k, (lo, hi) in enumerate([XLIM, YLIM, ZLIM]):
        d = a[k]-p[k]
        if d > 0:
            s_max = min(s_max, (hi-p[k])/d)
        elif d < 0:
            s_max = min(s_max, (lo-p[k])/d)
    return s_max


def panel(fig, rect, g, meshes, waist, q, N, label, scale=False):
    g.pose(q)
    d = g.robot.data
    ax = fig.add_axes(rect, projection='3d', proj_type='ortho', computed_zorder=False)
    moving = {gid for gid, _, _ in g.geoms}
    light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
    for gid, tris in meshes.items():
        world = tris@d.geom_xmat[gid].reshape(3, 3).T+d.geom_xpos[gid]
        base = DARK if gid in moving else MID if gid in waist else LIGHT
        rgb = np.array(matplotlib.colors.to_rgb(base))
        n = np.cross(world[:, 1]-world[:, 0], world[:, 2]-world[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        shade = 0.80+0.20*np.abs(n@light)
        cols = np.clip(rgb[None, :]*shade[:, None], 0, 1)
        pc = Poly3DCollection(world, facecolors=cols, linewidths=0, zorder=1)
        pc.set_rasterized(True)
        ax.add_collection3d(pc)
    for i, a in enumerate(g.anchors):
        s = exit_parameter(g.p, a)
        seg = np.array([g.p, g.p+s*(a-g.p)])
        ax.plot(*seg.T, color='black', lw=1.1 if i == 0 else 0.6, ls=(0, (4, 2)) if i == 0 else '-', zorder=5)
        lab = g.p+0.80*s*(a-g.p)
        ax.text(*lab, str(i+1), fontsize=6.5, ha='center', va='center', zorder=8,
                bbox=dict(boxstyle='circle,pad=0.15', fc='white', ec='black', lw=0.5))
    c, w = g.clearance(q, N, witnesses=True)
    hit = np.array(w[0]['point_m'])
    ax.scatter(*g.p, marker='D', s=12, c='black', edgecolors='white', linewidths=0.4, zorder=10, depthshade=False)
    ax.scatter(*hit, marker='x', s=30, c='black', linewidths=1.2, zorder=9, depthshade=False)
    if scale:
        x0, y0, z0 = SCALE
        ax.plot([x0, x0], [y0, y0], [z0, z0+0.2], color='black', lw=1.0, zorder=6)
        ax.text(x0, y0, z0+0.1, '0.2 m ', fontsize=6.5, ha='right', va='center', zorder=8)
    ax.set(xlim=XLIM, ylim=YLIM, zlim=ZLIM)
    ax.set_box_aspect((XLIM[1]-XLIM[0], YLIM[1]-YLIM[0], ZLIM[1]-ZLIM[0]), zoom=ZOOM)
    ax.view_init(18, 28)
    ax.set_axis_off()
    fig.text(rect[0]+0.01, 0.97, label, ha='left', va='top')
    fig.text(rect[0]+0.01, 0.885, r'$\hat{c}_1=%s$ m' % ('%.4g' % c[0]).replace('-', '\u2212'), ha='left', va='top', fontsize=7)
    return c


def main(base_json, hard_json, out_pdf):
    ghostscript = os.environ.get('AMCS_GHOSTSCRIPT') or next(
        (p for name in ('gswin64c', 'gs', 'mgs') if (p := shutil.which(name))), None)
    if not ghostscript:
        # The outlining step is cosmetic (AMCS wants a PDF without embedded fonts); the geometry,
        # the witnesses and the printed clearances do not depend on it.
        print('Ghostscript not found: keeping the Matplotlib PDF with embedded fonts.', file=sys.stderr)
    fig = plt.figure(figsize=(17/2.54, 4.4/2.54))
    labels = [r'(a) $q_0$', r'(b) $q_b$', r'(c) $q_0$', r'(d) $q_b$']
    k = 0
    values = {}
    for path in [base_json, hard_json]:
        result = json.loads(Path(path).read_text())
        g = G1Geometry(result['config'])
        meshes, waist = solids(g)
        for key in ['reference', 'selected']:
            c = panel(fig, [0.25*k, -0.07, 0.25, 1.07], g, meshes, waist, result[key]['q'],
                      result['config']['budget']['N'], labels[k], scale=(k == 0))
            values[labels[k]] = dict(q=result[key]['q'], c_hat=c.tolist(), stored_c_hat=result[key]['c_hat'])
            k += 1
    out_pdf = Path(out_pdf)
    font_pdf = out_pdf.with_suffix('.fonts.pdf') if ghostscript else out_pdf
    fig.savefig(font_pdf, dpi=600)
    plt.close(fig)
    if not ghostscript:
        print(json.dumps(values, indent=1))
        return
    subprocess.run([ghostscript, '-dSAFER', '-dBATCH', '-dNOPAUSE',
                    '-dNoOutputFonts', '-sDEVICE=pdfwrite',
                    '-dCompatibilityLevel=1.3', '-dAutoRotatePages=/None',
                    '-o', str(out_pdf), str(font_pdf)], check=True)
    font_pdf.unlink()
    print(json.dumps(values, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:4])
