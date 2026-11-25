"""Render sampled maps from saved NPZ; no additional posture evaluations."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(__import__('tempfile').gettempdir(), 'amcs-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path


def plot_map(x,y,z,result,out):
    with plt.rc_context({'font.size':13,'axes.labelsize':14,'axes.titlesize':14,
                         'xtick.labelsize':12,'ytick.labelsize':12}):
        fig,ax=plt.subplots(figsize=(5.4,4.5))
        im=ax.pcolormesh(x,y,z,shading='auto',cmap='viridis')
        fig.colorbar(im,ax=ax,pad=.025,label=r'Sampled $f^-$')
        threshold=result['improvement_threshold']
        if z.min()<threshold<z.max():
            ax.contour(x,y,z,levels=[threshold],colors='white',linewidths=2)
            ax.contourf(x,y,z,levels=[threshold,z.max()+1],colors='none',hatches=['///'])
        for key,label,marker in [('reference',r'$q_0$','o'),('selected',r'$q_b$','*')]:
            a,c=result[key]['q']
            ax.scatter(a,c,marker=marker,color='red',edgecolor='white',s=130,zorder=5,clip_on=False)
            ax.annotate(label,(a,c),xytext=(10,16) if key=='reference' else (-42,-38),
                        textcoords='offset points',color='#b91c1c',fontsize=16,
                        arrowprops=dict(arrowstyle='-',color='#b91c1c',lw=1),
                        bbox=dict(facecolor='white',alpha=.8,edgecolor='none',pad=1))
        ax.set(xlabel='Shoulder pitch [rad]',ylabel='Elbow [rad]',title=r'Sampled $f^-$ map')
        fig.tight_layout();fig.savefig(out/'map.png',dpi=240);plt.close(fig)


if __name__=='__main__':
    import sys
    for name in sys.argv[1:]:
        path=Path(name);r=json.loads(path.read_text());m=np.load(path.parent/'map.npz')
        plot_map(m['shoulder_pitch'],m['elbow'],m['f_minus'],r,path.parent)
