"""Plot the recorded bounds; never reruns or infers the search."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(__import__('tempfile').gettempdir(), 'amcs-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
for name in sys.argv[1:]:
    path=Path(name);r=json.loads(path.read_text());h=r['history']
    plt.rcParams.update({'font.size':13,'axes.labelsize':14,'axes.titlesize':14,'xtick.labelsize':12,'ytick.labelsize':12})
    fig,ax=plt.subplots(figsize=(5.4,4.5))
    x=[v['splits'] for v in h]
    if r['LB'] is not None:ax.plot(x,[v['LB'] for v in h],label='LB')
    ax.plot(x,[v['UB'] for v in h],label='UB')
    ax.axhline(r['exclusion_threshold'],color='black',ls='--',label='Exclusion threshold')
    ax.axhline(r['geometry']['f_cap'],color='gray',ls=':',label='Calibration ceiling')
    ax.set(xlabel='subdivisions',ylabel='normalized information',title='Branch-and-bound history')
    ax.legend(fontsize=11,loc='lower right',bbox_to_anchor=(1,.10));fig.tight_layout();fig.savefig(path.parent/'convergence.png',dpi=240);plt.close(fig)
