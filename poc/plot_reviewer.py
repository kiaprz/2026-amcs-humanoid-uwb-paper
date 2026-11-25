"""Static scientific figures from saved sensitivity inputs and trajectories."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(__import__('tempfile').gettempdir(), 'amcs-mpl'))
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from diffraction_model import DiffractionEnvelope

out=Path('poc/results/reviewer')
config=json.loads(Path('poc/config_hard.json').read_text())
physical=json.loads((out/'physical_sensitivity.json').read_text())
d=np.linalg.norm(config['anchor_offsets_m'],axis=1)
env=DiffractionEnvelope(d,physical['torso_intersection_lengths_m'])
c=np.linspace(-.06,.35,300)
fig,axs=plt.subplots(1,2,figsize=(11,4.3))
for i,color in [(0,'#2563eb'),(2,'#c2410c')]:
    low=np.array([env.minus(np.full(4,x))[i] for x in c])
    high=np.array([env.plus(np.full(4,x))[i] for x in c])
    axs[0].fill_between(c,low,high,alpha=.22,color=color,label=f'Link {i+1}: model envelope')
    axs[0].plot(c,low,color=color);axs[0].plot(c,high,color=color,linestyle='--')
axs[0].set(xlabel='Moving-arm signed clearance [m]',ylabel='Normalized information weight',ylim=(0,1.04))
axs[0].legend(fontsize=9);axs[0].grid(alpha=.2)
rows=np.genfromtxt(out/'dynamics/trace.csv',delimiter=',',names=True)
rows=rows[rows['trial']==0]
axs[1].plot(rows['time_after_settle_s'],rows['tag_displacement_m']*1000,label='Tag displacement')
axs[1].plot(rows['time_after_settle_s'],rows['segment_frame_displacement_bound_m']*1000,label='Segment-frame displacement bound')
axs[1].axhline(2,color='black',ls=':',label='Assumed kinematic budget: 2 mm')
axs[1].set(xlabel='Time after standing settlement [s]',ylabel='Empirical displacement [mm]')
axs[1].legend(fontsize=9);axs[1].grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=200)
