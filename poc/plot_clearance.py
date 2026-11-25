"""Collision-solid posture visualization with projected, always-visible witnesses."""
import os
os.environ.setdefault('MPLCONFIGDIR', os.path.join(__import__('tempfile').gettempdir(), 'amcs-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from scipy.spatial import ConvexHull
import numpy as np
import mujoco
from pathlib import Path
import json
from g1_geometry import G1Geometry


def render(result_path):
    result=json.loads(result_path.read_text()); g=G1Geometry(result['config'])
    fig=plt.figure(figsize=(12,5.7))
    m=g.robot.model
    meshes={}
    visual_waist=set()
    for gid in range(m.ngeom):
        body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[gid]))
        waist_context=m.geom_group[gid]==2 and body in {'waist_yaw_link','waist_roll_link'}
        if m.geom_group[gid]!=3 and not waist_context: continue
        if waist_context: visual_waist.add(gid)
        if m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_MESH:
            mid=m.geom_dataid[gid]; start,n=m.mesh_vertadr[mid],m.mesh_vertnum[mid]
            pts=np.array(m.mesh_vert[start:start+n])
            if waist_context:
                first,count=m.mesh_faceadr[mid],m.mesh_facenum[mid]
                meshes[gid]=pts[m.mesh_face[first:first+count]]
            else:
                meshes[gid]=pts[ConvexHull(pts).simplices]
        elif m.geom_type[gid]==mujoco.mjtGeom.mjGEOM_CYLINDER:
            r,z=m.geom_size[gid,:2]; t=np.linspace(0,2*np.pi,33)[:-1]
            pts=np.array([[r*np.cos(a),r*np.sin(a),h] for h in [-z,z] for a in t])
            meshes[gid]=pts[ConvexHull(pts).simplices]
    moving={gid for gid,_,_ in g.geoms}
    for k,key in enumerate(['reference','selected'],1):
        q=result[key]['q']; g.pose(q); d=g.robot.data
        ax=fig.add_subplot(1,2,k,projection='3d')
        for gid,tris in meshes.items():
            world=tris@d.geom_xmat[gid].reshape(3,3).T+d.geom_xpos[gid]
            color='#e3933a' if gid in moving else '#7f8b99' if gid in visual_waist else '#acb8c7'
            ax.add_collection3d(Poly3DCollection(world,facecolor=color,linewidths=0,alpha=.8))
        for i,a in enumerate(g.anchors):
            ax.plot(*np.array([g.p,a]).T,color=['red','#2563eb','#16a34a','#9333ea'][i],lw=2 if i==0 else 1.3,ls='--' if i==0 else '-')
        c,w=g.clearance(q,result['config']['budget']['N'],witnesses=True); hit=np.array(w[0]['point_m'])
        ax.set(xlim=(-.35,.5),ylim=(-.45,.45),zlim=(.65,1.45),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]',title=('Reference q0' if k==1 else 'Selected qb')+'\nq='+str(q))
        ax.set_box_aspect((.85,.9,.8));ax.view_init(18,28)
        # Project after setting limits: annotations overlay solids without hiding witnesses.
        for point,label,color,pos,symbol in [(g.p,'TAG','black',(.03,.85),'◆'),(hit,'Link 1: min grid clearance\n'+f'{c[0]:.4g} m','red',(.52,.76),'×')]:
            x,y,_=proj3d.proj_transform(*point,ax.get_proj())
            ax.annotate(symbol,(x,y),color=color,fontsize=17,ha='center',va='center',zorder=1000)
            ax.annotate(label,(x,y),xytext=pos,textcoords='axes fraction',color=color,fontsize=10,
                        arrowprops=dict(arrowstyle='->',color=color,lw=1.2),zorder=1000)
    fig.suptitle('G1 collision solids with visual waist; moving left arm in orange; link 1 dashed red')
    fig.tight_layout();fig.savefig(result_path.parent/'postures_clearance.png',dpi=180);plt.close(fig)

if __name__=='__main__':
    import sys
    for name in sys.argv[1:]:render(Path(name))
