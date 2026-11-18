"""Independent P1 triangular FEM solve of the archived 2-D electrostatic sections."""
import json,math,sys,os
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
HERE=Path(__file__).resolve().parent
try:
    import triangle as tr
except ImportError:
    triangle_path=os.environ.get('V17C_TRIANGLE_PATH')
    if not triangle_path: raise
    sys.path.insert(0,triangle_path)
    import triangle as tr
import matplotlib
matplotlib.use('Agg')
import matplotlib.tri as mtri

def make_mesh(N,a,gap,hmax,nbound):
    B=1+a+gap;D=1.;verts=[];segs=[];markers=[];holes=[]
    outerN=max(160,int(math.ceil(2*math.pi*B/hmax)))
    angles=np.arange(outerN)*2*np.pi/outerN
    start=len(verts);verts.extend([[B*np.cos(q),B*np.sin(q)] for q in angles])
    for j in range(outerN):segs.append([start+j,start+(j+1)%outerN]);markers.append(N+1)
    for j in range(N):
        th=2*np.pi*j/N;center=np.array([D*np.cos(th),D*np.sin(th)])
        q=np.arange(nbound)*2*np.pi/nbound;start=len(verts)
        verts.extend((center+a*np.column_stack([np.cos(q),np.sin(q)])).tolist())
        for k in range(nbound):segs.append([start+k,start+(k+1)%nbound]);markers.append(j+1)
        holes.append(center.tolist())
    inp={'vertices':np.asarray(verts,float),'segments':np.asarray(segs,np.int32),
         'segment_markers':np.asarray(markers,np.int32)[:,None],'holes':np.asarray(holes,float)}
    maxarea=hmax*hmax*.5
    mesh=tr.triangulate(inp,f'pq28a{maxarea:.12g}Q')
    return mesh,B

def solve_fem(record,hmax,nbound):
    N=int(record['N']);m=int(record['m']);a=float(record['a_D']);gap=float(record['gap_D'])
    mesh,B=make_mesh(N,a,gap,hmax,nbound);pts=mesh['vertices'];tri=mesh['triangles'];segments=mesh['segments'];marks=mesh['segment_markers'].ravel()
    p=pts[tri];x=p[:,:,0];y=p[:,:,1]
    det=(x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(x[:,2]-x[:,0])*(y[:,1]-y[:,0]);area=np.abs(det)*.5
    b=np.column_stack([y[:,1]-y[:,2],y[:,2]-y[:,0],y[:,0]-y[:,1]])/(2*area[:,None])
    c=np.column_stack([x[:,2]-x[:,1],x[:,0]-x[:,2],x[:,1]-x[:,0]])/(2*area[:,None])
    ke=area[:,None,None]*(b[:,:,None]*b[:,None,:]+c[:,:,None]*c[:,None,:])
    rows=np.repeat(tri,3,axis=1).ravel();cols=np.tile(tri,(1,3)).ravel()
    K=coo_matrix((ke.ravel(),(rows,cols)),shape=(len(pts),len(pts))).tocsr()
    edge_nodes=segments.ravel();edge_marks=np.repeat(marks,2)
    fixed=np.unique(edge_nodes);bc_re=np.zeros(len(pts));bc_im=np.zeros(len(pts))
    for j in range(N):
        nodes=np.unique(segments[marks==j+1].ravel());phi=m*2*np.pi*j/N
        bc_re[nodes]=np.cos(phi);bc_im[nodes]=np.sin(phi)
    # Outer shield nodes stay grounded at zero.
    free=np.setdiff1d(np.arange(len(pts)),fixed,assume_unique=False)
    Kff=K[free][:,free];Kfb=K[free][:,fixed]
    u_re=bc_re.copy();u_im=bc_im.copy()
    u_re[free]=spsolve(Kff,-Kfb@bc_re[fixed]);u_im[free]=spsolve(Kff,-Kfb@bc_im[fixed])
    # Project the complex FEM solution onto signed solid harmonics on a central circle.
    radius=.35*(1-a);theta=np.arange(512)*2*np.pi/512
    xx=radius*np.cos(theta);yy=radius*np.sin(theta)
    triang=mtri.Triangulation(pts[:,0],pts[:,1],tri)
    real=np.asarray(mtri.LinearTriInterpolator(triang,u_re)(xx,yy))
    imag=np.asarray(mtri.LinearTriInterpolator(triang,u_im)(xx,yy))
    vals=real+1j*imag;stored={int(k):complex(re,im) for k,re,im in record['coefficients']}
    lead=m;counter=-(N-m)
    rnorm=radius/(1-a)
    coeff={k:complex(np.mean(vals*np.exp(-1j*k*theta))/(rnorm**abs(k))) for k in [lead,counter]}
    return dict(vertices=len(pts),triangles=len(tri),boundary_nodes=len(fixed),mesh_h=hmax,n_boundary=nbound,
      leading_FEM=[coeff[lead].real,coeff[lead].imag],leading_MFS=[stored[lead].real,stored[lead].imag],
      leading_abs_error=float(abs(coeff[lead]-stored[lead])),counter_FEM=[coeff[counter].real,coeff[counter].imag],
      counter_MFS=[stored[counter].real,stored[counter].imag],counter_abs_error=float(abs(coeff[counter]-stored[counter])))

def main():
    data=json.loads((HERE/'input'/'harmonics.json').read_text(encoding='utf-8'))
    specs={'6_ABC_moderate':[(.055,40),(.032,64),(.020,96)],
           '6_ABC_thin':[(.055,48),(.032,72),(.020,112)],
           '9_ABC_moderate':[(.055,40),(.032,64),(.020,96)],
           '12_ABC_moderate':[(.055,48),(.032,72),(.020,112)]}
    out=[]
    for name,levels in specs.items():
        rec=next(x for x in data if x['name']==name);rows=[]
        for h,n in levels:
            row=solve_fem(rec,h,n);rows.append(row)
            print(name,row,flush=True)
        out.append(dict(name=name,levels=rows))
    (HERE/'input'/'independent_fem_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')

if __name__=='__main__':main()
