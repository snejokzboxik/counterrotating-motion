"""Small independent P1 cross-check of the two electrostatic Dirichlet problems.
Uses circular polygons and the same constrained triangular mesh strategy as V17C.4.
No trajectory integration or fitting of forcing coefficients.
"""
import json
from pathlib import Path
import matplotlib.tri as mtri
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
import triangle

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT/"input/config.json").read_text())
# Unnormalised leading MFS coefficients from archive verify.log, p=48.
LEADS = np.array([0.8703168951521896-2.3335848298530805e-16j,
                  0.7982172694357996+3.745550400654463e-16j])
THETA = 2*np.pi*np.arange(24)/24
CENTERS = np.exp(1j*THETA)


def make_mesh(h, edges):
    circles = [1.10*np.exp(2j*np.pi*np.arange(384)/384)]
    circles += [z + .05*np.exp(2j*np.pi*np.arange(edges)/edges) for z in CENTERS]
    vertices, segments, labels = [], [], []
    for label, circle in enumerate(circles):
        nodes = len(vertices) + np.arange(len(circle))
        segments.extend(zip(nodes, np.roll(nodes,-1)))
        labels.extend([label+1]*len(nodes))
        vertices.extend(zip(circle.real,circle.imag))
    outline = dict(vertices=vertices, segments=segments, segment_markers=labels,
                   holes=np.c_[CENTERS.real, CENTERS.imag])
    return triangle.triangulate(outline, f"pq28a{h*h/2:.12f}Q")


def solve(h, edges):
    mesh = make_mesh(h,edges)
    pts, cells = mesh["vertices"],mesh["triangles"]
    seg, labels = mesh["segments"],mesh["segment_markers"].ravel()
    x,y = pts[cells].transpose(2,0,1)
    twice_area = (x[:,1]-x[:,0])*(y[:,2]-y[:,0])-(y[:,1]-y[:,0])*(x[:,2]-x[:,0])
    gx = (np.roll(y,-1,axis=1)-np.roll(y,-2,axis=1))/twice_area[:,None]
    gy = (np.roll(x,-2,axis=1)-np.roll(x,-1,axis=1))/twice_area[:,None]
    stiffness = (gx[:,:,None]*gx[:,None,:]+gy[:,:,None]*gy[:,None,:])*abs(twice_area[:,None,None])/2
    mat = coo_matrix((stiffness.ravel(),(np.repeat(cells,3,axis=1).ravel(),np.tile(cells,(1,3)).ravel())),
                     shape=(len(pts),len(pts))).tocsr()
    boundary = np.unique(seg)
    free = np.setdiff1d(np.arange(len(pts)),boundary)
    bc = np.zeros((len(pts),2),complex)
    for j,angle in enumerate(THETA):
        bc[np.unique(seg[labels==j+2])] = np.exp(1j*angle*np.array([2,-4]))
    rhs = -(mat[free][:,boundary] @ bc[boundary])
    lu = splu(mat[free][:,free].tocsc())
    bc[free] = lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
    triang = mtri.Triangulation(*pts.T, triangles=cells)
    angles = 2*np.pi*np.arange(2048)/2048
    results = []
    for family,k in enumerate([2,-4]):
        z = .90*.95*np.exp(1j*angles)
        interp = [mtri.LinearTriInterpolator(triang, part) for part in (bc[:,family].real,bc[:,family].imag)]
        values = interp[0](z.real,z.imag)+1j*interp[1](z.real,z.imag)
        assert not np.any(np.ma.getmaskarray(values))
        coefficient = complex(np.mean(values*np.exp(-1j*k*angles))/.90**abs(k))
        # Piecewise-constant Cartesian FEM field on the two nominal orbit circles.
        gradients = np.stack([np.sum(bc[cells,family]*d,axis=1) for d in (gx,gy)], axis=1)
        errors = {}
        for radius in (.304,.580):
            w = radius*np.exp(1j*angles)
            element = triang.get_trifinder()(w.real*.95,w.imag*.95)
            assert np.min(element)>=0
            exact = np.zeros((len(w),2),complex)
            for f,m,c in zip(CONFIG["family"],CONFIG["k"],CONFIG["coeff"]):
                if f == family:
                    dx = LEADS[family]*complex(*c)*abs(m)*(w if m>0 else np.conj(w))**(abs(m)-1)/.95
                    exact += np.column_stack((dx,1j*np.sign(m)*dx))
            errors[str(radius)] = float(np.linalg.norm(gradients[element]-exact)/np.linalg.norm(exact))
        results.append(dict(k=k, coefficient=[coefficient.real,coefficient.imag],
                            relative_complex_error=abs(coefficient/LEADS[family]-1),
                            relative_RMS_gradient_errors=errors))
    return dict(h_over_D=h,boundary_edges_per_rod=edges,nodes=len(pts),triangles=len(cells),bases=results)


if __name__=="__main__":
    output = [solve(.035,40),solve(.018,80)]
    (ROOT/"fem_check.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output))
