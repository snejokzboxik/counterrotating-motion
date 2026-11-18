from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import sys, hashlib, json, numpy as np

def extract(archive, dest):
    dest.mkdir(exist_ok=True)
    with ZipFile(archive) as z:
        with np.load(BytesIO(z.read("v17c_dualfreq/verification_coeffs.npz"))) as field, np.load(BytesIO(z.read("v17c_dualfreq/base128.npz"))) as run:
            c = field["coeff"]
            cfg = dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), initial=run["initial"].tolist(),
                       parameters=run["p"].tolist(), parameter_names=["A","B","nu","beta","s","C","lambda1","lambda2"],
                       family=field["family"].tolist(), k=field["k"].tolist(), coeff=np.column_stack((c.real,c.imag)).tolist(),
                       f1_Hz=100000, electrodes=24, a_D=0.05, B_D=1.10,
                       units="position/R, velocity/(R*Omega1), tau=Omega1*t, energy/(m1*R^2*Omega1^2)")
    (dest/"config.json").write_text(json.dumps(cfg,indent=2)+"\n", encoding="utf-8")

if __name__ == "__main__":
    extract(Path(sys.argv[1]),Path(__file__).resolve().parent/"input")
