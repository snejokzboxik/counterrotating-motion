"""Write SHA-256 checksums for every study artifact except the manifest itself."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"MANIFEST.json"

def main():
    rows=[]
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path==DEST or "__pycache__" in path.parts:
            continue
        h=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):
                h.update(chunk)
        rows.append({"path":path.relative_to(ROOT).as_posix(),"bytes":path.stat().st_size,
                     "sha256":h.hexdigest()})
    manifest={"study":"V17C.18B-19","algorithm":"SHA-256",
              "scope":"All regular files under this study directory except MANIFEST.json and __pycache__.",
              "file_count":len(rows),"files":rows}
    DEST.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(f"wrote {DEST} with {len(rows)} entries")

if __name__=="__main__":main()
