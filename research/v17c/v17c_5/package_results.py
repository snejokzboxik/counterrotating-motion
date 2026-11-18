"""Create and verify a portable ZIP containing this study's code and frozen results."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT/"V17C_5_results.zip"
MANIFEST = ROOT/"MANIFEST.json"
HASH = ROOT/"V17C_5_results.zip.sha256"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    paths = sorted(p for p in ROOT.rglob("*") if p.is_file() and
                   p not in (ARCHIVE, MANIFEST, HASH) and "__pycache__" not in p.parts and
                   p.suffix.lower() in {".py",".npz",".json",".md",".png",".txt",".csv",".zip"})
    entries = {p.relative_to(ROOT).as_posix():{"bytes":p.stat().st_size,"sha256":sha(p.read_bytes())} for p in paths}
    MANIFEST.write_text(json.dumps({"files": entries},indent=2)+"\n",encoding="utf-8")
    with ZipFile(ARCHIVE,"w",compression=ZIP_DEFLATED,compresslevel=3) as package:
        for path in [*paths,MANIFEST]:
            package.write(path,"v17c_5/"+path.relative_to(ROOT).as_posix())
    with ZipFile(ARCHIVE) as package:
        assert package.testzip() is None
        assert len(package.namelist()) == len(entries)+1
        for name, entry in entries.items():
            assert sha(package.read("v17c_5/"+name)) == entry["sha256"], name
    HASH.write_text(sha(ARCHIVE.read_bytes())+"  "+ARCHIVE.name+"\n",encoding="ascii")
    print(f"{len(entries)} payload files; {ARCHIVE.stat().st_size} ZIP bytes; {HASH.read_text().strip()}")


if __name__ == "__main__":
    main()
