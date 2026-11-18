"""Build a content-hash manifest and a portable ZIP of code, audit, and results."""
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

here=Path(__file__).resolve().parent
repo=here.parents[1]
source=repo.parent/'V17C_all_four_calculations (2).zip'
bundle=repo/'V17C_1_research_bundle.zip'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

files=sorted(p for p in here.rglob('*') if p.is_file() and '__pycache__' not in p.parts and
             p.name not in {'MANIFEST.json','V17C_1_research_bundle.zip'})
manifest={'format_version':1,'archive_source':str(source),'archive_sha256':sha(source),
          'python':sys.version.split()[0], 'files':{str(p.relative_to(repo)).replace('\\','/'):
          {'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}}
(here/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with ZipFile(bundle,'w',ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[here/'MANIFEST.json']:
        z.write(p,str(p.relative_to(repo)).replace('\\','/'))
print(bundle,'files',len(files)+1,'bytes',bundle.stat().st_size)
