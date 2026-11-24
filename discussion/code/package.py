from pathlib import Path
import zipfile
r=Path('discussion')
with zipfile.ZipFile('Counterrotating_Motion_Semyon_Discussion_RU.zip','w',zipfile.ZIP_DEFLATED) as z:
 z.write(r/'main.tex','main.tex')
 for p in sorted((r/'figures').glob('*.pdf')):z.write(p,'figures/'+p.name)
with zipfile.ZipFile('Counterrotating_Motion_Semyon_Discussion_RU.zip') as z:assert z.testzip() is None
