# PDF and Overleaf build status

- Target class: `revtex4-2`, AIP `cha` substyle, reprint, with UTF-8/T2A and babel for the requested Russian working manuscript.
- The manuscript contains a Chaos-style lead paragraph and English scientific figure captions. The author list remains a project placeholder and needs the real accountable authors before journal submission.
- Current environment discovery: `tectonic.exe` exists; `pdflatex`, `latexmk`, MiKTeX and TeX Live commands are absent. Tectonic was not used because the directive explicitly requires pdfLaTeX.
- Official AIP Chaos policy says submissions must be clearly written in English and requires a nonspecialist lead paragraph. Thus this Russian file is an internal Russian working article in the requested AIP layout, not an English-ready journal submission. Source: [Chaos editorial policies](https://pubs.aip.org/aip/cha/pages/policies) and [AIP Author Instructions](https://pubs.aip.org/pages/author-instructions/).
- `Counterrotating_Motion_Chaos_RU_Overleaf.zip` is packaged with `main.tex` at the archive root, local PDF figure assets and bibliography. It has not been compiled from a clean extraction because pdfLaTeX is missing; status: `UNVERIFIED_BUILD`.
- No PDF is claimed or published yet. A GitHub Actions workflow now installs pdfLaTeX, builds twice from a clean ZIP extraction and uploads the PDF/ZIP artifact. The workflow has not run until this branch is pushed; its eventual success or failure must be inspected before changing `UNVERIFIED_BUILD`.
- The earlier `counterrotating-motion-overleaf.zip` remains a historical draft and is not silently overwritten.
