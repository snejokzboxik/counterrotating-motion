# PDF and Overleaf build status

- Target class: `revtex4-2`, AIP `cha` substyle, reprint, with UTF-8/T2A and babel for the requested Russian working manuscript.
- The manuscript contains a Chaos-style lead paragraph and English scientific figure captions. The author list remains a project placeholder and needs the real accountable authors before journal submission.
- Current local environment lacks `pdflatex`; pdfLaTeX validation ran remotely in GitHub Actions. Tectonic was not substituted.
- Official AIP Chaos policy says submissions must be clearly written in English and requires a nonspecialist lead paragraph. Thus this Russian file is an internal Russian working article in the requested AIP layout, not an English-ready journal submission. Source: [Chaos editorial policies](https://pubs.aip.org/aip/cha/pages/policies) and [AIP Author Instructions](https://pubs.aip.org/pages/author-instructions/).
- Clean extraction of `Counterrotating_Motion_Chaos_RU_Overleaf.zip` compiled twice with pdfLaTeX successfully in GitHub Actions run [45458469061](https://github.com/snejokzboxik/counterrotating-motion/actions/runs/45458469061). The workflow uploaded the PDF and source ZIP as artifact `counterrotating-motion-chaos-russian-pdf-and-overleaf` (artifact ID `14950263409`; GitHub artifact retention expires 2027-02-22).
- The Overleaf ZIP has `main.tex` at its root and contains the three local PDF figure assets. This is a successful build check, not a journal submission or author approval.
- The earlier `counterrotating-motion-overleaf.zip` remains a historical draft and is not silently overwritten.
