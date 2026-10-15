# Slides

`VeriReason_Proposed_Slide.tex` is the Beamer source of the research idea proposal (29 slides);
`VeriReason_Proposed_Slide.pdf` is the compiled deck. It was rebuilt on 15 Oct 2026 from the PDF of
5 Oct 2026 because the original source was not available: text, structure and images (logos, background)
were recovered from that PDF, so small layout differences from the original are expected.

Changes against the 5 Oct version are listed in `docs/SLIDE_CORRECTIONS.md`. Dates follow the Project
Timeline (Week 0 from 10 Oct 2026; pilot 19 Oct to 1 Nov; go/no-go meeting Mon 2 Nov).

Build (TeX Live with `beamer`, `lmodern`, `pifont`, `tikz`, `colortbl`; run twice for the page numbers):

```bash
cd slides && pdflatex VeriReason_Proposed_Slide.tex && pdflatex VeriReason_Proposed_Slide.tex
```
