# Thesis

This folder contains the LaTeX thesis sources, presentation slides, generated PDFs, and supporting assets.

## Main files

- `thesis.tex` / `thesis.pdf`: English thesis source and compiled output.
- `thesis_zh.tex` / `thesis_zh.pdf`: Chinese thesis source and compiled output.
- `thesisreferences.bib`: Shared bibliography database.
- `aaltothesis.cls`: Aalto thesis document class used by the thesis files.
- `intro_mice_slides_en.tex` / `intro_mice_slides_en.pdf`: English slide deck.
- `intro_mice_slides_zh.tex` / `intro_mice_slides_zh.pdf`: Chinese slide deck.
- `mie_video_to_1d_dataflow.tex` / `mie_video_to_1d_dataflow.pdf`: Dataflow document.

## Folders

- `images/`: Figures referenced from the thesis sources.
- `logos/`: Aalto logo assets required by the thesis template.
- `others/`: Reference/template PDFs from the thesis template package.
- `build_artifacts/`: LaTeX-generated auxiliary files such as `.aux`, `.log`, `.toc`, `.bbl`, `.xdv`, and related files.

Keep source files and bibliography files in this directory unless the relative paths in the `.tex` files are updated.

## Environment and compilation

See [LATEX_ENVIRONMENT.md](LATEX_ENVIRONMENT.md) for the full environment specification,
required MiKTeX version, VS Code setup, and known issues.

**Quick start (Windows command line):**

```bat
cd C:\Users\Jiang\Documents\Mie_Postprocessing_Py\Thesis
build_thesis.cmd
```

**Quick start (WSL/bash from this repo):**

```bash
./Thesis/build_thesis_wsl.sh
```

**VS Code:** Save the file — the `pdflatex -> biber -> pdflatex x2` recipe runs automatically.
