# LaTeX Environment Setup

This document records the exact LaTeX environment used to compile `thesis.tex` and
`thesis_zh.tex`, and the steps needed to reproduce it on a new machine.

---

## Environment Summary

| Item | Value |
|------|-------|
| TeX Distribution | MiKTeX 26.2 (Windows) |
| LaTeX Kernel | LaTeX 2024-11-01 |
| Bibliography Engine | Biber 2.21 |
| Default Compiler | pdfLaTeX |
| Chinese Support | CJKutf8 (pdfLaTeX path) |
| Template | Aalto Thesis Class v4.10 (`aaltothesis.cls`) |
| PDF Standard | PDF/A-2b (`a-2b` option) |

---

## Why This Environment

Overleaf uses **TeX Live** (latest release, currently TeX Live 2025) on Ubuntu Linux.
MiKTeX 26.2 is the Windows equivalent: it carries the same LaTeX 2024-11-01 kernel and
CTAN package versions that Overleaf uses, so compilation output is visually identical.

The critical kernel version is `2024-11-01`. The earlier MiKTeX 24.1 (January 2024)
only had `2023-11-01`, which caused `figureversions.sty` (loaded by the font stack) to
fail with "undefined control sequence" errors. MiKTeX 26.2 fixed this.

---

## Reproducing on a New Windows Machine

### Step 1 — Install MiKTeX

Download the installer from https://miktex.org/download and install for the current user
(recommended) or all users.

After installation, open **MiKTeX Console** and run:

```
Update  ->  Update now
```

or from a terminal:

```
miktex packages update
```

This upgrades MiKTeX to the latest release (26.x or newer) and pulls in the
`2024-11-01` LaTeX kernel. Packages are downloaded on demand during the first
compilation.

### Step 2 — Verify the kernel version

```
pdflatex --version
```

The output should contain `MiKTeX-pdfTeX 4.26` or later. The LaTeX kernel version is
printed during the first compilation of a document.

### Step 3 — Install VS Code and LaTeX Workshop

1. Install [VS Code](https://code.visualstudio.com/)
2. Install the **LaTeX Workshop** extension (`james-yu.latex-workshop`)
3. The workspace `settings.json` in this repo already configures the build recipe
   (see `.vscode/settings.json`)

The active recipe is **`pdflatex -> biber -> pdflatex x2`**, which runs:

```
pdflatex  →  biber  →  pdflatex  →  pdflatex
```

This is triggered automatically on file save (`latex-workshop.latex.autoBuild.run: onSave`).

### Step 4 — Verify a clean compile

From the `Thesis/` directory, run the four steps manually to confirm everything works:

```bash
cd Thesis/

# English thesis
pdflatex -interaction=nonstopmode thesis.tex
biber thesis
pdflatex -interaction=nonstopmode thesis.tex
pdflatex -interaction=nonstopmode thesis.tex

# Chinese thesis
pdflatex -interaction=nonstopmode thesis_zh.tex
biber thesis_zh
pdflatex -interaction=nonstopmode thesis_zh.tex
pdflatex -interaction=nonstopmode thesis_zh.tex
```

Expected: `thesis.pdf` (≈48 pages) and `thesis_zh.pdf` (≈41 pages) with numbered
citations and a populated reference list.

> **Note on missing images**  
> Figures referenced as `../images/` or `images/` are stored outside this repository
> (large binary files). Their absence produces `pdftex.def Error: File not found`
> warnings but does not prevent PDF generation — placeholder boxes appear instead.
> Copy the `images/` folder from Overleaf or the original source to restore figures.

---

## Template Files

The Aalto thesis template (v4.10, released 2025-06-30) is stored locally in this
repository:

| File / Folder | Purpose |
|---------------|---------|
| `aaltothesis.cls` | Full Aalto thesis class (v4.10). Provides cover page, abstract page, PDF/A-2b metadata, and font setup via `\setupthesisfonts`. |
| `logos/` | Aalto school logo PDFs required by the cover page. |
| `Master_thesis_template/` | Original unmodified template from Aalto for reference. |

The class requires **MiKTeX 26.x / TeX Live 2024+** because it loads `figureversions.sty`
which depends on the `2024-11-01` LaTeX kernel.

---

## Packages Explicitly Loaded by the Thesis Files

These are packages loaded directly in `thesis.tex` / `thesis_zh.tex` (beyond what
`aaltothesis.cls` already loads internally):

| Package | Purpose |
|---------|---------|
| `graphicx` | Figure inclusion (`\includegraphics`) |
| `array`, `booktabs` | Enhanced tables |
| `siunitx` | SI unit formatting (`\SI{2000}{bar}`) |
| `doclicense` | Creative Commons license icon on copyright page |
| `biblatex` (style=numeric) | Bibliography management |
| `CJKutf8` | Chinese character support in pdfLaTeX (`thesis_zh.tex` only) |
| `iftex`, `fontspec`, `xeCJK` | Alternative Chinese support when compiling with XeLaTeX (`thesis_zh.tex` only) |

> `amsmath`, `amssymb`, `hyperref`, `xurl`, `fancyhdr`, `geometry`, `microtype`,
> `newtxtext`, `newtxmath`, `babel` are all loaded by `aaltothesis.cls` internally —
> do **not** load them again in the thesis files.

---

## Known Issues and Fixes Applied

### 1. `figureversions.sty` undefined control sequence (MiKTeX 24.1)
**Cause**: MiKTeX 24.1 bundles LaTeX kernel `2023-11-01`; the template requires `2024-11-01`.  
**Fix**: Updated MiKTeX to 26.2 via `miktex packages update`.

### 2. `\thesisabstract{}` — "File ended while scanning"
**Cause**: Unescaped `%` characters inside a `\newcommand` argument are treated as
TeX comment characters, discarding the rest of the line including the closing `}`.  
**Fix**: All `%` inside `\thesisabstract{...}` are escaped as `\%`.

### 3. Greek `σ` in pdfLaTeX text mode
**Cause**: Unicode σ (U+03C3) is not available in T1 font encoding text mode.  
**Fix**: Replaced with `$\sigma$` (math mode).

### 4. `abstractpage` language error in `thesis_zh.tex`
**Cause**: `\begin{abstractpage}` without an explicit language argument passes the macro
`\MainLang` literally to `\ifstrequal`, which does not match `english`.  
**Fix**: Use `\begin{abstractpage}[english]` explicitly.

### 5. Citations display as `[key]` instead of `[1]`
**Cause**: `biber` was not run after the first `pdflatex` pass.  
**Fix**: Always run the full four-step sequence (pdflatex → biber → pdflatex × 2).
The VS Code recipe handles this automatically.

### One-command build in this workspace

From Windows `cmd.exe`:

```bat
cd C:\Users\Jiang\Documents\Mie_Postprocessing_Py\Thesis
build_thesis.cmd
```

From WSL/bash:

```bash
./Thesis/build_thesis_wsl.sh
```

Both scripts run the same four-step sequence for `thesis.tex` and `thesis_zh.tex`:

```text
pdflatex -> biber -> pdflatex -> pdflatex
```

The thesis files load the local Aalto template directly through
`\documentclass[..., elec, a-2b, online]{aaltothesis}`. Keep `aaltothesis.cls`,
`pdfa.xmpi`, the `.xmpdata` files, and `logos/` in `Thesis/` so both PDFs keep the
Aalto cover, abstract-page styling, and PDF/A metadata.
