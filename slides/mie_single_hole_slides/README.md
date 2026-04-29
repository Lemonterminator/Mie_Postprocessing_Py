# mie_single_hole slides

Build the deck from this folder:

```bash
make
```

If `make` is not available, run:

```bash
pdflatex -interaction=nonstopmode slides.tex
pdflatex -interaction=nonstopmode slides.tex
```

The deck is written so it can compile before the generated figures exist. To
replace placeholders with real plots, run
`examples/mie/mie_single_hole_2.0.ipynb`; the added export cells save figures
into `slides/mie_single_hole_slides/figs/`.
