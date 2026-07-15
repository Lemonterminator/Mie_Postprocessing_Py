"""Layered evaluation pipeline for the penetration models (thesis Ch5).

Layer 1: ``run_eval.py``   (dataset roots + checkpoints) -> metrics + points
Layer 2: ``make_figures.py`` (Layer-1 run dir)           -> full figure suite
"""
