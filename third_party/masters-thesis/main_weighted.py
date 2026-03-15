"""Compatibility wrapper for the packaged Masters-thesis CLI."""

from OSCC_postprocessing.masters_thesis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
