"""
Compatibility shim for the merged penetration data pipeline.

The legacy ``dataframe_builder`` entry point now simply forwards to
``prepare_penetration_data`` so that existing scripts/imports continue to
work without modification.
"""

from __future__ import annotations

try:  # pragma: no cover
    from .prepare_penetration_data import (
        build_dataframe_for_directory,
        load_test_matrix,
        main as _pipeline_main,
    )
except ImportError:  # pragma: no cover
    from prepare_penetration_data import (  # type: ignore
        build_dataframe_for_directory,
        load_test_matrix,
        main as _pipeline_main,
    )

__all__ = ["build_dataframe_for_directory", "load_test_matrix", "main"]


def main() -> None:
    """Invoke the unified penetration data pipeline."""
    _pipeline_main()


if __name__ == "__main__":
    main()
