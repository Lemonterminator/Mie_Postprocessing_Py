"""CLI entrypoint for the packaged Masters-thesis workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import process_videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oscc-masters-thesis",
        description="Run the packaged OSCC Masters-thesis spray-processing workflow.",
    )
    parser.add_argument("--video", action="append", default=[], help="Path to a video file. Repeatable.")
    parser.add_argument("--videos", nargs="+", default=[], help="One or more video paths.")
    parser.add_argument("--output-dir", help="Directory for CSV, overlay video, and generated mask output.")
    parser.add_argument("--config-path", help="Optional config.json path. Defaults to the video-adjacent config.json.")
    parser.add_argument("--mask-path", help="Optional existing mask path. If missing and interactive, one is created.")
    parser.add_argument(
        "--use-gpu",
        choices=["auto", "always", "never"],
        default="auto",
        help="GPU usage policy for nozzle-alignment rotation.",
    )
    parser.add_argument(
        "--interactive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow GUI file selection and mask/background editing when needed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    videos = [Path(p) for p in [*args.video, *args.videos]]
    process_videos(
        videos if videos else None,
        output_dir=args.output_dir,
        config_path=args.config_path,
        mask_path=args.mask_path,
        use_gpu=args.use_gpu,
        interactive=bool(args.interactive),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
