#!/usr/bin/env python3

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, simpledialog

import imageio.v2 as imageio

try:
    import imageio_ffmpeg  # noqa: F401  # ensure ffmpeg plugin is available
except ImportError:  # pragma: no cover - handled at runtime
    imageio_ffmpeg = None


def select_base_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select base folder for penetration results")
    root.update()
    root.destroy()
    if not folder:
        print("No folder selected. Exiting.")
        return None
    return Path(folder)


def gather_overview_images(base_folder: Path):
    penetration_dir = base_folder / "penetration_results"
    if not penetration_dir.exists():
        raise FileNotFoundError(f"No 'penetration_results' directory found in {base_folder}")

    image_paths = []
    for subdir in sorted(p for p in penetration_dir.iterdir() if p.is_dir()):
        matches = sorted(subdir.glob("condition_*_overview.png"))
        if not matches:
            print(f"Warning: No overview image found in {subdir}")
            continue
        image_paths.append(matches[0])

    if not image_paths:
        raise FileNotFoundError("No overview images found under penetration_results.")

    return image_paths


def get_fps(default: float = 10.0):
    root = tk.Tk()
    root.withdraw()
    fps = simpledialog.askfloat("FPS", "Frames per second:", initialvalue=default, minvalue=0.1)
    root.update()
    root.destroy()
    if fps is None:
        return default
    return fps


def build_video(image_paths, output_path: Path, fps: float):
    if imageio_ffmpeg is None:
        raise RuntimeError(
            "imageio-ffmpeg is required for MP4 output. Install with 'pip install imageio-ffmpeg'."
        )

    first_frame = imageio.imread(image_paths[0])
    height, width = first_frame.shape[:2]

    with imageio.get_writer(
        str(output_path),
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        macro_block_size=None,
    ) as writer:
        writer.append_data(first_frame)
        for image_path in image_paths[1:]:
            frame = imageio.imread(image_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame size mismatch for {image_path}. Expected {width}x{height}."
                )
            writer.append_data(frame)


def main():
    base_folder = select_base_folder()
    if base_folder is None:
        return

    try:
        images = gather_overview_images(base_folder)
    except FileNotFoundError as exc:
        print(exc)
        return

    fps = get_fps()
    output_name = f"{base_folder.name}_overview.mp4"
    output_path = base_folder / output_name

    try:
        build_video(images, output_path, fps)
    except Exception as exc:
        print(f"Failed to build video: {exc}")
        return

    print(f"Video saved to {output_path} ({len(images)} frames at {fps} fps).")


if __name__ == "__main__":
    main()
