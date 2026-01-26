import json
import math
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes

class VideoDisplayer:
    def __init__(self, items, layout=None, figsize=None, default_cmap="gray"):
        if not isinstance(items, (list, tuple)):
            raise TypeError("items must be a list or tuple.")
        if len(items) == 0:
            raise ValueError("items cannot be empty.")

        self.items = list(items)
        self.rows, self.cols = self._normalize_layout(layout, len(self.items))
        if figsize is None:
            figsize = (5 * self.cols, 5 * self.rows)
        self.fig, axes = plt.subplots(self.rows, self.cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel().tolist()

        self.video_entries = []
        self.anim = None

        for index, item in enumerate(self.items):
            ax = axes[index]
            self._setup_item(ax, item, default_cmap)

        for ax in axes[len(self.items):]:
            ax.set_visible(False)

    def _update_frame(self, i):
        artists = []
        for entry in self.video_entries:
            data = entry["data"]
            frame_index = i % data.shape[0]
            entry["im"].set_array(data[frame_index])
            artists.append(entry["im"])
        return artists

    def _normalize_layout(self, layout, count):
        if layout is None:
            cols = math.ceil(math.sqrt(count))
            rows = math.ceil(count / cols)
            return rows, cols
        if isinstance(layout, str):
            if "x" in layout:
                parts = layout.lower().split("x")
                layout = [int(parts[0]), int(parts[1])]
            else:
                raise ValueError("layout string must be in 'rowsxcols' format.")
        if not isinstance(layout, (list, tuple)) or len(layout) != 2:
            raise ValueError("layout must be a (rows, cols) tuple or list.")
        rows, cols = int(layout[0]), int(layout[1])
        if rows <= 0 or cols <= 0:
            raise ValueError("layout rows and cols must be positive.")
        if rows * cols < count:
            raise ValueError("layout is too small for the number of items.")
        return rows, cols

    def _setup_item(self, ax, item, default_cmap):
        if isinstance(item, np.ndarray) and item.ndim == 3:
            im = ax.imshow(item[0], cmap=default_cmap, animated=True)
            ax.set_xticks([])
            ax.set_yticks([])
            self.video_entries.append({"data": item, "im": im})
            return
        if callable(item):
            item(ax)
            return
        if isinstance(item, Axes):
            self._copy_axes_content(item, ax)
            return
        raise TypeError("Unsupported item type. Use a 3D numpy array, a callable, or a Matplotlib Axes.")

    def _copy_axes_content(self, source_ax, target_ax):
        for line in source_ax.get_lines():
            target_ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize()
            )
        for image in source_ax.images:
            target_ax.imshow(
                image.get_array(),
                cmap=image.get_cmap(),
                interpolation=image.get_interpolation()
            )
        target_ax.set_title(source_ax.get_title())
        target_ax.set_xlabel(source_ax.get_xlabel())
        target_ax.set_ylabel(source_ax.get_ylabel())
        target_ax.set_xlim(source_ax.get_xlim())
        target_ax.set_ylim(source_ax.get_ylim())

    def _locate_ffmpeg(self):
        """Return a usable ffmpeg executable path or None if none found."""
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError:
            return None

        try:
            return get_ffmpeg_exe()
        except Exception:
            return None

    def show(self):
        """Display the animation."""
        if not self.video_entries:
            plt.show()
            return
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self._max_frames(),
            interval=50,  # Delay between frames in ms
            blit=True,
            repeat=True
        )
        plt.show()

    def save_video(self, filename="output.mp4", fps=20):
        """Save the animation to a video file."""
        if not self.video_entries:
            print("No video items to save.")
            return
        print(f"Saving video to {filename}...")
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self._max_frames(),
            interval=1000/fps,
            blit=True
        )
        # You need ffmpeg installed for this to work
        try:
            ffmpeg_path = self._locate_ffmpeg()
            if not ffmpeg_path:
                raise EnvironmentError(
                    "ffmpeg executable not found on PATH and imageio-ffmpeg is not installed."
                )

            plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
            writer = animation.FFMpegWriter(fps=fps)
            self.anim.save(filename, writer=writer)
            print("Video saved successfully.")
        except Exception as e:
            print(f"Error saving video: {e}")
            print("Please ensure that ffmpeg is installed and in your system's PATH.")

    def _max_frames(self):
        return max(entry["data"].shape[0] for entry in self.video_entries)

def create_dummy_data():
    # Dummy static data
    static_x = np.linspace(0, 2 * np.pi, 100)
    static_y = np.sin(static_x)

    # Dummy grayscale video data (100 frames, 64x64)
    num_frames = 100
    height, width = 64, 64
    video_data = np.random.rand(num_frames, height, width)

    return static_x, static_y, video_data

def build_static_plot(static_x, static_y):
    def _plot(ax):
        ax.plot(static_x, static_y)
        ax.set_title('Static Plot')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
    return _plot

def main():
    config_path = Path(__file__).with_suffix(".json")
    default_config = {
        "debug": False,
        "output": "videos/static_and_video.mp4",
        "fps": 30,
        "layout": [1, 2]
    }
    if not config_path.exists():
        config_path.write_text(
            json.dumps(default_config, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"Config file created at {config_path}. Edit it and run again.")
        return

    with config_path.open("r", encoding="utf-8") as file_handle:
        config = json.load(file_handle)

    debug = bool(config.get("debug", default_config["debug"]))
    output = str(config.get("output", default_config["output"]))
    fps = int(config.get("fps", default_config["fps"]))
    layout = config.get("layout", default_config["layout"])
    
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path(__file__).parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    static_x, static_y, video = create_dummy_data()

    # video1 = np.load(r"G:\MeOH_test\NFL\Processed_Results\Rotated_Videos\T56_NFL_Cam_5_experiement.npz")
    # video2 = np.load(r"G:\MeOH_test\Schlieren\Processed_Results\Rotated_Videos\T56_Schlieren Cam_5.npz")
    items = [
        build_static_plot(static_x, static_y),
        video,
        build_static_plot(static_x, static_y),
        video
    ]

    displayer = VideoDisplayer(items, layout=layout)

    if debug:
        displayer.show()
    else:
        displayer.save_video(str(output_path), fps=fps)


if __name__ == '__main__':
    main()
