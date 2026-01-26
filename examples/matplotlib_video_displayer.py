import argparse
import shutil

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class VideoDisplayer:
    def __init__(self, static_data_x, static_data_y, video_data):
        self.static_data_x = static_data_x
        self.static_data_y = static_data_y
        self.video_data = video_data
        self.num_frames = video_data.shape[0]

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Setup the static plot
        self.ax1.plot(self.static_data_x, self.static_data_y)
        self.ax1.set_title('Static Plot')
        self.ax1.set_xlabel('X-axis')
        self.ax1.set_ylabel('Y-axis')

        # Setup the video plot
        self.im = self.ax2.imshow(self.video_data[0], cmap='gray', animated=True)
        self.ax2.set_title('Grayscale Video')
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])

        self.anim = None

    def _update_frame(self, i):
        self.im.set_array(self.video_data[i])
        return self.im,

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
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.num_frames,
            interval=50,  # Delay between frames in ms
            blit=True,
            repeat=True
        )
        plt.show()

    def save_video(self, filename="output.mp4", fps=20):
        """Save the animation to a video file."""
        print(f"Saving video to {filename}...")
        self.anim = animation.FuncAnimation(
            self.fig,
            self._update_frame,
            frames=self.num_frames,
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

def create_dummy_data():
    # Dummy static data
    static_x = np.linspace(0, 2 * np.pi, 100)
    static_y = np.sin(static_x)

    # Dummy grayscale video data (100 frames, 64x64)
    num_frames = 100
    height, width = 64, 64
    video_data = np.random.rand(num_frames, height, width)

    return static_x, static_y, video_data

def main():
    parser = argparse.ArgumentParser(
        description="Render the static plot alongside a sample grayscale animation."
    )
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Display the animation live instead of saving it."
    )
    parser.add_argument(
        "--output",
        "-o",
        default="static_and_video.mp4",
        help="Filename for the saved animation."
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frame rate to use when writing the video."
    )

    args = parser.parse_args()
    
    static_x, static_y, video = create_dummy_data()
    displayer = VideoDisplayer(static_x, static_y, video)

    if args.debug:
        displayer.show()
    else:
        displayer.save_video(args.output, fps=args.fps)


if __name__ == '__main__':
    main()
