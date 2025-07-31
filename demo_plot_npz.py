from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from mie_postprocessing.functions_videos import *
from mie_postprocessing.cone_angle import plot_angle_signal_density



def plot_npz(path: Path) -> None:
    data = np.load(path)
    print(f"Loaded {path}")
    if 'average_segment' in data:
        # plt.figure()
        # plt.imshow(data['average_segment'], aspect='auto', origin='lower', cmap='gray')
        # plt.title('Average Segment')
        # plt.colorbar()
        play_video_cv2(data['average_segment'], intv=17)
        
    if 'ssim' in data:
        plt.figure()
        plt.imshow(data['ssim'], aspect='auto', origin='lower', cmap='viridis')
        plt.title('SSIM Matrix')
        plt.colorbar()

    for key in data.files:
        if key.endswith('_area'):
            plt.figure()
            plt.plot(data[key])
            plt.title(key)
            plt.xlabel('Frame')
            plt.ylabel('Area (pixels)')
        elif key.endswith('_signal'):
            '''
            plt.figure()
            plt.imshow(data[key], aspect='auto', origin='lower', cmap='viridis')
            plt.title(key)
            plt.colorbar()
            '''
            # size = data[key].shape
            signal_density_bins = np.linspace(0, 1, 180) # Example bins, adjust as needed
            plot_angle_signal_density(signal_density_bins, data[key])
        elif key.endswith('_time_distance_intensity'):
            plt.figure()
            plt.imshow(data[key], aspect='auto', origin='lower', cmap='viridis')
            plt.title(key)
            plt.colorbar()
    plt.show()


def select_folder() -> Path | None:
    """Open a folder selection dialog starting in the script directory."""
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(
        title='Select Folder Containing .npz Files',
        initialdir=Path(__file__).resolve().parent
    )
    root.destroy()
    return Path(folder) if folder else None


def main() -> None:
    folder = select_folder()
    if folder is None:
        return

    for file in sorted(folder.glob('*.npz')):
        # plot_npz(file)
        data = np.load(file)
        for key in data.files:
            match key:
            
                case _ if key.endswith('average_segment'):
                    # play_video_cv2(data[key], intv=17)
                    video = data[key]
                    bw = np.zeros(video.shape)
                    for i in range(0, video.shape[0]):
                        bw[i], _ = triangle_binarize(video[i])
                    play_videos_side_by_side((video, bw))
                case _:
                    continue

            '''
            case _ if key.endswith('_signal'):
                angle = data[key]
                frames, bins = angle.shape
                half = round(bins/2)
                ang = np.concatenate((angle[:,half:], angle[:,0:half]), axis=1)
            
                bw, thres = triangle_binarize(ang)
                cone_angle_per_frame = bw.sum(axis=1)/bins
            case _ if key.endswith('_time_distance_intensity'):
                penetration_2d = data[key]
                bw, thres = triangle_binarize(penetration_2d)
                plt.imshow(bw, origin="lower")
            '''




                

            


if __name__ == '__main__':
    main()