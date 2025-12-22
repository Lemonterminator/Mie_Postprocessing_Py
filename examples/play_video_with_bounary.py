import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

video_npz = np.load(r"G:\MeOH_test\Mie\Processed_Results\Postprocessed_Data\T2_Mie Camera_1.cine_foreground.npz")
if isinstance(video_npz, np.lib.npyio.NpzFile):
    video = video_npz[video_npz.files[0]]
else:
    video = video_npz

boundary = pd.read_csv(
    r"G:\MeOH_test\Mie\Processed_Results\Postprocessed_Data\T2_Mie Camera_1.cine_boundary_points.csv"
)

# Pre-group boundary points by frame for fast lookup during playback.
boundary_by_frame = {
    int(frame): grp[["x", "y"]].to_numpy()
    for frame, grp in boundary.groupby("frame", sort=True)
}

plt.ion()
fig, ax = plt.subplots()
im = ax.imshow(video[0], cmap="gray", vmin=video.min(), vmax=video.max())
scat = ax.scatter([], [], s=5, c="r")
ax.set_title("Frame 0")
ax.set_xlim(0, video.shape[2] - 1)
ax.set_ylim(video.shape[1] - 1, 0)

for frame_idx in range(video.shape[0]):
    im.set_data(video[frame_idx])
    points = boundary_by_frame.get(frame_idx)
    if points is not None:
        points[:, 0] += video.shape[1]//2 # Adjust x-coordinates if needed
    if points is None:
        scat.set_offsets(np.empty((0, 2)))
    else:
        scat.set_offsets(points)
    ax.set_title(f"Frame {frame_idx}")
    plt.pause(0.001)

plt.ioff()
plt.show()

