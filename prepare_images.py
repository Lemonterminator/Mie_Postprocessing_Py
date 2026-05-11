import os
import shutil
import cv2

src_tif_dir = r'C:\Users\Jiang\Documents\Mie_Postprocessing_Py\images\Phantom_Snapshots'
src_plot = r'C:\Users\Jiang\Documents\Mie_Postprocessing_Py\images\penetration_compare\BC20241014_HZ_Nozzle3\T16\T16(5).png'
dest_dir = r'C:\Users\Jiang\Documents\Mie_Postprocessing_Py\Thesis\images\failure_mode_nozzle3'

os.makedirs(dest_dir, exist_ok=True)

# Convert and copy TIFs
for i in range(3, 7):
    filename = f'Cam30660-FromFile_000{i}.tif'
    tif_path = os.path.join(src_tif_dir, filename)
    png_path = os.path.join(dest_dir, f'phantom_t{i}.png')
    if os.path.exists(tif_path):
        img = cv2.imread(tif_path)
        cv2.imwrite(png_path, img)

# Copy plot
shutil.copy(src_plot, os.path.join(dest_dir, 'nozzle3_t16_plot.png'))

print('Images successfully prepared in', dest_dir)
