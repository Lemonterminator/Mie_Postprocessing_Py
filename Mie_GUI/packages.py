import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed

import pycine.file as cine  # Ensure the pycine package is installed
from scipy.ndimage import median_filter as ndi_median_filter
from scipy.ndimage import generic_filter, binary_opening, binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import disk
import gc

