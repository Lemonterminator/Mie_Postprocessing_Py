import numpy as np
import pycine.file as cine

class CineReader:
    """Utility to open a Phantom ``.cine`` file and read frames on demand."""

    def __init__(self):
        self.path = None
        self.frame_offsets = []
        self.frame_count = 0
        self.width = 0
        self.height = 0

    def load(self, path):
        """Load header information from ``path``."""
        header = cine.read_header(path)
        self.path = path
        self.frame_offsets = header['pImage']
        self.frame_count = len(self.frame_offsets)
        self.width = header['bitmapinfoheader'].biWidth
        self.height = header['bitmapinfoheader'].biHeight

    def read_frame(self, idx):
        """Return frame ``idx`` as a ``numpy.ndarray``."""
        if self.path is None:
            raise RuntimeError('No video loaded')
        if not (0 <= idx < self.frame_count):
            raise IndexError('Frame index out of range')
        offset = self.frame_offsets[idx]
        with open(self.path, 'rb') as f:
            f.seek(offset)
            f.read(8)  # skip per-frame header
            data = np.fromfile(f, dtype=np.uint16,
                               count=self.width * self.height)
        frame = data.reshape(self.height, self.width)
        return np.flipud(frame)
