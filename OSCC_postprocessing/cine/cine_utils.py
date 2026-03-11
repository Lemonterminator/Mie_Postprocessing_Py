"""On-demand Phantom Cine reading helpers used by the GUI stack."""

import numpy as np
import pycine.file as cine

_CINE_FRAME_HEADER_BYTES = 8

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
            # ``pImage`` points to the per-frame block, which begins with a
            # small frame header before the pixel payload.
            header = f.read(_CINE_FRAME_HEADER_BYTES)
            if len(header) != _CINE_FRAME_HEADER_BYTES:
                raise ValueError(
                    f'Failed to read {_CINE_FRAME_HEADER_BYTES}-byte frame header '
                    f'for frame {idx}.'
                )
            data = np.fromfile(
                f,
                dtype=np.uint16,
                count=self.width * self.height,
            )
        if data.size != self.width * self.height:
            raise ValueError(
                f'Incomplete frame payload for frame {idx}: expected '
                f'{self.width * self.height} pixels, got {data.size}.'
            )
        frame = data.reshape(self.height, self.width)
        # return np.flipud(frame)
        return frame
