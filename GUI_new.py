"""
Minimal PyQt6 viewer for a single frame from a Phantom `.cine` video.

How it works (high-level)
-------------------------
1) Read frames on-demand from a `.cine` file.
2) Convert one frame to a Qt QImage (`frame_to_qimage`).
3) Put that QImage into a QPixmap and display it in a QGraphicsScene via
   QGraphicsPixmapItem (good for future overlays like masks/paths).
4) Optionally draw a vector overlay (QGraphicsPathItem) on top of the image.
"""
global BITS
BITS = 12


import os
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QColor, QImage, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QLabel,
    QDoubleSpinBox,
)

from OSCC_postprocessing.cine_utils import CineReader


class ImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom = 1.0
        self._zoom_min = 0.05
        self._zoom_max = 20.0

        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def reset_zoom(self) -> None:
        self.resetTransform()
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)

        factor = 1.25 if delta > 0 else 0.8
        new_zoom = self._zoom * factor
        if new_zoom < self._zoom_min:
            factor = self._zoom_min / self._zoom
            new_zoom = self._zoom_min
        elif new_zoom > self._zoom_max:
            factor = self._zoom_max / self._zoom
            new_zoom = self._zoom_max

        self.scale(factor, factor)
        self._zoom = new_zoom
        event.accept()


def frame_to_qimage(frame: np.ndarray) -> QImage:
    """
    Convert a single grayscale NumPy frame into a Qt `QImage`.

    Parameters
    ----------
    frame:
        Expected shape is (height, width) (2D).
        Common dtypes:
        - uint16: typical raw cine data (12-bit stored in 16-bit container)
        - uint8 : already display-ready grayscale

    Returns
    -------
    QImage
        A `QImage` wrapping the array memory. The caller must keep the backing
        array alive for as long as Qt uses the image.

    Notes
    -----
    For speed, this function avoids deep-copying pixel data.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"Expected numpy ndarray, got {type(frame)!r}")
    if frame.ndim != 2:
        raise ValueError(f"Expected 2D grayscale frame, got shape {frame.shape}")

    # Ensure the array is contiguous; QImage expects rows to be laid out in a
    # predictable way. A contiguous array also makes `bytes_per_line` correct.
    frame_c = np.ascontiguousarray(frame)
    height, width = frame_c.shape

    if frame_c.dtype == np.uint16:
        # One row stride in bytes. For uint16, this is typically width * 2.
        bytes_per_line = frame_c.strides[0]
        # Grayscale16 stores 16-bit pixel values (0..65535). For 12-bit cine,
        # values usually occupy a subset of that range; Qt will still display it.
        return QImage(
            frame_c.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale16,
        )

    if frame_c.dtype == np.uint8:
        bytes_per_line = frame_c.strides[0]
        return QImage(
            frame_c.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8,
        )

    # Fallback: convert arbitrary numeric data to an 8-bit display image by
    # normalizing the frame to [0, 255]. This is helpful if you later feed in
    # float frames, background-subtracted frames, etc.
    frame_u8 = frame_c.astype(np.float32)
    vmin = float(np.nanmin(frame_u8))
    vmax = float(np.nanmax(frame_u8))
    if vmax <= vmin:
        frame_u8[:] = 0
    else:
        frame_u8 = (frame_u8 - vmin) * (255.0 / (vmax - vmin))
    frame_u8 = np.clip(frame_u8, 0, 255).astype(np.uint8)

    bytes_per_line = frame_u8.strides[0]
    qimg = QImage(
        frame_u8.data,
        width,
        height,
        bytes_per_line,
        QImage.Format.Format_Grayscale8,
    )
    return qimg.copy()


class FrameViewer(QMainWindow):
    """
    Main window for viewing a single video frame.

    We use a QGraphicsView/QGraphicsScene stack because it makes it easy to:
    - display an image (QGraphicsPixmapItem)
    - add vector overlays (QGraphicsPathItem for masks, outlines, ROIs, etc.)
    - later add interaction (mouse drawing, zoom/pan, selection tools)
    """
    def __init__(self, video_path: str | None = None, frame_number: int = 0):
        super().__init__()
        self.setWindowTitle("Mie Postprocessing - Frame Viewer")

        self.reader = CineReader()
        self.video_path: str | None = None
        self.frame_index = 0
        self.total_frames = 0
        self._frame_buffer: np.ndarray | None = None
        self._frame_u8_raw: np.ndarray | None = None
        self._frame_u8_disp: np.ndarray | None = None

        self.gain = 1.0
        self._gamma = 1.0
        self._gain_gamma_lut: np.ndarray | None = None
        self._lut_gain = 1.0
        self._lut_gamma = 1.0

        # QGraphicsView is a QWidget: this must only be constructed *after*
        # a QApplication exists (we ensure that by creating the window in main()).
        self.view = ImageView(self)
        self.view.setRenderHints(self.view.renderHints())
        self.view.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setCentralWidget(self.view)

        self._build_toolbar()

        # The scene holds display "items" (pixmap, vector paths, etc.).
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        # The image itself: setPixmap(...) updates what you see.
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # Calibration point
        self.calibration_point = QGraphicsEllipseItem()
        self.calibration_point.setPen(QPen(QColor("yellow"), 1))

        # Overlay item for masks/annotations. Right now it's an empty path,
        # but later you can set it to any QPainterPath (polyline, polygon, etc.).
        self.mask_item = QGraphicsPathItem()
        self.mask_item.setPen(QPen(QColor("red"), 2))
        self.scene.addItem(self.mask_item)

        self._set_nav_enabled(False)
        if video_path:
            self.load_video(video_path, frame_number)

    def _build_toolbar(self) -> None:
        # Toolbar with Open, Prev, Next buttons.
        tb = QToolBar("Controls", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        # Open video action
        self.open_action = QAction("Load Video", self)
        self.open_action.triggered.connect(self.open_video_dialog)
        tb.addAction(self.open_action)


        # Frame navigation actions, previous and next frames
        tb.addSeparator()

        self.prev_action = QAction("Prev Frame", self)
        self.prev_action.setShortcut("Left")
        self.prev_action.triggered.connect(self.prev_frame)
        tb.addAction(self.prev_action)

        self.next_action = QAction("Next Frame", self)
        self.next_action.setShortcut("Right")
        self.next_action.triggered.connect(self.next_frame)
        tb.addAction(self.next_action)

        # Gamma adjustment placeholder (not implemented)
        tb.addSeparator()
        tb.addWidget(QLabel(" Gamma: ", self))
        
        self.gamma_spin = QDoubleSpinBox(self)
        self.gamma_spin.setRange(0.1, 10.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setDecimals(2)
        self.gamma_spin.setFixedWidth(90)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        tb.addWidget(self.gamma_spin)

        # Gain adjustment placeholder (not implemented)
        tb.addSeparator()
        tb.addWidget(QLabel(" Gain: ", self))
        self.gain_spin = QDoubleSpinBox(self)
        self.gain_spin.setRange(0.0, 10.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setDecimals(2)
        self.gain_spin.setFixedWidth(90)
        self.gain_spin.setValue(1.0)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        tb.addWidget(self.gain_spin)

        # Calibration action placeholder (not implemented)
        self.calibration_action = QAction("Calibration", self)
        self.calibration_action.setShortcut("Right")
        self.calibration_action.triggered.connect(self._on_calibration)
        tb.addAction(self.calibration_action)
        

    def _ensure_gain_gamma_lut(self) -> np.ndarray:
        gain = float(self.gain)
        gamma = float(self._gamma)
        if not np.isfinite(gain):
            gain = 1.0
        if not np.isfinite(gamma) or gamma <= 0.0:
            gamma = 1.0
        if gain < 0.0:
            gain = 0.0

        if (
            self._gain_gamma_lut is not None
            and gain == self._lut_gain
            and gamma == self._lut_gamma
        ):
            return self._gain_gamma_lut

        x = (np.arange(256, dtype=np.float32) * (1.0 / 255.0)) * gain
        np.clip(x, 0.0, 1.0, out=x)
        if gamma != 1.0:
            np.power(x, gamma, out=x)
        lut = np.rint(x * 255.0).astype(np.uint8, copy=False)

        self._gain_gamma_lut = lut
        self._lut_gain = gain
        self._lut_gamma = gamma
        return lut

    def _apply_gain_gamma_u8(self) -> Optional[np.ndarray]:
        if self._frame_u8_raw is None:
            # raise RuntimeError("No 8-bit frame buffer available")
            print("No 8-bit frame buffer available")
            return None

        gain = float(self.gain)
        gamma = float(self._gamma)
        if not np.isfinite(gain) or gain == 1.0:
            gain = 1.0
        if not np.isfinite(gamma) or gamma <= 0.0 or gamma == 1.0:
            gamma = 1.0
        if gain < 0.0:
            gain = 0.0

        if gain == 1.0 and gamma == 1.0:
            return self._frame_u8_raw

        h, w = self._frame_u8_raw.shape
        if self._frame_u8_disp is None or self._frame_u8_disp.shape != (h, w):
            self._frame_u8_disp = np.empty((h, w), dtype=np.uint8)

        lut = self._ensure_gain_gamma_lut()
        np.take(lut, self._frame_u8_raw, out=self._frame_u8_disp)
        return self._frame_u8_disp

    def _on_gamma_changed(self, value: float) -> None:
        self._gamma = float(value)
        if self.video_path and self._frame_u8_raw is not None:
            frame_u8 = self._apply_gain_gamma_u8()
            qimg = frame_to_qimage(frame_u8)
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

    def _on_gain_changed(self, value: float) -> None:
        self.gain = float(value)
        if self.video_path and self._frame_u8_raw is not None:
            frame_u8 = self._apply_gain_gamma_u8()
            qimg = frame_to_qimage(frame_u8)
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

    def _on_calibration(self, mode="None") -> None:
        points = []
        if mode == "point":
            pass
        elif mode == "line":
            pass
        elif mode == "circle":
            pass
        pass  # Placeholder for calibration change handling

    def _set_nav_enabled(self, enabled: bool) -> None:
        self.prev_action.setEnabled(enabled)
        self.next_action.setEnabled(enabled)

    def _update_title(self) -> None:
        if not self.video_path or self.total_frames <= 0:
            self.setWindowTitle("Mie Postprocessing - Frame Viewer")
            return
        base = os.path.basename(self.video_path)
        self.setWindowTitle(f"{base} - Frame {self.frame_index + 1}/{self.total_frames}")

    def open_video_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open .cine video",
            "",
            "Cine Videos (*.cine);;All Files (*)",
        )
        if path:
            self.load_video(path, 0)

    def load_video(self, video_path: str, frame_number: int = 0) -> None:
        if not os.path.exists(video_path):
            QMessageBox.critical(self, "Load Video", f"Video not found:\n{video_path}")
            return
        try:
            self.reader.load(video_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load Video", f"Failed to load video:\n{video_path}\n\n{exc}")
            return

        self.video_path = video_path
        self.total_frames = int(self.reader.frame_count)
        self.view.reset_zoom()
        self._set_nav_enabled(self.total_frames > 0)
        self.show_frame(max(0, min(int(frame_number), self.total_frames - 1)))

    def show_frame(self, frame_number: int) -> None:
        """
        Read `frame_number` and update the scene.
        """
        if not self.video_path:
            return
        if not (0 <= frame_number < self.total_frames):
            return

        try:
            frame_u16 = self.reader.read_frame(int(frame_number))
        except Exception as exc:
            QMessageBox.critical(self, "Read Frame", f"Failed to read frame {frame_number}:\n\n{exc}")
            return

        self._frame_buffer = np.ascontiguousarray(frame_u16)

        # Fast display path: Phantom cine data is typically 12-bit stored in uint16.
        if self._frame_buffer.dtype == np.uint16:
            h, w = self._frame_buffer.shape
            if self._frame_u8_raw is None or self._frame_u8_raw.shape != (h, w):
                self._frame_u8_raw = np.empty((h, w), dtype=np.uint8)
            bits_shifted = 16 - BITS
            np.right_shift(self._frame_buffer, bits_shifted, out=self._frame_u8_raw, casting="unsafe")
            frame_u8 = self._apply_gain_gamma_u8()
            qimg = frame_to_qimage(frame_u8)
        else:
            qimg = frame_to_qimage(self._frame_buffer)

        # Convert the selected NumPy frame to a QImage, then to a QPixmap for display.
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.frame_index = int(frame_number)

        # Reset overlay. Replace this with a real path when you implement drawing.
        path = QPainterPath()
        self.mask_item.setPath(path)

        # Make the scene rect match the image; helps with scrollbars / fitting.
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self._update_title()

    def prev_frame(self) -> None:
        if self.video_path and self.frame_index > 0:
            self.show_frame(self.frame_index - 1)

    def next_frame(self) -> None:
        if self.video_path and self.frame_index < self.total_frames - 1:
            self.show_frame(self.frame_index + 1)


def main() -> int:
    """
    Script entrypoint.

    Key rule: Create `QApplication` before any QWidget.
    """
    video_path: str | None = None
    frame_number = 0

    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-"):
        video_path = argv[0]
        if len(argv) >= 2:
            try:
                frame_number = int(argv[1])
            except ValueError:
                frame_number = 0

    # Every PyQt app needs exactly one QApplication instance.
    app = QApplication(sys.argv)

    # Create and show the main window.
    viewer = FrameViewer(video_path=video_path, frame_number=frame_number)
    viewer.resize(900, 900)
    viewer.show()

    # Enter the Qt event loop (blocks until the window closes).
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
