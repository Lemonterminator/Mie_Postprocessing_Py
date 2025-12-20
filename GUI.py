"""
Qt-based GUI (PySide6) for cine video annotation.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from OSCC_postprocessing.cine_utils import CineReader
from OSCC_postprocessing.circ_calculator import calc_circle
from OSCC_postprocessing.cone_angle import angle_signal_density, plot_angle_signal_density
from OSCC_postprocessing.functions_videos import video_histogram_with_contour
from OSCC_postprocessing.zoom_utils import enlarge_image

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except ModuleNotFoundError as e:  # pragma: no cover
    raise ModuleNotFoundError(
        "PySide6 is required to run `GUI.py`.\n"
        "Install it with:\n"
        "  python -m pip install PySide6"
    ) from e

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

bits = 12
levels = 2.0**bits


def _pil_rgba_to_qimage(img: Image.Image) -> QtGui.QImage:
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(
        data, img.width, img.height, img.width * 4, QtGui.QImage.Format_RGBA8888
    )
    return qimg.copy()


class TiledCompositedView(QtWidgets.QGraphicsView):
    zoom_changed = QtCore.Signal(int, QtCore.QPointF)  # new zoom, viewport pos
    paint_at = QtCore.Signal(int, int, bool)  # x, y, paint(True)/erase(False)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self._tile_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._tile_item)

        self._center_item = QtWidgets.QGraphicsEllipseItem()
        self._center_item.setPen(QtGui.QPen(QtGui.QColor("yellow"), 2))
        self._center_item.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self._scene.addItem(self._center_item)

        self._calib_circle_item = QtWidgets.QGraphicsEllipseItem()
        calib_pen = QtGui.QPen(QtGui.QColor("red"), 1)
        calib_pen.setDashPattern([4, 4])
        self._calib_circle_item.setPen(calib_pen)
        self._calib_circle_item.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        self._scene.addItem(self._calib_circle_item)

        self._plume_items: list[QtWidgets.QGraphicsLineItem] = []

        self._update_timer = QtCore.QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._redraw_now)

        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("black")))
        self.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )

        self.horizontalScrollBar().valueChanged.connect(self.schedule_redraw)
        self.verticalScrollBar().valueChanged.connect(self.schedule_redraw)

        self.zoom_factor = 1
        self.display_pad = 10

        self._base_rgba_pad: Optional[Image.Image] = None
        self._orig_w = 0
        self._orig_h = 0
        self._mask: Optional[np.ndarray] = None

        self._show_mask = True
        self._brush_color = (255, 0, 0)
        self._alpha = 255

        self._center_x = 0.0
        self._center_y = 0.0
        self._calib_radius = 0.0
        self._num_plumes = 0
        self._plume_offset = 0.0

        self._is_painting = False
        self._paint_mode = True

    def clear(self):
        self._base_rgba_pad = None
        self._mask = None
        self._orig_w = 0
        self._orig_h = 0
        self._tile_item.setPixmap(QtGui.QPixmap())
        self._center_item.setVisible(False)
        self._calib_circle_item.setVisible(False)
        for it in self._plume_items:
            self._scene.removeItem(it)
        self._plume_items.clear()
        self._scene.setSceneRect(QtCore.QRectF(0, 0, 1, 1))

    def set_zoom(self, zoom_factor: int, anchor_viewport_pos: Optional[QtCore.QPointF] = None):
        zoom_factor = max(1, int(zoom_factor))
        if zoom_factor == self.zoom_factor:
            return

        if anchor_viewport_pos is None:
            anchor_viewport_pos = QtCore.QPointF(
                self.viewport().width() / 2, self.viewport().height() / 2
            )

        old_zoom = self.zoom_factor
        anchor_scene = self.mapToScene(anchor_viewport_pos.toPoint())

        self.zoom_factor = zoom_factor
        self._update_scene_rect()

        if old_zoom > 0:
            scale = self.zoom_factor / old_zoom
            new_anchor_scene = QtCore.QPointF(
                anchor_scene.x() * scale, anchor_scene.y() * scale
            )
            self._scroll_to_anchor(new_anchor_scene, anchor_viewport_pos)

        self.schedule_redraw()

    def set_sources(
        self,
        base_rgba_pad: Image.Image,
        orig_width: int,
        orig_height: int,
        mask: np.ndarray,
        display_pad: int = 10,
    ):
        self._base_rgba_pad = base_rgba_pad
        self._orig_w = int(orig_width)
        self._orig_h = int(orig_height)
        self._mask = mask
        self.display_pad = int(display_pad)
        self._update_scene_rect()
        self.schedule_redraw()

    def set_mask_style(self, show_mask: bool, brush_color: tuple[int, int, int], alpha: int):
        self._show_mask = bool(show_mask)
        self._brush_color = tuple(int(c) for c in brush_color)
        self._alpha = int(alpha)
        self.schedule_redraw()

    def set_overlay_params(
        self,
        center_x: float,
        center_y: float,
        calib_radius: float,
        num_plumes: int,
        plume_offset: float,
    ):
        self._center_x = float(center_x)
        self._center_y = float(center_y)
        self._calib_radius = float(calib_radius)
        self._num_plumes = int(num_plumes)
        self._plume_offset = float(plume_offset)
        self.schedule_redraw()

    def schedule_redraw(self):
        if not self._update_timer.isActive():
            self._update_timer.start(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.schedule_redraw()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y()
        direction = 1 if delta > 0 else -1
        new_zoom = max(1, self.zoom_factor + direction)
        self.zoom_changed.emit(new_zoom, event.position())
        event.accept()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            self._is_painting = True
            self._paint_mode = event.button() == QtCore.Qt.MouseButton.LeftButton
            self._emit_paint(event.position())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._is_painting:
            self._emit_paint(event.position())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if event.button() in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.RightButton,
        ):
            self._is_painting = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _emit_paint(self, viewport_pos: QtCore.QPointF):
        if self._mask is None:
            return
        scene_pos = self.mapToScene(viewport_pos.toPoint())
        x = int(scene_pos.x() / self.zoom_factor)
        y = int(scene_pos.y() / self.zoom_factor)
        x = max(0, min(x, self._mask.shape[1] - 1))
        y = max(0, min(y, self._mask.shape[0] - 1))
        self.paint_at.emit(x, y, self._paint_mode)

    def _update_scene_rect(self):
        if self._base_rgba_pad is None:
            self._scene.setSceneRect(QtCore.QRectF(0, 0, 1, 1))
            return
        scaled_w = (self._orig_w + self.display_pad) * self.zoom_factor
        scaled_h = (self._orig_h + self.display_pad) * self.zoom_factor
        self._scene.setSceneRect(QtCore.QRectF(0, 0, scaled_w, scaled_h))

    def _scroll_to_anchor(self, anchor_scene: QtCore.QPointF, anchor_viewport: QtCore.QPointF):
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hbar.setValue(int(anchor_scene.x() - anchor_viewport.x()))
        vbar.setValue(int(anchor_scene.y() - anchor_viewport.y()))

    def _mask_tile(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        if self._mask is None:
            return np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)

        h, w = self._mask.shape[:2]
        tile = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)

        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)
        if src_x1 <= src_x0 or src_y1 <= src_y0:
            return tile

        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        tile[dst_y0:dst_y1, dst_x0:dst_x1] = self._mask[src_y0:src_y1, src_x0:src_x1]
        return tile

    def _update_overlays(self):
        if self._base_rgba_pad is None or self._mask is None:
            self._center_item.setVisible(False)
            self._calib_circle_item.setVisible(False)
            for it in self._plume_items:
                self._scene.removeItem(it)
            self._plume_items.clear()
            return

        cx_scene = self._center_x * self.zoom_factor
        cy_scene = self._center_y * self.zoom_factor

        r = 5
        self._center_item.setRect(cx_scene - r, cy_scene - r, 2 * r, 2 * r)
        self._center_item.setVisible(True)

        if self._calib_radius > 0:
            rr = self._calib_radius * self.zoom_factor
            self._calib_circle_item.setRect(
                cx_scene - rr, cy_scene - rr, 2 * rr, 2 * rr
            )
            self._calib_circle_item.setVisible(True)
        else:
            self._calib_circle_item.setVisible(False)

        for it in self._plume_items:
            self._scene.removeItem(it)
        self._plume_items.clear()

        n_plumes = int(self._num_plumes) if self._num_plumes > 0 else 0
        if n_plumes <= 0:
            return

        step = 360.0 / n_plumes
        offset = float(self._plume_offset) % 360.0
        length = max(self._mask.shape) * self.zoom_factor
        for i in range(n_plumes):
            ang = np.deg2rad(offset + i * step)
            x_end = cx_scene + length * np.cos(ang)
            y_end = cy_scene - length * np.sin(ang)

            plume = QtWidgets.QGraphicsLineItem(cx_scene, cy_scene, x_end, y_end)
            plume.setPen(QtGui.QPen(QtGui.QColor("cyan"), 1))
            self._scene.addItem(plume)
            self._plume_items.append(plume)

            mid_ang = np.deg2rad(offset + i * step + step / 2)
            mx = cx_scene + length * np.cos(mid_ang)
            my = cy_scene - length * np.sin(mid_ang)
            mid = QtWidgets.QGraphicsLineItem(cx_scene, cy_scene, mx, my)
            mid_pen = QtGui.QPen(QtGui.QColor("white"), 1)
            mid_pen.setDashPattern([5, 5])
            mid.setPen(mid_pen)
            self._scene.addItem(mid)
            self._plume_items.append(mid)

    def _redraw_now(self):
        if self._base_rgba_pad is None:
            self.clear()
            return

        self._update_scene_rect()
        self._update_tile()
        self._update_overlays()

    def _update_tile(self):
        if self._base_rgba_pad is None:
            return

        scaled_w = (self._orig_w + self.display_pad) * self.zoom_factor
        scaled_h = (self._orig_h + self.display_pad) * self.zoom_factor

        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        x0s = max(0, int(np.floor(visible.left())))
        y0s = max(0, int(np.floor(visible.top())))
        x1s = min(int(np.ceil(visible.right())), int(scaled_w))
        y1s = min(int(np.ceil(visible.bottom())), int(scaled_h))

        if x1s <= x0s or y1s <= y0s:
            x0s, y0s = 0, 0
            x1s, y1s = int(scaled_w), int(scaled_h)

        x0 = x0s // self.zoom_factor
        y0 = y0s // self.zoom_factor
        x1 = int(np.ceil(x1s / self.zoom_factor))
        y1 = int(np.ceil(y1s / self.zoom_factor))

        max_w, max_h = self._base_rgba_pad.width, self._base_rgba_pad.height
        x0 = max(0, min(x0, max_w))
        y0 = max(0, min(y0, max_h))
        x1 = max(0, min(x1, max_w))
        y1 = max(0, min(y1, max_h))
        if x1 <= x0 or y1 <= y0:
            return

        base_tile = self._base_rgba_pad.crop((x0, y0, x1, y1))
        mask_tile = self._mask_tile(x0, y0, x1, y1)

        base_tile = enlarge_image(base_tile, int(self.zoom_factor)).convert("RGBA")
        composited = base_tile

        if self._show_mask and self._mask is not None:
            mask_img = Image.fromarray((mask_tile * 255).astype(np.uint8))
            mask_img = enlarge_image(mask_img, int(self.zoom_factor)).convert("L")
            overlay = Image.new(
                "RGBA", mask_img.size, (*self._brush_color, int(self._alpha))
            )
            composited = composited.copy()
            composited.paste(overlay, (0, 0), mask_img)

        pixmap = QtGui.QPixmap.fromImage(_pil_rgba_to_qimage(composited))
        self._tile_item.setPixmap(pixmap)
        self._tile_item.setPos(x0 * self.zoom_factor, y0 * self.zoom_factor)


class VideoAnnotatorUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cine Video Annotator")
        self.resize(1400, 900)

        self.reader = CineReader()
        self.total_frames = 0
        self.current_index = 0

        self.display_pad = 10
        self.orig_img: Optional[Image.Image] = None
        self.base_rgba_pad: Optional[Image.Image] = None
        self.mask: Optional[np.ndarray] = None
        self.current_img8: Optional[np.ndarray] = None

        self.brush_color = (255, 0, 0)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        self._build_controls(root)
        self._build_content(root)
        self._set_controls_enabled(False)
        self._update_calib_button()

    def _build_controls(self, root_layout: QtWidgets.QVBoxLayout):
        panel = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(panel)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)
        root_layout.addWidget(panel, 0)

        # Row 0: Load / navigation
        self.load_btn = QtWidgets.QPushButton("Load Video")
        self.frame_label = QtWidgets.QLabel("Frame: 0/0")
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(1)
        self.frame_spin.setValue(1)
        self.frame_spin.setFixedWidth(90)
        self.go_btn = QtWidgets.QPushButton("Go")
        self.prev_btn = QtWidgets.QPushButton("Prev Frame")
        self.next_btn = QtWidgets.QPushButton("Next Frame")
        self.select_btn = QtWidgets.QPushButton("Select Frame")
        self.export_btn = QtWidgets.QPushButton("Export Mask")

        grid.addWidget(self.load_btn, 0, 0)
        grid.addWidget(self.frame_label, 0, 1)
        grid.addWidget(self.frame_spin, 0, 2)
        grid.addWidget(self.go_btn, 0, 3)
        grid.addWidget(self.prev_btn, 0, 4)
        grid.addWidget(self.next_btn, 0, 5)
        grid.addWidget(self.select_btn, 0, 6)
        grid.addWidget(self.export_btn, 0, 7)
        grid.setColumnStretch(8, 1)

        # Row 1: Plumes + calibration params
        self.num_plumes = QtWidgets.QSpinBox()
        self.num_plumes.setMinimum(0)
        self.num_plumes.setMaximum(360)
        self.num_plumes.setFixedWidth(90)
        self.plume_offset = QtWidgets.QDoubleSpinBox()
        self.plume_offset.setDecimals(2)
        self.plume_offset.setRange(-3600.0, 3600.0)
        self.plume_offset.setFixedWidth(90)

        self.coord_x = QtWidgets.QDoubleSpinBox()
        self.coord_x.setDecimals(3)
        self.coord_x.setRange(-1e9, 1e9)
        self.coord_x.setFixedWidth(120)
        self.coord_y = QtWidgets.QDoubleSpinBox()
        self.coord_y.setDecimals(3)
        self.coord_y.setRange(-1e9, 1e9)
        self.coord_y.setFixedWidth(120)

        self.calib_radius = QtWidgets.QDoubleSpinBox()
        self.calib_radius.setDecimals(3)
        self.calib_radius.setRange(0.0, 1e9)
        self.calib_radius.setFixedWidth(120)

        self.circle_btn = QtWidgets.QPushButton("Calibration")
        self.save_cfg_btn = QtWidgets.QPushButton("Save Config")

        grid.addWidget(QtWidgets.QLabel("Plumes:"), 1, 0)
        grid.addWidget(self.num_plumes, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Offset:"), 1, 2)
        grid.addWidget(self.plume_offset, 1, 3)
        grid.addWidget(QtWidgets.QLabel("Centre X:"), 1, 4)
        grid.addWidget(self.coord_x, 1, 5)
        grid.addWidget(QtWidgets.QLabel("Centre Y:"), 1, 6)
        grid.addWidget(self.coord_y, 1, 7)
        grid.addWidget(QtWidgets.QLabel("Calib R:"), 1, 8)
        grid.addWidget(self.calib_radius, 1, 9)
        grid.addWidget(self.circle_btn, 1, 10)
        grid.addWidget(self.save_cfg_btn, 1, 11)

        # Row 2: Gain/Gamma/Black/White + Apply
        self.vars: dict[str, QtWidgets.QDoubleSpinBox] = {}
        for i, name in enumerate(["Gain", "Gamma", "Black", "White"]):
            label = QtWidgets.QLabel(f"{name}:")
            box = QtWidgets.QDoubleSpinBox()
            box.setDecimals(3)
            box.setRange(-1e9, 1e9)
            if name in ("Gain", "Gamma"):
                box.setValue(1.0)
            elif name == "White":
                box.setValue(255.0)
            else:
                box.setValue(0.0)
            box.setFixedWidth(90)
            self.vars[name.lower()] = box
            grid.addWidget(label, 2, i * 2)
            grid.addWidget(box, 2, i * 2 + 1)

        self.apply_btn = QtWidgets.QPushButton("Apply")
        grid.addWidget(self.apply_btn, 2, 10)

        # Row 3: Brush controls
        self.brush_shape = QtWidgets.QComboBox()
        self.brush_shape.addItems(["circle", "square"])
        self.brush_shape.setFixedWidth(120)
        self.show_mask = QtWidgets.QCheckBox("Show Mask")
        self.show_mask.setChecked(True)
        self.brush_size = QtWidgets.QSpinBox()
        self.brush_size.setRange(1, 10000)
        self.brush_size.setValue(10)
        self.brush_size.setFixedWidth(90)
        self.color_btn = QtWidgets.QPushButton("Color")
        self.alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 255)
        self.alpha_slider.setValue(255)
        self.alpha_slider.setFixedWidth(140)

        grid.addWidget(QtWidgets.QLabel("Brush:"), 3, 0)
        grid.addWidget(self.brush_shape, 3, 1)
        grid.addWidget(self.show_mask, 3, 2, 1, 2)
        grid.addWidget(QtWidgets.QLabel("Size:"), 3, 4)
        grid.addWidget(self.brush_size, 3, 5)
        grid.addWidget(self.color_btn, 3, 6)
        grid.addWidget(QtWidgets.QLabel("Alpha:"), 3, 7)
        grid.addWidget(self.alpha_slider, 3, 8, 1, 2)

        # Row 4: Ring mask parameters
        self.inner_radius = QtWidgets.QSpinBox()
        self.inner_radius.setRange(0, 10_000_000)
        self.inner_radius.setValue(0)
        self.inner_radius.setFixedWidth(90)
        self.outer_radius = QtWidgets.QSpinBox()
        self.outer_radius.setRange(0, 10_000_000)
        self.outer_radius.setValue(0)
        self.outer_radius.setFixedWidth(90)
        self.add_ring_btn = QtWidgets.QPushButton("Add Ring")

        grid.addWidget(QtWidgets.QLabel("Inner R:"), 4, 0)
        grid.addWidget(self.inner_radius, 4, 1)
        grid.addWidget(QtWidgets.QLabel("Outer R:"), 4, 2)
        grid.addWidget(self.outer_radius, 4, 3)
        grid.addWidget(self.add_ring_btn, 4, 4)
        grid.setColumnStretch(12, 1)

        self.load_btn.clicked.connect(self.load_video)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.go_btn.clicked.connect(self._on_go_frame)
        self.select_btn.clicked.connect(self.open_frame_selector)
        self.export_btn.clicked.connect(self.export_mask)
        self.apply_btn.clicked.connect(lambda: self.update_image(compute_global=False))
        self.num_plumes.valueChanged.connect(self._on_plumes_changed)
        self.plume_offset.valueChanged.connect(self._on_overlay_params_changed)
        self.coord_x.valueChanged.connect(self._on_overlay_params_changed)
        self.coord_y.valueChanged.connect(self._on_overlay_params_changed)
        self.calib_radius.valueChanged.connect(self._on_overlay_params_changed)
        self.circle_btn.clicked.connect(self.open_circle_selector)
        self.save_cfg_btn.clicked.connect(self.save_config)
        self.show_mask.toggled.connect(self._on_mask_style_changed)
        self.alpha_slider.valueChanged.connect(self._on_mask_style_changed)
        self.color_btn.clicked.connect(self.choose_color)
        self.add_ring_btn.clicked.connect(self.add_ring_mask)

    def _build_content(self, root_layout: QtWidgets.QVBoxLayout):
        self.view = TiledCompositedView()
        root_layout.addWidget(self.view, 1)

        # Plot panel disabled for performance; uncomment to restore.
        # splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        # root_layout.addWidget(splitter, 1)
        # splitter.addWidget(self.view)
        #
        # right = QtWidgets.QWidget()
        # right_layout = QtWidgets.QVBoxLayout(right)
        # right_layout.setContentsMargins(6, 6, 6, 6)
        # splitter.addWidget(right)
        # splitter.setStretchFactor(0, 3)
        # splitter.setStretchFactor(1, 2)
        #
        # self.fig = Figure(figsize=(4, 4))
        # self.ax_hist = self.fig.add_subplot(211)
        # self.ax_angle = self.fig.add_subplot(212)
        # self.ax_angle.set_xlim(-360, 360)
        #
        # self.canvas_hist = FigureCanvas(self.fig)
        # right_layout.addWidget(self.canvas_hist, 1)
        self.fig = None
        self.ax_hist = None
        self.ax_angle = None
        self.canvas_hist = None

        self.view.zoom_changed.connect(self._on_zoom_changed)
        self.view.paint_at.connect(self._on_paint)

    def _set_controls_enabled(self, enabled: bool):
        for w in (
            self.prev_btn,
            self.next_btn,
            self.apply_btn,
            self.select_btn,
            self.export_btn,
        ):
            w.setEnabled(enabled)
        self._update_calib_button()

    def _update_calib_button(self):
        self.circle_btn.setEnabled(bool(self.total_frames) and self.num_plumes.value() > 0)

    def _on_plumes_changed(self, _value: int):
        self._update_calib_button()
        self._on_overlay_params_changed()

    def _on_overlay_params_changed(self, *_args):
        self.view.set_overlay_params(
            self.coord_x.value(),
            self.coord_y.value(),
            self.calib_radius.value(),
            self.num_plumes.value(),
            self.plume_offset.value(),
        )
        # Real-time angle plot updates disabled for performance.
        # self._update_angle_plot_only()

    def _on_mask_style_changed(self, *_args):
        self.view.set_mask_style(
            self.show_mask.isChecked(),
            self.brush_color,
            self.alpha_slider.value(),
        )

    def _on_zoom_changed(self, new_zoom: int, viewport_pos: QtCore.QPointF):
        self.view.set_zoom(new_zoom, anchor_viewport_pos=viewport_pos)

    def load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Cine", "", "Cine (*.cine)"
        )
        if not path:
            return
        try:
            self.reader.load(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot load video:\n{e}")
            return

        self.total_frames = int(self.reader.frame_count)
        self.current_index = 0
        self.frame_spin.setMaximum(max(1, self.total_frames))
        self.frame_spin.setValue(1)
        self.mask = np.zeros((self.reader.height, self.reader.width), dtype=np.uint8)
        self._set_controls_enabled(True)
        self.update_image(compute_global=False)

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image(compute_global=False)

    def next_frame(self):
        if self.current_index < self.total_frames - 1:
            self.current_index += 1
            self.update_image(compute_global=False)

    def _on_go_frame(self):
        idx = self.frame_spin.value() - 1
        if 0 <= idx < self.total_frames:
            self.current_index = idx
            self.update_image(compute_global=False)

    def update_image(self, compute_global: bool = False):
        if self.total_frames == 0:
            return

        frame = self.reader.read_frame(self.current_index).astype(np.float32)
        if self.mask is None or self.mask.shape != frame.shape:
            self.mask = np.zeros(frame.shape, dtype=np.uint8)

        gain = float(self.vars["gain"].value())
        gamma = float(self.vars["gamma"].value())
        black = float(self.vars["black"].value())
        white = float(self.vars["white"].value())

        img = frame / levels * gain
        img = np.clip(img, 0, 1)
        if gamma > 0 and gamma != 1:
            img = img**gamma

        img8 = (img * 255).astype(np.uint8)
        if white > black and black >= 0:
            img8[img8 < black] = 0
            img8[img8 > white] = 255

        self.current_img8 = img8
        self.orig_img = Image.fromarray(img8)

        base_rgba = self.orig_img.convert("RGBA")
        self.base_rgba_pad = ImageOps.expand(
            base_rgba,
            border=(0, 0, self.display_pad, self.display_pad),
            fill=(0, 0, 0, 255),
        )

        # Real-time histogram/angle plot updates disabled for performance.
        # self._update_plots(img8, compute_global=compute_global)

        self.view.set_sources(
            self.base_rgba_pad,
            orig_width=self.orig_img.width,
            orig_height=self.orig_img.height,
            mask=self.mask,
            display_pad=self.display_pad,
        )
        self._on_overlay_params_changed()
        self._on_mask_style_changed()
        self.frame_label.setText(f"Frame: {self.current_index + 1}/{self.total_frames}")
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(self.current_index + 1)
        self.frame_spin.blockSignals(False)

    def _update_plots(self, img8: np.ndarray, compute_global: bool = False):
        # Real-time histogram and angle plots disabled for performance.
        # self.ax_hist.clear()
        # self.ax_hist.hist(img8.ravel(), bins=256, log=True)
        # self.ax_hist.set_title("Processed Histogram")
        # self.ax_hist.set_ylabel("Count (log scale)")
        #
        # self._update_angle_plot_only()
        # self.canvas_hist.draw_idle()
        #
        # if compute_global:
        #     self._compute_global_plots()
        return

    def _update_angle_plot_only(self):
        # Real-time angular density plot disabled for performance.
        # self.ax_angle.clear()
        # img8 = self.current_img8
        # if img8 is None:
        #     self.canvas_hist.draw_idle()
        #     return
        #
        # cx = float(self.coord_x.value())
        # cy = float(self.coord_y.value())
        # if cx != 0 or cy != 0:
        #     bins, sig, _ = angle_signal_density(img8, cx, cy, N_bins=90)
        #     bins_ext = np.concatenate((bins - 360, bins))
        #     sig_ext = np.concatenate((sig, sig))
        #     self.ax_angle.plot(bins_ext, sig_ext, color="green", alpha=0.5)
        #
        #     n_plumes = int(self.num_plumes.value())
        #     if n_plumes > 0:
        #         step = 360.0 / n_plumes
        #         offset = float(self.plume_offset.value()) % 360.0
        #         for i in range(n_plumes):
        #             ang = offset + i * step
        #             for shift in (-360, 0):
        #                 if ang + shift > -180:
        #                     display_ang = (ang + shift) % 360
        #                     if display_ang > 180:
        #                         display_ang -= 360
        #                     self.ax_angle.axvline(
        #                         display_ang, color="cyan", linestyle="--"
        #                     )
        #
        # self.ax_angle.set_xlim(-180, 180)
        # self.ax_angle.set_title("Angular Signal Density")
        # self.ax_angle.set_xlabel("Angle (deg)")
        # self.ax_angle.set_ylabel("Signal")
        # self.canvas_hist.draw_idle()
        return

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        gain = float(self.vars["gain"].value())
        gamma = float(self.vars["gamma"].value())
        black = float(self.vars["black"].value())
        white = float(self.vars["white"].value())

        img = frame / levels * gain
        img = np.clip(img, 0, 1)
        if gamma > 0 and gamma != 1:
            img = img**gamma
        img8 = (img * 255).astype(np.uint8)
        if white > black and black >= 0:
            img8 = np.clip((img8 - black) * (255.0 / (white - black)), 0, 255).astype(
                np.uint8
            )
        return img8.astype(np.float32) / 255.0

    def _compute_global_plots(self):
        if self.total_frames == 0:
            return

        frames: list[np.ndarray] = []
        for i in range(self.total_frames):
            f = self.reader.read_frame(i).astype(np.float32)
            frames.append(self._process_frame(f))
        video = np.stack(frames, axis=0)

        video_histogram_with_contour(video, bins=100, exclude_zero=True, log=True)

        cx = float(self.coord_x.value())
        cy = float(self.coord_y.value())
        if cx != 0 or cy != 0:
            bins, sig, _ = angle_signal_density(video, cx, cy, N_bins=360)
            n_plumes = int(self.num_plumes.value())
            plume_angles = None
            if n_plumes > 0:
                step = 360.0 / n_plumes
                offset = float(self.plume_offset.value()) % 360.0
                plume_angles = [offset + i * step for i in range(n_plumes)]
            plot_angle_signal_density(bins, sig, log=True, plume_angles=plume_angles)

    def add_ring_mask(self):
        if self.mask is None:
            return

        cx = float(self.coord_x.value())
        cy = float(self.coord_y.value())
        r_in = max(0, int(self.inner_radius.value()))
        r_out = int(self.outer_radius.value())
        if r_out <= 0:
            return

        yy, xx = np.ogrid[: self.mask.shape[0], : self.mask.shape[1]]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        ring = dist2 <= r_out**2
        if r_in > 0:
            ring &= dist2 >= r_in**2
        self.mask[ring] = 1
        self.view.schedule_redraw()

    def choose_color(self):
        initial = QtGui.QColor(*self.brush_color)
        color = QtWidgets.QColorDialog.getColor(initial, self, "Select brush color")
        if not color.isValid():
            return
        self.brush_color = (color.red(), color.green(), color.blue())
        self._on_mask_style_changed()

    def _on_paint(self, x: int, y: int, paint: bool):
        if self.mask is None:
            return

        value = 1 if paint else 0
        size = int(self.brush_size.value())
        height, width = self.mask.shape[:2]

        if self.brush_shape.currentText() == "circle":
            x0 = max(0, x - size)
            x1 = min(width, x + size + 1)
            y0 = max(0, y - size)
            y1 = min(height, y + size + 1)
            yy, xx = np.ogrid[y0 - y : y1 - y, x0 - x : x1 - x]
            area = xx * xx + yy * yy <= size * size
            region = self.mask[y0:y1, x0:x1]
            region[area] = value
        else:
            x0 = max(0, x - size)
            x1 = min(width, x + size)
            y0 = max(0, y - size)
            y1 = min(height, y + size)
            self.mask[y0:y1, x0:x1] = value

        self.view.schedule_redraw()

    def open_frame_selector(self):
        if self.total_frames:
            dlg = FrameSelectorDialog(self)
            dlg.exec()

    def open_circle_selector(self):
        if not self.total_frames:
            return
        n = int(self.num_plumes.value())
        if n <= 0:
            QtWidgets.QMessageBox.information(
                self, "Calibration", "Set number of plumes before calibration."
            )
            return
        if self.orig_img is None:
            return
        dlg = CircleSelectorDialog(self, self.orig_img, n)
        dlg.exec()

    def set_calibration_from_circle(
        self, center: tuple[float, float], radius: float, offset: float
    ):
        self.coord_x.setValue(float(center[0]))
        self.coord_y.setValue(float(center[1]))
        self.calib_radius.setValue(float(radius))
        try:
            self.inner_radius.setValue(int(round(radius)))
        except Exception:
            pass
        try:
            self.plume_offset.setValue(float(offset % 360.0))
        except Exception:
            pass
        self.update_image(compute_global=False)

    def export_mask(self):
        if self.total_frames == 0 or self.mask is None:
            QtWidgets.QMessageBox.critical(self, "Error", "No video loaded")
            return

        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Mask", "", "NumPy file (*.npy)"
        )
        if not file_path:
            return
        if not file_path.lower().endswith(".npy"):
            file_path += ".npy"

        np.save(file_path, self.mask.astype(np.bool_))
        img = Image.fromarray((self.mask * 255).astype(np.uint8))
        img.save(os.path.splitext(file_path)[0] + ".jpg")
        QtWidgets.QMessageBox.information(self, "Export", "Mask exported")

    def save_config(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Folder to Save Config"
        )
        if not folder:
            return
        cfg = {
            "plumes": int(self.num_plumes.value()),
            "offset": float(self.plume_offset.value()),
            "centre_x": float(self.coord_x.value()),
            "centre_y": float(self.coord_y.value()),
            "calib_radius": float(self.calib_radius.value()),
        }
        path = os.path.join(folder, "config.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            QtWidgets.QMessageBox.information(
                self, "Config", f"Configuration saved to {path}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"Could not save config:\n{e}"
            )


class FrameSelectorDialog(QtWidgets.QDialog):
    def __init__(self, parent: "VideoAnnotatorUI"):
        super().__init__(parent)
        self.parent_ui = parent
        self.reader = parent.reader
        self.current_index = parent.current_index
        self.zoom_factor = 1

        self.setWindowTitle("Select Frame")
        self.resize(900, 700)

        layout = QtWidgets.QVBoxLayout(self)

        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        self.prev_btn = QtWidgets.QPushButton("Prev")
        self.next_btn = QtWidgets.QPushButton("Next")
        ctrl.addWidget(self.prev_btn)
        ctrl.addWidget(self.next_btn)

        ctrl.addSpacing(10)
        ctrl.addWidget(QtWidgets.QLabel("Frame:"))
        self.frame_spin = QtWidgets.QSpinBox()
        self.frame_spin.setMinimum(1)
        self.frame_spin.setMaximum(max(1, int(self.reader.frame_count)))
        self.frame_spin.setValue(self.current_index + 1)
        self.frame_spin.setFixedWidth(90)
        ctrl.addWidget(self.frame_spin)

        self.go_btn = QtWidgets.QPushButton("Go")
        ctrl.addWidget(self.go_btn)

        self.use_btn = QtWidgets.QPushButton("Use Frame")
        ctrl.addWidget(self.use_btn)

        self.info = QtWidgets.QLabel("")
        ctrl.addWidget(self.info)
        ctrl.addStretch(1)

        self.view = QtWidgets.QGraphicsView()
        self.view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("black")))
        self.view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        self.scene = QtWidgets.QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.pix_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)
        layout.addWidget(self.view, 1)

        self.prev_btn.clicked.connect(self.prev_frame)
        self.next_btn.clicked.connect(self.next_frame)
        self.go_btn.clicked.connect(self.goto_frame)
        self.use_btn.clicked.connect(self.use_frame)

        self.view.viewport().installEventFilter(self)
        self.show_frame(self.current_index)

    def eventFilter(self, obj, event):
        if obj is self.view.viewport() and event.type() == QtCore.QEvent.Type.Wheel:
            delta = event.angleDelta().y()
            direction = 1 if delta > 0 else -1
            self.zoom_factor = max(1, self.zoom_factor + direction)
            self.show_frame(self.current_index)
            return True
        return super().eventFilter(obj, event)

    def read_frame(self, idx: int) -> Image.Image:
        frame = self.reader.read_frame(idx)
        img8 = np.clip(frame / 16, 0, 255).astype(np.uint8)
        return Image.fromarray(img8)

    def show_frame(self, idx: int):
        if not (0 <= idx < int(self.reader.frame_count)):
            return
        self.current_index = idx
        img = enlarge_image(self.read_frame(idx), int(self.zoom_factor)).convert("RGBA")
        pixmap = QtGui.QPixmap.fromImage(_pil_rgba_to_qimage(img))
        self.pix_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(0, 0, pixmap.width(), pixmap.height()))
        self.info.setText(f"Frame {idx + 1}/{int(self.reader.frame_count)}")
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(idx + 1)
        self.frame_spin.blockSignals(False)

    def prev_frame(self):
        if self.current_index > 0:
            self.show_frame(self.current_index - 1)

    def next_frame(self):
        if self.current_index < int(self.reader.frame_count) - 1:
            self.show_frame(self.current_index + 1)

    def goto_frame(self):
        idx = self.frame_spin.value() - 1
        self.show_frame(idx)

    def use_frame(self):
        self.parent_ui.current_index = self.current_index
        self.parent_ui.update_image(compute_global=False)
        self.accept()


class CircleSelectorDialog(QtWidgets.QDialog):
    def __init__(self, parent: "VideoAnnotatorUI", image: Image.Image, num_points: int):
        super().__init__(parent)
        self.parent_ui = parent
        self.image = image
        self.num_points = max(3, int(num_points))
        self.zoom_factor = 1
        self.points: list[tuple[float, float]] = []

        self.setWindowTitle("Calibration (Pick Points)")
        self.resize(900, 700)

        layout = QtWidgets.QVBoxLayout(self)
        ctrl = QtWidgets.QHBoxLayout()
        layout.addLayout(ctrl)

        self.info = QtWidgets.QLabel(f"Click {self.num_points} points on the nozzle rim")
        ctrl.addWidget(self.info)
        ctrl.addStretch(1)

        self.view = QtWidgets.QGraphicsView()
        self.view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor("black")))
        self.view.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, False)
        self.scene = QtWidgets.QGraphicsScene(self.view)
        self.view.setScene(self.scene)
        self.pix_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pix_item)
        layout.addWidget(self.view, 1)

        self._point_items: list[QtWidgets.QGraphicsEllipseItem] = []

        self.view.viewport().installEventFilter(self)
        self.view.viewport().setMouseTracking(True)
        self.show_frame()

    def eventFilter(self, obj, event):
        if obj is self.view.viewport():
            if event.type() == QtCore.QEvent.Type.Wheel:
                delta = event.angleDelta().y()
                direction = 1 if delta > 0 else -1
                self.zoom_factor = max(1, self.zoom_factor + direction)
                self.show_frame()
                return True
            if (
                event.type() == QtCore.QEvent.Type.MouseButtonPress
                and event.button() == QtCore.Qt.MouseButton.LeftButton
            ):
                scene_pos = self.view.mapToScene(event.position().toPoint())
                x = float(scene_pos.x() / self.zoom_factor)
                y = float(scene_pos.y() / self.zoom_factor)
                self.points.append((x, y))
                self._draw_points()
                if len(self.points) == self.num_points:
                    center, radius, offset = calc_circle(*self.points)
                    self.parent_ui.set_calibration_from_circle(center, radius, offset)
                    self.accept()
                return True
        return super().eventFilter(obj, event)

    def show_frame(self):
        img = enlarge_image(self.image, int(self.zoom_factor)).convert("RGBA")
        pixmap = QtGui.QPixmap.fromImage(_pil_rgba_to_qimage(img))
        self.pix_item.setPixmap(pixmap)
        self.scene.setSceneRect(QtCore.QRectF(0, 0, pixmap.width(), pixmap.height()))
        self._draw_points()

    def _draw_points(self):
        for it in self._point_items:
            self.scene.removeItem(it)
        self._point_items.clear()

        r = 3
        for x, y in self.points:
            sx = x * self.zoom_factor
            sy = y * self.zoom_factor
            it = QtWidgets.QGraphicsEllipseItem(sx - r, sy - r, 2 * r, 2 * r)
            it.setPen(QtGui.QPen(QtGui.QColor("red"), 2))
            it.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            self.scene.addItem(it)
            self._point_items.append(it)


def main() -> int:
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(sys.argv)
    win = VideoAnnotatorUI()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
