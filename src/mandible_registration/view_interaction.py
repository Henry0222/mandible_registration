"""Shared Qt mouse mapping for the 3D and section windows."""
import numpy as np
from PySide6.QtCore import QEvent, QObject, QPointF, Qt


class ViewGestures(QObject):
    """Left rotates/clicks, middle pans, right drags zoom, wheel is view-specific."""

    def __init__(self, view):
        super().__init__(view.vtk_widget)
        self.view = view
        self.button = self.press = self.last = None
        self.dragged = False

    def eventFilter(self, watched, event):
        kind = event.type()
        if kind == QEvent.Type.Wheel:
            steps = event.angleDelta().y() / 120
            if not steps and not event.pixelDelta().isNull():
                steps = event.pixelDelta().y() / 40
            self.view.scroll(steps)
            return True
        if kind in (QEvent.Type.MouseButtonPress, QEvent.Type.MouseButtonDblClick):
            self.button = event.button()
            self.press = self.last = QPointF(event.position())
            self.dragged = False
            return True
        if kind == QEvent.Type.MouseMove and self.button is not None:
            pos = QPointF(event.position())
            total = pos - self.press
            self.dragged |= total.x() ** 2 + total.y() ** 2 > 16
            if self.dragged:
                delta = pos - self.last
                if self.button == Qt.MouseButton.LeftButton:
                    self.view.rotate(delta.x(), delta.y())
                elif self.button == Qt.MouseButton.MiddleButton:
                    self.view.pan(delta.x(), delta.y())
                elif self.button == Qt.MouseButton.RightButton:
                    self.view.zoom(float(np.exp(np.clip(delta.y() * .01, -.5, .5))), interactive=True)
            self.last = pos
            return True
        if kind == QEvent.Type.MouseButtonRelease and self.button is not None:
            delta = event.position() - self.press
            if (self.button == Qt.MouseButton.LeftButton and not self.dragged
                    and delta.x() ** 2 + delta.y() ** 2 <= 16):
                self.view.pick_qt_position(event.position())
            self.button = None
            self.view.finish_interaction()
            return True
        if kind in (QEvent.Type.Hide, QEvent.Type.FocusOut) and self.button is not None:
            self.button = None
            self.view.finish_interaction()
        return False
