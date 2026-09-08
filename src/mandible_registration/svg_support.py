"""Editable SVG geometry and presentation shared by the workflow renderer."""
from __future__ import annotations

import re

from fontTools.pens.basePen import BasePen
from fontTools.svgLib.path import parse_path
from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainterPath


class _QtPathPen(BasePen):
    def __init__(self):
        super().__init__()
        self.path = QPainterPath()

    def _moveTo(self, point):
        self.path.moveTo(*point)

    def _lineTo(self, point):
        self.path.lineTo(*point)

    def _curveToOne(self, first, second, end):
        self.path.cubicTo(QPointF(*first), QPointF(*second), QPointF(*end))

    def _qCurveToOne(self, control, end):
        self.path.quadTo(QPointF(*control), QPointF(*end))

    def _closePath(self):
        self.path.closeSubpath()


def svg_shape_path(element):
    """Keep the click geometry in sync with editor-exported lines and curves."""
    tag = element.tag.rsplit("}", 1)[-1]
    path = QPainterPath()
    if tag == "path":
        pen = _QtPathPen()
        parse_path(element.get("d", ""), pen)
        path = pen.path
    elif tag in ("polyline", "polygon"):
        tokens = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", element.get("points", ""))
        values = [float(token) for token in tokens]
        if len(values) < 4 or len(values) % 2:
            raise ValueError(f"流程图连线坐标不完整：{element.get('id')}")
        path.moveTo(*values[:2])
        for index in range(2, len(values), 2):
            path.lineTo(*values[index:index + 2])
        if tag == "polygon":
            path.closeSubpath()
    elif tag == "line":
        path.moveTo(float(element.get("x1", 0)), float(element.get("y1", 0)))
        path.lineTo(float(element.get("x2", 0)), float(element.get("y2", 0)))
    else:
        raise ValueError(f"流程图阶段连线需要 path、polyline 或 line：{element.get('id')}")
    if path.isEmpty():
        raise ValueError(f"流程图阶段连线为空：{element.get('id')}")
    return path


def set_svg_properties(element, **properties):
    """Inline editor styles override attributes; update both, preserving layout."""
    style = {}
    for declaration in element.get("style", "").split(";"):
        if ":" in declaration:
            name, value = declaration.split(":", 1)
            style[name.strip()] = value.strip()
    for name, value in properties.items():
        element.set(name, str(value))
        style[name] = str(value)
    element.set("style", ";".join(f"{name}:{value}" for name, value in style.items()))
