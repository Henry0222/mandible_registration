import xml.etree.ElementTree as ET

import pytest
from PySide6.QtCore import QPointF

from mandible_registration.svg_support import set_svg_properties, svg_shape_path


@pytest.mark.parametrize("attributes,end", [
    ({"d": "M 571.93362,196.1413 H 971.76188"}, (971.76188, 196.1413)),
    ({"d": "M0 0 h20 v30 l20 -10"}, (40, 20)),
    ({"d": "M0 0 C20 30 40 30 60 0 S80 -30 100 0 Q120 30 140 0 T180 0"}, (180, 0)),
    ({"d": "M0 0 A20 20 0 0 1 40 0"}, (40, 0)),
])
def test_inkscape_straight_curved_and_arc_paths(attributes, end):
    path = svg_shape_path(ET.Element("path", {"id": "test", **attributes}))
    assert not path.isEmpty()
    assert path.currentPosition().x() == pytest.approx(end[0])
    assert path.currentPosition().y() == pytest.approx(end[1])


def test_line_polyline_and_invalid_geometry():
    line = svg_shape_path(ET.Element("line", {"x1": "5", "y1": "5", "x2": "15", "y2": "15"}))
    assert line.pointAtPercent(.5) == QPointF(10, 10)
    polyline = svg_shape_path(ET.Element("polyline", {"points": "1e1,2e1 30,20 30,40"}))
    assert polyline.currentPosition() == QPointF(30, 40)
    for tag, attrs in (("path", {"d": ""}), ("polyline", {"points": "1,2,3"}), ("rect", {})):
        with pytest.raises(ValueError):
            svg_shape_path(ET.Element(tag, attrs))


def test_dynamic_color_updates_editor_style_without_removing_layout():
    element = ET.Element("path", {"style": "stroke:#c6cdd6;stroke-width:5.36;stroke-linecap:round;display:inline"})
    set_svg_properties(element, stroke="#199958")
    assert element.get("stroke") == "#199958"
    assert "stroke:#199958" in element.get("style")
    assert "stroke-width:5.36" in element.get("style")
    assert "display:inline" in element.get("style")
