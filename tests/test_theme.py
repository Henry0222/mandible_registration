from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QTreeView
import pytest

from mandible_registration.drop_import import AssignStlDialog
from mandible_registration.gui import MainWindow
from mandible_registration.input_dialog import SequentialStlDialog


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication(["theme-test", "-platform", "offscreen"])
    instance.setQuitOnLastWindowClosed(False)
    return instance


def dark_palette(app):
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    palette = QPalette()
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(role, QColor("#202020"))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(role, QColor("#eeeeee"))
    app.setPalette(palette)


def assert_readable(widget, background, foreground):
    palette = widget.palette()
    assert palette.color(background).lightness() > 220
    assert palette.color(foreground).lightness() < 100


def test_dialogs_and_popup_are_readable_after_system_dark_palette(app, tmp_path):
    dark_palette(app)
    parent = MainWindow()
    parent.show()
    dark_palette(app)  # emulate another dark theme update before opening a child
    dialog = AssignStlDialog([], {}, parent)
    dialog.show()
    app.processEvents()
    try:
        assert_readable(dialog, QPalette.ColorRole.Window, QPalette.ColorRole.WindowText)
        combo = next(iter(dialog.combos.values()))
        combo.showPopup()
        app.processEvents()
        assert_readable(combo, QPalette.ColorRole.Base, QPalette.ColorRole.Text)
        assert_readable(combo.view(), QPalette.ColorRole.Base, QPalette.ColorRole.Text)
        combo.hidePopup()
        buttons = dialog.findChild(QDialogButtonBox)
        assert_readable(buttons, QPalette.ColorRole.Button, QPalette.ColorRole.ButtonText)
    finally:
        dialog.close()
        dialog.deleteLater()
    dark_palette(app)
    dialog = SequentialStlDialog(parent, directory=str(tmp_path))
    dialog.show()
    app.processEvents()
    try:
        assert_readable(dialog, QPalette.ColorRole.Window, QPalette.ColorRole.WindowText)
        tree = dialog.findChild(QTreeView, "treeView")
        assert_readable(tree, QPalette.ColorRole.Base, QPalette.ColorRole.Text)
        assert_readable(dialog.stage_label.parentWidget(), QPalette.ColorRole.Window, QPalette.ColorRole.WindowText)
        assert dialog.stage_label.palette().color(QPalette.ColorRole.WindowText).lightness() < 100
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.close()
        parent.deleteLater()
