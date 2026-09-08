"""Application-owned light palette, independent of Windows dark/light mode."""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


LIGHT_STYLE = """
QWidget { color: #182230; font-family: "Microsoft YaHei UI"; font-size: 13px; }
QMainWindow, QDialog { background-color: #f4f7fb; }
QLabel { background: transparent; }
QLabel#measureHint { background: #e8f1fc; color: #164f83; padding: 9px;
                      border: 1px solid #c9dcf2; border-radius: 5px; }
QGroupBox { background: #ffffff; border: 1px solid #dce4ee; border-radius: 6px;
            margin-top: 10px; padding-top: 9px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QAbstractSpinBox {
    background: #ffffff; color: #182230; border: 1px solid #b8c6d8;
    border-radius: 4px; padding: 4px; selection-background-color: #1769aa;
    selection-color: #ffffff;
}
QComboBox QAbstractItemView, QAbstractItemView {
    background: #ffffff; alternate-background-color: #f2f5f9; color: #182230;
    border: 1px solid #dce4ee; selection-background-color: #1769aa;
    selection-color: #ffffff;
}
QAbstractItemView::item:selected { background: #1769aa; color: #ffffff; }
QHeaderView::section { background: #eef3f8; color: #182230; padding: 6px;
                       border: 1px solid #dce4ee; }
QPushButton, QToolButton { color: #182230; background: #ffffff;
    border: 1px solid #b8c6d8; border-radius: 4px; padding: 5px 8px; }
QPushButton:hover:enabled, QToolButton:hover:enabled { background: #edf5ff; border-color: #5a91ce; }
QPushButton:pressed, QToolButton:pressed { background: #d9eafd; }
QPushButton:disabled, QToolButton:disabled, QComboBox:disabled, QAbstractSpinBox:disabled {
    background: #edf0f5; color: #65758b;
}
QLabel:disabled, QCheckBox:disabled { color: #65758b; }
QMenu, QToolTip { color: #182230; background: #ffffff; border: 1px solid #b8c6d8; }
QMenu::item:selected { background: #1769aa; color: #ffffff; }
QScrollArea { border: 0; background: #f4f7fb; }
QStatusBar { background: #eef3f8; color: #182230; }
QProgressBar { background: #ffffff; color: #183d5a; border: 1px solid #b8c6d8;
               border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #c4e3fb; border-radius: 3px; }
"""


def apply_light_theme() -> None:
    """Set both palette and stylesheet: styling text alone breaks dark dialogs.

    This changes only this Qt process, never the user's OS theme. Calling from
    each top-level window also covers standalone viewers and dialog tests.
    """
    app = QApplication.instance()
    if app is None:
        return
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    palette = QPalette()
    colors = {
        "Window": "#f4f7fb", "WindowText": "#182230", "Base": "#ffffff",
        "AlternateBase": "#f2f5f9", "Text": "#182230", "Button": "#ffffff",
        "ButtonText": "#182230", "ToolTipBase": "#ffffff", "ToolTipText": "#182230",
        "Highlight": "#1769aa", "HighlightedText": "#ffffff", "Link": "#1769aa",
        "LinkVisited": "#6846a5", "PlaceholderText": "#65758b", "Light": "#ffffff",
        "Midlight": "#eef3f8", "Mid": "#b8c6d8", "Dark": "#65758b", "Shadow": "#65758b",
        "BrightText": "#ffffff", "Accent": "#1769aa",
    }
    for role, value in colors.items():
        palette.setColor(getattr(QPalette.ColorRole, role), QColor(value))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#65758b"))
    app.setPalette(palette)
    if app.styleSheet() != LIGHT_STYLE:
        app.setStyleSheet(LIGHT_STYLE)
