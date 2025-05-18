from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCursor, QImage, QIcon, QPixmap
from PyQt6.QtWidgets import QLabel, QMessageBox


def cast(obj):
	return obj


def timer(interval, func):
	t = QTimer()
	t.setInterval(interval)
	cast(t).timeout.connect(func)
	return t


def pixmap(label, color):
	pm = QPixmap(label.minimumSize()) if label.pixmap().cacheKey() == 0 else label.pixmap()
	pm.fill(color)
	label.setPixmap(pm)
	return pm


def mask(face, path=None, color=None, hide=False, pointer=False):
	label = QLabel(parent=face.parent())
	label.setFixedSize(face.minimumSize())
	label.setGeometry(*cast(label.parent()).layout().getContentsMargins()[:2], label.width(), label.height())
	label.setHidden(hide)
	label.setCursor(QCursor(Qt.CursorShape.PointingHandCursor)) if pointer else None
	label.setPixmap(QPixmap(QImage(path))) if path is not None else None
	pixmap(label, color) if color is not None else None
	return label


def dialog(msg, msg_type):
	message_box = QMessageBox()
	message_box.setText(msg)
	message_box.setWindowIcon(QIcon(f"../static/common/{msg_type}.png"))
	message_box.setWindowTitle(" ")

	if msg_type == "warning":
		message_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
	return message_box.exec() == QMessageBox.StandardButton.Yes
