from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCursor, QImage, QIcon, QPixmap
from PyQt6.QtWidgets import QLabel, QMessageBox
import os
import pickle
import pyqtgraph


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


def plot(title, widget, xrng=None, yrng=None):
	plot_widget = pyqtgraph.PlotWidget(title=title)
	plot_widget.setMouseEnabled(x=False, y=False)
	plot_widget.getPlotItem().hideButtons()
	plot_widget.setXRange(*xrng) if xrng is not None else None
	plot_widget.setYRange(*yrng) if yrng is not None else None
	widget.parent().layout().replaceWidget(widget, plot_widget)
	return plot_widget


def record_time(tmr, label):
	tmr.second += 1
	h, m, s = tmr.second // 3600, (tmr.second // 60) % 60, tmr.second % 60
	label.setText(f"Time spent: {h:02}:{m:02}:{s:02}")


def xrange(limit):
	return tuple(range(1, limit + 1))


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


def read(file_path):
	extension = os.path.splitext(file_path)[-1]
	if extension == ".pkl":
		with open(file_path, mode="rb") as file:
			datas = pickle.load(file)
	else:
		with open(file_path, mode="r", encoding="utf-8") as file:
			datas = file.read()
	print(f"[READ] {file_path}")
	return datas


def write(file_path, datas):
	extension = os.path.splitext(file_path)[-1]
	if extension == ".pkl":
		with open(file_path, mode="wb") as file:
			pickle.dump(datas, cast(file))
	else:
		with open(file_path, mode="w", encoding="utf-8") as file:
			file.write(datas)
	print(f"[WRITE] {file_path}")
