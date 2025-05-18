from PyQt6.QtCore import QTimer
import pyqtgraph


def cast(obj):
	return obj


def plot(title, widget, x=None, y=None):
	plot_widget = pyqtgraph.PlotWidget(title=title)
	plot_widget.setMouseEnabled(x=False, y=False)
	plot_widget.getPlotItem().hideButtons()
	plot_widget.setXRange(*x) if x is not None else None
	plot_widget.setYRange(*y) if y is not None else None
	widget.parent().layout().replaceWidget(widget, plot_widget)
	return plot_widget


def xrange(limit):
	return tuple(range(1, limit + 1))


def clock(label):
	def func():
		timer.second += 1
		h, m, s = timer.second // 3600, (timer.second // 60) % 60, timer.second % 60
		label.setText(f"Time spent: {h:02}:{m:02}:{s:02}")

	timer = QTimer()
	timer.setInterval(1000)
	timer.second = 0
	cast(timer).timeout.connect(func)
	return timer
