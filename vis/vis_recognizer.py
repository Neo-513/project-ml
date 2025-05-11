from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow

from vis.vis_recognizer_ui import Ui_MainWindow
from util import util

from train.train_recognizer import DATASET, HYPERPARAMETER, train
import math
import pyqtgraph
import sys


class MyCore(QMainWindow, Ui_MainWindow):
	XRANGE1 = math.ceil(len(DATASET["train"]) / HYPERPARAMETER["batch_size"]) * HYPERPARAMETER["epoch"]
	XRANGE2 = XRANGE1 // HYPERPARAMETER["validate_freq"]

	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/recognizer/logo.png"))

		plot_widget1 = pyqtgraph.PlotWidget(title="Loss Value")
		plot_widget1.setMouseEnabled(x=False, y=False)
		plot_widget1.getPlotItem().hideButtons()
		plot_widget1.setXRange(0, self.XRANGE1)
		self.centralwidget.layout().addWidget(plot_widget1)

		plot_widget2 = pyqtgraph.PlotWidget(title="Accuracy")
		plot_widget2.setMouseEnabled(x=False, y=False)
		plot_widget2.getPlotItem().hideButtons()
		plot_widget2.setXRange(0, self.XRANGE2)
		plot_widget2.setYRange(80, 100)
		self.centralwidget.layout().addWidget(plot_widget2)

		self.timer = util.timer(1000, self.record_time)
		self.timer.second = 0
		self.timer.start()

		self.my_thread = MyThread(plot_widget1, plot_widget2)
		self.my_thread.start()

	def record_time(self):
		self.timer.second += 1
		h, m, s = self.timer.second // 3600, (self.timer.second // 60) % 60, self.timer.second % 60
		self.label_time.setText(f"Time spent: {h:02}:{m:02}:{s:02}")


class MyThread(QThread):
	signal_update1 = pyqtSignal(float, int)
	signal_update2 = pyqtSignal(float)
	signal_update3 = pyqtSignal(int)

	def __init__(self, plot_widget1, plot_widget2):
		super().__init__()
		util.cast(self.signal_update1).connect(self.update1)
		util.cast(self.signal_update2).connect(self.update2)
		util.cast(self.signal_update3).connect(self.update3)
		self.plot_widget1 = plot_widget1
		self.plot_widget2 = plot_widget2

		self.loss_values = []
		self.accuracies = []
		self.XRANGE1 = tuple(range(1, MyCore.XRANGE1 + 1))
		self.XRANGE2 = tuple(range(1, MyCore.XRANGE2 + 1))

		self.curve1 = self.plot_widget1.plot([], [], pen="r")
		self.curve2 = self.plot_widget2.plot([], [], pen="y")
		self.curve3 = pyqtgraph.InfiniteLine(pen="g")
		self.curve4 = pyqtgraph.InfiniteLine(pen="c", angle=0)
		self.plot_widget1.addItem(self.curve3)
		self.plot_widget2.addItem(self.curve4)

	def run(self):
		train({
			"loss_value": self.signal_update1,
			"accuracy": self.signal_update2,
			"step": self.signal_update3
		})
		
		my_core.timer.stop()
		self.plot_widget1.removeItem(self.curve3)
		self.plot_widget2.removeItem(self.curve4)

	def update1(self, loss_value, epoch):
		self.loss_values.append(loss_value)
		self.curve1.setData(self.XRANGE1[:len(self.loss_values)], self.loss_values)
		self.curve3.setPos(len(self.loss_values))
		my_core.label_epoch.setText(f"Current epoch: {epoch}")
		my_core.label_step.setText(f"Current step: {len(self.loss_values)}")

	def update2(self, accuracy):
		self.accuracies.append(accuracy)
		self.curve2.setData(self.XRANGE2[:len(self.accuracies)], self.accuracies)
		self.curve4.setPos(accuracy)

	def update3(self, step):
		self.plot_widget1.addItem(pyqtgraph.InfiniteLine(pos=step, pen="m"))


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
