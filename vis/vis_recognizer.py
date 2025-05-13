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
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/recognizer/logo.png"))

		steps = math.ceil(len(DATASET["train"]) / HYPERPARAMETER["batch_size"]) * HYPERPARAMETER["epoch"]
		xranges = {
			"loss": util.xrange(steps),
			"accuracy": util.xrange(steps // HYPERPARAMETER["validate_freq"])
		}
		plots = {
			"loss": util.plot("Loss", self.widget_loss, xrng=(0, len(xranges["loss"]))),
			"accuracy": util.plot("Accuracy", self.widget_accuracy, xrng=(0, len(xranges["accuracy"])), yrng=(85, 100))
		}
		graphs = {
			"loss": plots["loss"].plot([], [], pen="r"),
			"accuracy": plots["accuracy"].plot([], [], pen="y", symbolBrush="y", symbolPen="y", symbol="o", symbolSize=3)
		}

		self.timer = util.timer(1000, lambda: util.record_time(self.timer, self.label_timer))
		self.timer.second = 0
		self.timer.start()

		self.my_thread = MyThread({
			"xrange": xranges,
			"yvalue": {"loss": [], "accuracy": []},
			"plot": plots,
			"graph": graphs,
			"indicator": pyqtgraph.InfiniteLine(pen="g"),
			"label": {"epoch": self.label_epoch, "step": self.label_step},
			"timer": self.timer
		})
		self.my_thread.start()


class MyThread(QThread):
	signal_loss = pyqtSignal(float, int)
	signal_accuracy = pyqtSignal(float)
	signal_step = pyqtSignal(int)

	def __init__(self, params):
		super().__init__()
		util.cast(self.signal_loss).connect(self.update_loss)
		util.cast(self.signal_accuracy).connect(self.update_accuracy)
		util.cast(self.signal_step).connect(self.update_step)

		self.params = params
		self.params["plot"]["loss"].addItem(self.params["indicator"])

	def run(self):
		train({
			"loss": self.signal_loss,
			"accuracy": self.signal_accuracy,
			"step": self.signal_step
		})

		self.params["timer"].stop()
		self.params["plot"]["loss"].removeItem(self.params["indicator"])

	def update_loss(self, loss, epoch):
		self.params["yvalue"]["loss"].append(loss)
		step = len(self.params["yvalue"]["loss"])
		self.params["indicator"].setPos(step)

		x_data = self.params["xrange"]["loss"][:step]
		y_data = self.params["yvalue"]["loss"]
		self.params["graph"]["loss"].setData(x_data, y_data)

		my_core.label_epoch.setText(f"Current epoch: {epoch}")
		my_core.label_step.setText(f'Current step: {step}')

	def update_accuracy(self, accuracy):
		self.params["yvalue"]["accuracy"].append(accuracy)
		step = len(self.params["yvalue"]["accuracy"])

		x_data = self.params["xrange"]["accuracy"][:step]
		y_data = self.params["yvalue"]["accuracy"]
		self.params["graph"]["accuracy"].setData(x_data, y_data)

	def update_step(self, step):
		self.params["plot"]["loss"].addItem(pyqtgraph.InfiniteLine(pos=step, pen="m"))


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
