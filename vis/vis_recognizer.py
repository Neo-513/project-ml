from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow
from train.train_recognizer import Trainer
from util import util_plot, util_ui
from vis.vis_recognizer_ui import Ui_MainWindow
import math
import pyqtgraph
import sys


class MyCore(QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/recognizer/logo.png"))

		steps = math.ceil(len(Trainer.DATASET["train"]) / Trainer.HPARAM["bs"]) * Trainer.HPARAM["epoch"]
		self.value_loss, self.value_accuracy = [], []
		self.indicator = pyqtgraph.InfiniteLine(pen="g")
		self.timer = util_plot.clock(self.label_timer)

		self.xrange_loss = util_plot.xrange(steps)
		self.xrange_accuracy = util_plot.xrange(steps // Trainer.HPARAM["validate_freq"])
		self.plot_loss = util_plot.plot("Loss", self.widget_loss, x=(0, len(self.xrange_loss)))
		self.plot_accuracy = util_plot.plot("Accuracy", self.widget_accuracy, x=(0, len(self.xrange_accuracy)), y=(85, 100))
		self.graph_loss = self.plot_loss.plot([], [], pen="r")
		self.graph_accuracy = self.plot_accuracy.plot([], [], pen="y", symbolBrush="y", symbolPen="y", symbol="o", symbolSize=3)

		self.thread = Thread()
		util_ui.cast(self.thread.signal_start).connect(self.thread_start)
		util_ui.cast(self.thread.signal_loss).connect(self.thread_loss)
		util_ui.cast(self.thread.signal_accuracy).connect(self.thread_accuracy)
		util_ui.cast(self.thread.signal_section).connect(self.thread_section)
		util_ui.cast(self.thread.signal_finish).connect(self.thread_finish)
		self.thread.start()

	def thread_start(self):
		self.plot_loss.addItem(self.indicator)
		self.timer.start()

	def thread_loss(self, loss, epoch, step):
		self.value_loss.append(loss)
		self.graph_loss.setData(self.xrange_loss[:step], self.value_loss)
		self.indicator.setPos(step)
		self.label_epoch.setText(f"Current epoch: {epoch}")
		self.label_step.setText(f"Current step: {step}")

	def thread_accuracy(self, accuracy):
		self.value_accuracy.append(accuracy)
		self.graph_accuracy.setData(self.xrange_accuracy[:len(self.value_accuracy)], self.value_accuracy)

	def thread_section(self, step):
		self.plot_loss.addItem(pyqtgraph.InfiniteLine(pos=step, pen="m"))

	def thread_finish(self):
		self.timer.stop()
		self.plot_loss.removeItem(self.indicator)


class Thread(QThread):
	signal_start = pyqtSignal()
	signal_loss = pyqtSignal(float, int, int)
	signal_accuracy = pyqtSignal(float)
	signal_section = pyqtSignal(int)
	signal_finish = pyqtSignal()

	def run(self):
		util_ui.cast(self.signal_start).emit()
		Trainer.train({"loss": self.signal_loss, "accuracy": self.signal_accuracy, "section": self.signal_section})
		util_ui.cast(self.signal_finish).emit()


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
