from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow

from vis.vis_greedysnake_ui import Ui_MainWindow
from util import util

from pyqtgraph import BarGraphItem
from train.train_greedysnake import HYPERPARAMETER, train
import pyqtgraph
import sys


class MyCore(QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/greedysnake/logo.png"))

		plot_widget1 = pyqtgraph.PlotWidget(title="Loss Value")
		plot_widget1.setMouseEnabled(x=False, y=False)
		plot_widget1.getPlotItem().hideButtons()
		plot_widget1.setXRange(0, HYPERPARAMETER["episode"])
		self.centralwidget.layout().addWidget(plot_widget1)

		plot_widget2 = pyqtgraph.PlotWidget(title="Reward")
		plot_widget2.setMouseEnabled(x=False, y=False)
		plot_widget2.getPlotItem().hideButtons()
		plot_widget2.setXRange(0, HYPERPARAMETER["episode"])
		self.centralwidget.layout().addWidget(plot_widget2)

		self.timer = util.timer(1000, self.record_time)
		self.timer.second = 0
		self.timer.start()

		self.my_thread = MyThread(self.timer, plot_widget1, plot_widget2)
		self.my_thread.start()

	def record_time(self):
		self.timer.second += 1
		h, m, s = self.timer.second // 3600, (self.timer.second // 60) % 60, self.timer.second % 60
		self.label_time.setText(f"Time spent: {h:02}:{m:02}:{s:02}")


class MyThread(QThread):
	signal_update1 = pyqtSignal(float)
	signal_update2 = pyqtSignal(float)
	signal_update3 = pyqtSignal(int)

	def __init__(self, timer, plot_widget1, plot_widget2):
		super().__init__()
		util.cast(self.signal_update1).connect(self.update1)
		util.cast(self.signal_update2).connect(self.update2)
		util.cast(self.signal_update3).connect(self.update3)
		
		self.timer = timer
		self.plot_widget1 = plot_widget1
		self.plot_widget2 = plot_widget2

		self.loss_values = []
		self.rewards = []
		self.EPISODE = tuple(range(1, HYPERPARAMETER["episode"] + 1))

		self.curve1 = self.plot_widget1.plot([], [], pen="r")
		self.curve2 = BarGraphItem(x=[], height=[], pen=None, brush="y", width=1)
		self.curve3 = pyqtgraph.InfiniteLine(pen="g")

		self.plot_widget1.addItem(self.curve3)
		self.plot_widget2.addItem(self.curve2)

	def run(self):
		train({
			"loss_value": self.signal_update1,
			"reward": self.signal_update2,
			"episode": self.signal_update3
		})
		self.timer.stop()

	def update1(self, loss_value):
		self.loss_values.append(loss_value)
		self.curve1.setData(self.EPISODE[:len(self.loss_values)], self.loss_values)
		self.curve3.setPos(len(self.loss_values))
		my_core.label_episode.setText(f"Current episode: {len(self.loss_values)}")

	def update2(self, reward):
		self.rewards.append(reward)
		self.curve2.setOpts(x=self.EPISODE[:len(self.rewards)], height=self.rewards)

	def update3(self, episode):
		self.plot_widget1.addItem(pyqtgraph.InfiniteLine(pos=episode, pen="m"))


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
