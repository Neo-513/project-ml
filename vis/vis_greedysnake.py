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

		self.my_thread = MyThread(plot_widget1, plot_widget2)
		self.my_thread.timer = self.timer
		self.my_thread.start()

	def record_time(self):
		self.timer.second += 1
		h, m, s = self.timer.second // 3600, (self.timer.second // 60) % 60, self.timer.second % 60
		self.label_time.setText(f"Time spent: {h:02}:{m:02}:{s:02}")


class MyThread(QThread):
	signal_update1 = pyqtSignal(float, float)
	signal_update2 = pyqtSignal(int)

	def __init__(self, plot_widget1, plot_widget2):
		super().__init__()
		util.cast(self.signal_update1).connect(self.update1)
		util.cast(self.signal_update2).connect(self.update2)

		self.loss_values = [0]
		self.rewards = []
		self.EPISODE = tuple(range(HYPERPARAMETER["episode"]))

		self.plot_widget1 = plot_widget1
		self.plot_widget2 = plot_widget2

		self.curve1 = self.plot_widget1.plot([], [], pen="r")
		self.curve2 = BarGraphItem(x=[], height=[], pen=None, brush="y", width=1)
		self.curve3 = pyqtgraph.InfiniteLine(pen="g")

		self.plot_widget1.addItem(self.curve3)
		self.plot_widget2.addItem(self.curve2)

	def run(self):
		train({"update1": self.signal_update1, "update2": self.signal_update2})
		my_core.timer.stop()

	def update1(self, loss_value, reward):
		self.loss_values.append(loss_value)
		self.rewards.append(reward)

		episode = len(self.rewards)
		my_core.label_episode.setText(f"Current episode: {episode}")

		self.curve1.setData(self.loss_values)
		self.curve2.setOpts(x=self.EPISODE[:episode], height=self.rewards)
		self.curve3.setPos(episode)

	def update2(self, episode):
		self.plot_widget1.addItem(pyqtgraph.InfiniteLine(pos=episode, pen="b"))


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
