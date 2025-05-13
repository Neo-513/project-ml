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
		
		xranges = {
			"loss": util.xrange(HYPERPARAMETER["episode"]),
			"reward": util.xrange(HYPERPARAMETER["episode"])
		}
		plots = {
			"loss": util.plot("Loss", self.widget_loss, xrng=(0, len(xranges["loss"]))),
			"reward": util.plot("Reward", self.widget_reward, xrng=(0, len(xranges["reward"])))
		}
		graphs = {
			"loss": plots["loss"].plot([], [], pen="r"),
			"reward": BarGraphItem(x=[], height=[], pen=None, brush="y", width=1)
		}
		
		self.timer = util.timer(1000, lambda: util.record_time(self.timer, self.label_timer))
		self.timer.second = 0
		self.timer.start()

		self.my_thread = MyThread({
			"xrange": xranges,
			"yvalue": {"loss": [], "reward": []},
			"plot": plots,
			"graph": graphs,
			"indicator": pyqtgraph.InfiniteLine(pen="g"),
			"label": {"episode": self.label_episode},
			"timer": self.timer
		})
		self.my_thread.start()


class MyThread(QThread):
	signal_loss = pyqtSignal(float, int)
	signal_reward = pyqtSignal(float, int)
	signal_section = pyqtSignal(int)

	def __init__(self, params):
		super().__init__()
		util.cast(self.signal_loss).connect(self.update_loss)
		util.cast(self.signal_reward).connect(self.update_reward)
		util.cast(self.signal_section).connect(self.update_section)
		
		self.params = params
		self.params["plot"]["loss"].addItem(self.params["indicator"])
		self.params["plot"]["reward"].addItem(self.params["graph"]["reward"])

	def run(self):
		train({
			"loss": self.signal_loss,
			"reward": self.signal_reward,
			"section": self.signal_section
		})
		
		self.params["timer"].stop()
		self.params["plot"]["loss"].removeItem(self.params["indicator"])

	def update_loss(self, loss, episode):
		self.params["yvalue"]["loss"].append(loss)
		self.params["indicator"].setPos(episode)
		self.params["label"]["episode"].setText(f"Current episode: {episode}")

		data = self.params["xrange"]["loss"][:episode], self.params["yvalue"]["loss"]
		self.params["graph"]["loss"].setData(*data)

	def update_reward(self, reward, episode):
		self.params["yvalue"]["reward"].append(reward)

		x_data = self.params["xrange"]["reward"][:episode]
		y_data = self.params["yvalue"]["reward"]
		self.params["graph"]["reward"].setOpts(x=x_data, height=y_data)

	def update_section(self, episode):
		self.params["plot"]["loss"].addItem(pyqtgraph.InfiniteLine(pos=episode, pen="m"))


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
