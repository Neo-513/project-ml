from PyQt6.QtCore import pyqtSignal, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow
from pyqtgraph import BarGraphItem
from trainer.trainer_greedysnake import Trainer
from util import util_plot, util_ui
from vis.vis_greedysnake_ui import Ui_MainWindow
import pyqtgraph
import sys


class MyCore(QMainWindow, Ui_MainWindow):
	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/greedysnake/logo.png"))
		
		self.value_loss, self.value_reward = [], []
		self.indicator = pyqtgraph.InfiniteLine(pen="g")
		self.timer = util_plot.clock(self.label_timer)

		self.xrange_loss = util_plot.xrange(Trainer.HPARAM["episode"])
		self.xrange_reward = util_plot.xrange(Trainer.HPARAM["episode"])
		self.plot_loss = util_plot.plot("Loss", self.widget_loss, x=(0, len(self.xrange_loss)))
		self.plot_reward = util_plot.plot("Reward", self.widget_reward, x=(0, len(self.xrange_reward)))
		self.graph_loss = self.plot_loss.plot([], [], pen="r")
		self.graph_reward = BarGraphItem(x=[], height=[], pen=None, brush="y", width=1)
		
		self.my_thread = MyThread()
		util_ui.cast(self.my_thread.signal_start).connect(self.thread_start)
		util_ui.cast(self.my_thread.signal_loss).connect(self.thread_loss)
		util_ui.cast(self.my_thread.signal_reward).connect(self.thread_reward)
		util_ui.cast(self.my_thread.signal_section).connect(self.thread_section)
		util_ui.cast(self.my_thread.signal_finish).connect(self.thread_finish)
		self.my_thread.start()
	
	def thread_start(self):
		self.plot_loss.addItem(self.indicator)
		self.plot_reward.addItem(self.graph_reward)
		self.timer.start()

	def thread_loss(self, loss, episode):
		self.value_loss.append(loss)
		self.graph_loss.setData(self.xrange_loss[:episode], self.value_loss)
		self.indicator.setPos(episode)
		self.label_episode.setText(f"Current episode: {episode}")

	def thread_reward(self, reward, episode):
		self.value_reward.append(reward)
		self.graph_reward.setOpts(x=self.xrange_reward[:episode], height=self.value_reward)

	def thread_section(self, episode):
		self.plot_loss.addItem(pyqtgraph.InfiniteLine(pos=episode, pen="m"))

	def thread_finish(self):
		self.timer.stop()
		self.plot_loss.removeItem(self.indicator)


class MyThread(QThread):
	signal_start = pyqtSignal()
	signal_loss = pyqtSignal(float, int)
	signal_reward = pyqtSignal(float, int)
	signal_section = pyqtSignal(int)
	signal_finish = pyqtSignal()

	def run(self):
		util_ui.cast(self.signal_start).emit()
		Trainer.train({"loss": self.signal_loss, "reward": self.signal_reward, "section": self.signal_section})
		util_ui.cast(self.signal_finish).emit()


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.show()
	sys.exit(app.exec())
