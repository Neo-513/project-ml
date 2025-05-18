from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow
from itertools import product
from game.game_greedysnake import Game
from net.net_greedysnake import NN
from src.src_greedysnake_ui import Ui_MainWindow
from util import util_ui
import numpy as np
import sys
import torch


class MyCore(QMainWindow, Ui_MainWindow):
	POSITION = {(i, j): (j * 20 + 25, i * 20 + 25, 20, 20) for i, j in product(range(10), repeat=2)}

	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/greedysnake/logo.png"))

		self.label_theme1 = util_ui.mask(self.label_canvas, path="../static/greedysnake/theme1.png")
		self.label_theme2 = util_ui.mask(self.label_canvas, path="../static/greedysnake/theme2.png", hide=True)
		self.label_pause = util_ui.mask(self.label_canvas, color=QColor(255, 255, 255, 80), hide=True)
		util_ui.pixmap(self.label_canvas, Qt.GlobalColor.black)

		self.model = NN().to(NN.DEVICE)
		self.model.load_state_dict(torch.load("../model/model_greedysnake.pt", map_location=NN.DEVICE))
		self.model.eval()

		self.board = self.snake = self.food = self.direction = self.recent = None
		self.timer = util_ui.timer(50, self.timeout)
		self.restart()

	def restart(self):
		self.board, self.snake, self.food, self.direction, self.recent = Game.initialize()
		self.timer.start()

	def mousePressEvent(self, event):
		if event.button() == Qt.MouseButton.RightButton:
			self.label_theme1.setHidden(not self.label_theme1.isHidden())
			self.label_theme2.setHidden(not self.label_theme2.isHidden())

	def keyPressEvent(self, event):
		if event.key() == Qt.Key.Key_Space:
			if self.timer.isActive():
				self.timer.stop()
				self.label_pause.show()
			else:
				self.timer.start()
				self.label_pause.hide()
		if event.key() == Qt.Key.Key_Return:
			if self.timer.isActive():
				self.restart()

	def timeout(self):
		tensor_states = torch.tensor(np.array([self.board])).long().to(NN.DEVICE)
		with torch.no_grad():
			q_values = self.model(tensor_states)
		action = q_values.argmax().item()

		self.direction = Game.DIRECTION[self.direction][action]
		self.food, _, done = Game.act(self.board, self.snake, self.food, self.direction, self.recent)
		if done:
			util_ui.dialog("Game over", "error")
			return self.restart()

		pixmap = util_ui.pixmap(self.label_canvas, Qt.GlobalColor.black)
		with QPainter(pixmap) as painter:
			painter.fillRect(*self.POSITION[self.food], Qt.GlobalColor.red)
			fade = int(200 / len(self.snake))
			for i, pos in enumerate(self.snake):
				painter.fillRect(*self.POSITION[pos], QColor(0, 255, 0, 255 - fade * i))
			painter.setPen(Qt.GlobalColor.white)
			painter.setFont(QFont("", 12, QFont.Weight.Bold))
			painter.drawText(35, 50, f"Score {len(self.snake) - 3}")
		self.label_canvas.setPixmap(pixmap)


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.setFixedSize(my_core.window().size())
	my_core.show()
	sys.exit(app.exec())
