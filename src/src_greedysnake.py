from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QFont, QIcon
from PyQt6.QtWidgets import QApplication, QMainWindow

from ui.ui_greedysnake import Ui_MainWindow
from util import util
from net.net_greedysnake import NN

from collections import deque
from itertools import product
from scipy.spatial.distance import cityblock
import numpy as np
import random
import sys
import torch


class MyCore(QMainWindow, Ui_MainWindow):
	POSITION = {(i, j): (j * 20 + 25, i * 20 + 25, 20, 20) for i, j in product(range(10), repeat=2)}

	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("../static/greedysnake/logo.png"))

		self.label_theme1 = util.mask(self.label_canvas, path="../static/greedysnake/theme1.png")
		self.label_theme2 = util.mask(self.label_canvas, path="../static/greedysnake/theme2.png", hide=True)
		self.label_pause = util.mask(self.label_canvas, color=QColor(255, 255, 255, 80), hide=True)
		util.pixmap(self.label_canvas, Qt.GlobalColor.black)

		self.model = NN().to(NN.DEVICE)
		self.model.load_state_dict(torch.load("../model/model_greedysnake.pt", map_location=NN.DEVICE))
		self.model.eval()

		self.board = self.snake = self.food = self.direction = self.recent = None
		self.timer = util.timer(50, self.timeout)
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
			util.dialog("Game over", "error")
			return self.restart()

		pixmap = util.pixmap(self.label_canvas, Qt.GlobalColor.black)
		with QPainter(pixmap) as painter:
			painter.fillRect(*self.POSITION[self.food], Qt.GlobalColor.red)
			fade = int(200 / len(self.snake))
			for i, pos in enumerate(self.snake):
				painter.fillRect(*self.POSITION[pos], QColor(0, 255, 0, 255 - fade * i))
			painter.setPen(Qt.GlobalColor.white)
			painter.setFont(QFont("", 12, QFont.Weight.Bold))
			painter.drawText(35, 50, f"Score {len(self.snake) - 3}")
		self.label_canvas.setPixmap(pixmap)


class Game:
	LARVA = {"U": (1, 0), "D": (-1, 0), "L": (0, 1), "R": (0, -1)}
	HEADING = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
	DIRECTION = {"U": "LUR", "D": "RDL", "L": "DLU", "R": "URD"}

	@staticmethod
	def initialize():
		direction, recent = random.choice("UDLR"), deque(maxlen=7)
		snake = deque([tuple(random.choices((4, 5), k=2))])
		snake.append(tuple(sum(x) for x in zip(snake[0], Game.LARVA[direction])))
		snake.append(tuple(sum(x) for x in zip(snake[0], Game.LARVA[direction], Game.LARVA[direction])))

		board = np.zeros((10, 10), dtype=np.uint8)
		board[*zip(*snake)], board[snake[0]] = 1, 2
		food, board[food]= tuple(random.choice(np.argwhere(board == 0))), 3

		return board, snake, food, direction, recent

	@staticmethod
	def act(board, snake, food, direction, recent):
		old_snake, old_food = snake.copy(), food
		recent.append(snake[0])

		head = tuple(sum(x) for x in zip(snake[0], Game.HEADING[direction]))
		if not 0 <= head[0] < 10 or not 0 <= head[1] < 10 or head in snake:
			return food, -1, 1

		snake.appendleft(head)
		board[snake[1]], board[snake[0]] = 1, 2
		if head == food:
			food, board[food] = tuple(random.choice(np.argwhere(board == 0))), 3
		else:
			board[snake.pop()] = 0

		new_snake, new_food = snake.copy(), food
		old_distance, new_distance = cityblock(old_snake[0], old_food), cityblock(new_snake[0], new_food)

		if head == food:
			reward = 1
		elif head in recent:
			reward = -0.2
		elif new_distance < old_distance:
			reward = (1 - new_distance) / 350 + 0.15
		else:
			reward = (18 - new_distance) / 350 - 0.15
		return food, round(reward, 2), 0


if __name__ == "__main__":
	app = QApplication(sys.argv)
	my_core = MyCore()
	my_core.setFixedSize(my_core.window().size())
	my_core.show()
	sys.exit(app.exec())
