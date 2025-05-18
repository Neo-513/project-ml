from collections import deque
from scipy.spatial.distance import cityblock
import numpy as np
import random


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
		food, board[food] = tuple(random.choice(np.argwhere(board == 0))), 3

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
