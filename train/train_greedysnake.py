from collections import deque
from net.net_greedysnake import NN
from game.game_greedysnake import Game
import numpy as np
import random
import torch


class Trainer:
	HPARAM = {
		"lr": 0.001,
		"bs": 64,
		"episode": 800,
		"replay_buffer": 10000,
		"initial_buffer": 1000,
		"step": 30,
		"gamma": 0.99
	}

	@staticmethod
	def train_model(model, epsilon, step):
		model.eval()
		board, snake, food, direction, recent = Game.initialize()
		transitions = []

		for i in range(step):
			*transition, food, direction = Trainer.self_play(model, epsilon, board, snake, food, direction, recent)
			transitions.append(transition)
			if transition[-1]:
				board, snake, food, direction, recent = Game.initialize()
		return transitions

	@staticmethod
	def test_model(model, epsilon):
		model.eval()
		board, snake, food, direction, recent = Game.initialize()
		done = rewards = steps = 0

		while not done and rewards >= -30 and steps <= 300:
			_, _, reward, _, done, food, direction = Trainer.self_play(model, epsilon, board, snake, food, direction, recent)
			rewards += reward
			steps += 1
		return rewards

	@staticmethod
	def self_play(model, epsilon, board, snake, food, direction, recent):
		state = board.copy()
		if random.random() < epsilon:
			action = random.choice(range(3))
		else:
			tensor_states = torch.tensor(np.array([board])).long().to(NN.DEVICE)
			with torch.no_grad():
				q_values = model(tensor_states)
			action = q_values.argmax().item()

		direction = Game.DIRECTION[direction][action]
		food, reward, done = Game.act(board, snake, food, direction, recent)
		next_state = board.copy()
		return state, action, reward, next_state, done, food, direction

	@staticmethod
	def dqn(online_model, target_model, optimizer, batch):
		online_model.train()
		target_model.eval()

		states, actions, rewards, next_states, dones = zip(*batch)
		tensor_states = torch.tensor(np.array(states)).long().to(NN.DEVICE)
		tensor_actions = torch.tensor(actions).long().unsqueeze(1).to(NN.DEVICE)
		tensor_rewards = torch.tensor(rewards).float().unsqueeze(1).to(NN.DEVICE)
		tensor_next_states = torch.tensor(np.array(next_states)).long().to(NN.DEVICE)
		tensor_dones = torch.tensor(dones).float().unsqueeze(1).to(NN.DEVICE)

		current_q = online_model(tensor_states).gather(1, tensor_actions)
		with torch.no_grad():
			next_q = target_model(tensor_next_states).max(1, keepdim=True)[0]
			target_q = tensor_rewards + Trainer.HPARAM["gamma"] * (1 - tensor_dones) * next_q
		loss = torch.nn.functional.smooth_l1_loss(current_q, target_q.detach())

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(online_model.parameters(), max_norm=1.0)
		optimizer.step()
		return loss.item()

	@staticmethod
	def train(signals):
		online_model = NN().to(NN.DEVICE)
		optimizer = torch.optim.Adam(online_model.parameters(), lr=Trainer.HPARAM["lr"])

		target_model = NN().to(NN.DEVICE)
		target_model.load_state_dict(online_model.state_dict())

		replay_buffer = deque(maxlen=Trainer.HPARAM["replay_buffer"])
		replay_buffer.extend(Trainer.train_model(online_model, 1, Trainer.HPARAM["initial_buffer"]))

		for episode in range(Trainer.HPARAM["episode"]):
			epsilon = 1 - 0.99 * episode / Trainer.HPARAM["episode"]
			replay_buffer.extend(Trainer.train_model(online_model, epsilon, 100))

			loss_values = []
			for s in range(Trainer.HPARAM["step"]):
				batch = random.sample(replay_buffer, Trainer.HPARAM["bs"])
				loss_values.append(Trainer.dqn(online_model, target_model, optimizer, batch))

				if (episode * Trainer.HPARAM["step"] + s + 1) % 1000 == 0:
					target_model.load_state_dict(online_model.state_dict())
					signals["section"].emit(episode)

			loss_value = sum(loss_values) / len(loss_values)
			reward = Trainer.test_model(online_model, 0)
			signals["loss"].emit(loss_value, episode + 1)
			signals["reward"].emit(reward, episode + 1)

		torch.save(online_model.state_dict(), "model_greedysnake.pt")
