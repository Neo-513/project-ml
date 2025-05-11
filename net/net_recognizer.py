import torch
import torch.nn as nn


class NN(nn.Module):
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def __init__(self):
		super().__init__()
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(28 * 28, 512),
			nn.ReLU(),
			nn.Linear(512, 10)
		)

	def forward(self, x):
		return self.fc(x)
