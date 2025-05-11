import torch
import torch.nn as nn


class NN(nn.Module):
	DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def __init__(self):
		super().__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, padding=1),
			nn.GroupNorm(4, 16),
			nn.ReLU(),

			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.GroupNorm(8, 32),
			nn.ReLU()
		)
		self.fc = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 10 * 10, 128),
			nn.LayerNorm(128),
			nn.ReLU(),
			nn.Dropout(0.1),
			nn.Linear(128, 3)
		)

	def forward(self, x):
		x = nn.functional.one_hot(x, 4).permute(0, 3, 1, 2)[:, 1:, :, :].float()
		x = self.conv(x)
		x = self.fc(x)
		return x
