from net.net_recognizer import NN
from torch.utils.data import DataLoader, random_split
import os
import torch
import torchvision
import torchvision.transforms as transforms

HYPERPARAMETER = {
	"learning_rate": 0.001,
	"batch_size": 64,
	"epoch": 5,
	"validate_freq": 50,
	"validate_size": 1000
}

DATASET = {
	"train": torchvision.datasets.MNIST(os.path.abspath("../dataset"), train=True, transform=transforms.ToTensor()),
	"test": torchvision.datasets.MNIST(os.path.abspath("../dataset"), train=False, transform=transforms.ToTensor())
}
DATALOADER = {
	"train": DataLoader(DATASET["train"], batch_size=HYPERPARAMETER["batch_size"], shuffle=True),
	"test": DataLoader(DATASET["test"], batch_size=HYPERPARAMETER["batch_size"], shuffle=True)
}


def train_model(features, labels, model, optimizer):
	model.train()
	tensor_features = features.to(NN.DEVICE)
	tensor_labels = labels.to(NN.DEVICE)
	tensor_predictions = model(tensor_features)
	loss = torch.nn.functional.cross_entropy(tensor_predictions, tensor_labels)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	return loss.item()


def test_model(features, labels, model):
	model.eval()
	tensor_features = features.to(NN.DEVICE)
	tensor_labels = labels.to(NN.DEVICE)
	with torch.no_grad():
		tensor_predictions = model(tensor_features)

	tensor_indices = tensor_predictions.max(dim=1).indices
	return torch.sum(tensor_indices == tensor_labels).item()


def validate_mode(model):
	rest = len(DATASET["test"]) - HYPERPARAMETER["validate_size"]
	dataset = random_split(DATASET["test"], [HYPERPARAMETER["validate_size"], rest])[0]
	dataloader = DataLoader(dataset, batch_size=HYPERPARAMETER["batch_size"], shuffle=True)
	prediction = sum(test_model(features, labels, model) for features, labels in dataloader)
	return round(100 * prediction / len(dataset), 2)


def train(signals):
	model = NN().to(NN.DEVICE)
	optimizer = torch.optim.Adam(model.parameters(), lr=HYPERPARAMETER["learning_rate"])
	step = 0

	for epoch in range(HYPERPARAMETER["epoch"]):
		for features, labels in DATALOADER["train"]:
			loss_value = train_model(features, labels, model, optimizer)
			signals["loss"].emit(loss_value, epoch + 1)

			step += 1
			if step % HYPERPARAMETER["validate_freq"] == 0:
				signals["accuracy"].emit(validate_mode(model))
		signals["step"].emit(step)
	torch.save(model.state_dict(), "model_recognizer.pt")
