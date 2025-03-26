import torch
import numpy as np


class N3BC:
	"""
	MLBCK.N3BC
	=========
	
	Description
	-----------
	Naive neural network bias corrector

	"""
	def __init__(self, n_features = 2, window = 60, epochs = 20):
		self.n_features = n_features
		self.window = window
		self.epochs = epochs

		self.neuralnetwork = torch.nn.Sequential(
			torch.nn.Flatten(start_dim=0),
            torch.nn.Linear(window * n_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, window * n_features),
			torch.nn.Unflatten(0, (window, n_features))
        )

		self.lossfunction = torch.nn.L1Loss()

		self.optimizer = torch.optim.Adam(self.neuralnetwork.parameters(), lr=0.001)
	
	def fit(self, Y0, X0):
		assert(isinstance(X0, np.ndarray))
		assert(len(X0.shape) == 2)
		assert(X0.shape[1] == self.n_features)
		assert(isinstance(Y0, np.ndarray))
		assert(len(Y0.shape) == 2)
		assert(Y0.shape[1] == self.n_features)
		assert(X0.shape[0] == Y0.shape[0])

		# calculate min-max values based on calibration data
		self.min_values = X0.min(axis=0)
		self.max_values = X0.max(axis=0)
		# min-max normalization
		X0 = (X0 - self.min_values) / (self.max_values - self.min_values)

		n_samples = X0.shape[0]

		self.neuralnetwork.train()

		tensor_x0 = torch.tensor(X0, dtype=torch.float32)
		tensor_yo = torch.tensor(Y0, dtype=torch.float32)

		for epoch in range(self.epochs):

			print(f"epoch: {epoch}")

			for i in range(n_samples - self.window + 1):

				self.neuralnetwork.zero_grad()

				x0 = tensor_x0[i : i + self.window, :]
				y0 = tensor_yo[i : i + self.window, :]

				z0 = self.neuralnetwork(x0)

				loss = self.lossfunction(z0, y0)

				loss.backward()

				self.optimizer.step()

			Z0 = self._predict(X0)
	
			Z = torch.tensor(Z0.transpose(), dtype=torch.float32)
			Y = torch.tensor(Y0.transpose(), dtype=torch.float32)
			loss = self.lossfunction(Z, Y)
			print(f"evaluation loss: {loss}")
	
	
	def predict(self, X1):
		assert(isinstance(X1, np.ndarray))
		assert(len(X1.shape) == 2)
		assert(X1.shape[1] == self.n_features)

		# min-max input normalization
		X1 = (X1 - self.min_values) / (self.max_values - self.min_values)

		Z1 = self._predict(X1)

		return Z1


	def _predict(self, X):
		n_samples = X.shape[0]

		self.neuralnetwork.eval()

		X = torch.tensor(X, dtype=torch.float32)

		Z_extended = torch.zeros((n_samples - self.window + 1, n_samples, self.n_features))

		for i in range(n_samples - self.window + 1):
			sample = X[i : i + self.window, :]
			output = self.neuralnetwork(sample)
			Z_extended[i, i : i + self.window, :] = output

		Z = Z_extended.mean(dim=0)

		return Z.detach().numpy()