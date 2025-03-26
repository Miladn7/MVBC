import torch
import numpy as np
import matplotlib.pyplot as plt

class ConvBC:
	"""
	MLBCK.ConvBC
	=========
	
	Description
	-----------
	Convolutional neural network bias corrector

	"""
	def __init__(self, n_features = 2, window_size = 90, batch_size = 64, epochs = 200):
		self.n_features = n_features
		self.window_size = window_size
		self.batch_size = batch_size
		self.epochs = epochs

		self.encoder = torch.nn.Sequential(
			torch.nn.Conv1d(in_channels=n_features, out_channels=8, kernel_size=3, stride=1, padding=1),
			torch.nn.LeakyReLU(0.01),
			torch.nn.Conv1d(in_channels=8, out_channels=32, kernel_size=5, stride=2, padding=2),
			torch.nn.LeakyReLU(0.01),
			torch.nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=4, padding=4),
			torch.nn.LeakyReLU(0.01),
			torch.nn.Flatten(),
            torch.nn.Linear(768, 96)
		)
		
		self.decoder = torch.nn.Sequential(
			torch.nn.Linear(96, 768),
			torch.nn.Unflatten(1, (64, 12)),
			torch.nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=7, stride=4, padding=4, output_padding=2),
			torch.nn.LeakyReLU(0.01),
			torch.nn.ConvTranspose1d(in_channels=32, out_channels=8, kernel_size=5, stride=2, padding=2, output_padding=1),
            torch.nn.LeakyReLU(0.01),
			torch.nn.ConvTranspose1d(in_channels=8, out_channels=n_features, kernel_size=3, padding=1, stride=1),
            torch.nn.LeakyReLU(0.01),
		)

		self.neuralnetwork = torch.nn.Sequential(self.encoder, self.decoder)

		self.lossfunction = torch.nn.L1Loss()

		self.optimizer = torch.optim.Adam(self.neuralnetwork.parameters(), lr=0.01)


	def fit(self, Y0, X0):
		assert(isinstance(X0, np.ndarray))
		assert(len(X0.shape) == 2)
		assert(X0.shape[1] == self.n_features)
		assert(isinstance(Y0, np.ndarray))
		assert(len(Y0.shape) == 2)
		assert(Y0.shape[1] == self.n_features)
		assert(X0.shape[0] == Y0.shape[0])

		# calculate min-max values based on calibration input data
		self.min_values = X0.min(axis=0)
		self.max_values = X0.max(axis=0)

		# min-max input normalization
		X0 = (X0 - self.min_values) / (self.max_values - self.min_values)

		n_samples = X0.shape[0] - self.window_size + 1

		Y0_extended = np.zeros((n_samples, self.n_features, self.window_size))
		X0_extended = np.zeros((n_samples, self.n_features, self.window_size))

		for i in range(n_samples):
			Y0_extended[i, :, :] = Y0[i : i + self.window_size, :].transpose()
			X0_extended[i, :, :] = X0[i : i + self.window_size, :].transpose()

		self.neuralnetwork.train()

		eval_losses = []

		for epoch in range(self.epochs):

			for i in range(n_samples // self.batch_size):

				self.neuralnetwork.zero_grad()

				# sample a random batch
				idx = np.random.choice(n_samples, self.batch_size)
				x0 = torch.tensor(X0_extended[idx, :, :], dtype=torch.float32)
				y0 = torch.tensor(Y0_extended[idx, :, :], dtype=torch.float32)

				z0 = self.neuralnetwork(x0)

				loss = self.lossfunction(z0, y0)

				loss.backward()

				self.optimizer.step()

			print(f"epoch: {epoch}")
			
			Z0 = self._predict(X0)
	
			Z = torch.tensor(Z0.transpose(), dtype=torch.float32)
			Y = torch.tensor(Y0.transpose(), dtype=torch.float32)
			eval_loss = self.lossfunction(Z, Y)
			print(f"evaluation loss: {eval_loss}")
			eval_losses.append(eval_loss)
	
		plt.plot(eval_losses)
		plt.show()

	def predict(self, X1):
		assert(isinstance(X1, np.ndarray))
		assert(len(X1.shape) == 2)
		assert(X1.shape[1] == self.n_features)

		# min-max input normalization
		X1 = (X1 - self.min_values) / (self.max_values - self.min_values)

		Z1 = self._predict(X1)

		return Z1


	def _predict(self, X):
		n_samples = X.shape[0] - self.window_size + 1

		self.neuralnetwork.eval()

		X_extended = np.zeros((n_samples, self.n_features, self.window_size))
		Z_extended = np.zeros((n_samples,) + X.shape)

		for i in range(n_samples):
			X_extended[i, :, :] = X[i : i + self.window_size, :].transpose()

		X_tensor = torch.tensor(X_extended, dtype=torch.float32)
		Z_tensor = self.neuralnetwork(X_tensor).detach().numpy()

		for i in range(n_samples):
			Z_extended[i, i : i + self.window_size, :] = Z_tensor[i, :, :].transpose()

		Z = Z_extended.mean(axis=0)

		return Z

