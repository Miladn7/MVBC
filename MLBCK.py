import os
import torch
import numpy as np
import sklearn.model_selection

from models import FCNeuralMVBC, ConvNeuralMVBC, LSTMNeuralMVBC


class N3BC:
	"""
	MLBCK.N3BC
	=========
	
	Description
	-----------
	Naive neural network bias corrector

	"""
	def __init__(self, neuralnetwork):
		self.model = neuralnetwork
		self.lossfunction = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
		
		self.kfold = sklearn.model_selection.KFold(n_splits=5)

		self.name = self.model.__class__.__name__
	
	
	def fit(self, y_calibration, x_calibration):
		x_calibration = x_calibration.reshape(-1, 365, 2)
		y_calibration = y_calibration.reshape(-1, 365, 2)

		# calculate min-max values based on calibration data
		self.min_values = x_calibration.min(axis=(0, 1))
		self.max_values = x_calibration.max(axis=(0, 1))
		# min-max normalization of all predictions
		x_calibration = (x_calibration - self.min_values) / (self.max_values - self.min_values)
		# min-max normalization of all observations
		y_calibration = (y_calibration - self.min_values) / (self.max_values - self.min_values)
		
		fold_losses = []

		for fold, (train_ids, test_ids) in enumerate(self.kfold.split(x_calibration)):

			self.model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)

			x_train = x_calibration[train_ids, :, :]
			x_test = x_calibration[test_ids, :, :]
			y_train = y_calibration[train_ids, :, :]
			y_test = y_calibration[test_ids, :, :]

			train_losses = []
			test_losses = []

			for epoch in range(1000):

				self.model.train()
					
				self.model.zero_grad()

				# sample a random batch
				idx = np.random.choice(len(x_train), len(x_test))
				x_train_sample = torch.tensor(x_train[idx, :, :], dtype=torch.float32)
				y_train_sample = torch.tensor(y_train[idx, :, :], dtype=torch.float32)

				z_train_sample = self.model(x_train_sample)

				train_loss = self.lossfunction(z_train_sample, y_train_sample)

				train_loss.backward()

				self.optimizer.step()

				if (epoch % 10 == 0):

					self.model.eval()

					x_test_sample = torch.tensor(x_test, dtype=torch.float32)
					y_test_sample = torch.tensor(y_test, dtype=torch.float32)

					z_test_sample = self.model(x_test_sample)

					test_loss = self.lossfunction(z_test_sample, y_test_sample)

					# print(f"fold: {fold} - epoch: {epoch} - test loss: {test_loss}")

					train_losses.append(train_loss.item())
					test_losses.append(test_loss.item())

					if test_loss <= min(test_losses):
						torch.save(self.model.state_dict(), os.path.join('checkpoints', f'N3BC_{self.name}' + str(fold)))

			min_test_loss = min(test_losses)
			# print(f"fold: {fold} - minimum test loss: {min_test_loss}")

			fold_losses.append(min_test_loss)

			if min_test_loss <= min(fold_losses):
				modelparameters = torch.load(os.path.join('checkpoints', f'N3BC_{self.name}' + str(fold)), weights_only=True)
				torch.save(modelparameters, os.path.join('checkpoints', f'N3BC_{self.name}'))

		# average_min_test_loss = sum(fold_losses) / len(fold_losses)
		# print(f"average minimum test loss: {average_min_test_loss}")

		# overall_min_test_loss_fold, overall_min_test_loss = min(enumerate(fold_losses), key=lambda l: l[1])
		# print(f"overall minimum test loss: {overall_min_test_loss} (fold: {overall_min_test_loss_fold})")
	
	
	def predict(self, x_validation):
		x_validation = x_validation.reshape(-1, 365, 2)

		# min-max normalization of all predictions
		x_validation = (x_validation - self.min_values) / ( self.max_values -  self.min_values)

		self.model.load_state_dict(torch.load(os.path.join('checkpoints', f'N3BC_{self.name}'), weights_only=True))
		
		self.model.eval()

		x_validation_sample = torch.tensor(x_validation, dtype=torch.float32)

		z_validation_sample = self.model(x_validation_sample)

		z_validation = z_validation_sample.detach().numpy()

		# min-max de-normalization
		z_validation = self.min_values + z_validation * (self.max_values - self.min_values)

		return z_validation.reshape(-1, 2)
