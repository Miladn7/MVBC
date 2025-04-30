import torch

from torch.nn import Conv1d, LSTM, Linear
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class ConvLSTMNeuralMVBC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.lstm = LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = Linear(32, 2)  # Output layer

    def forward(self, x):
        x = x.transpose(1, 2)  # Convert to (batch, channels, time)
        x = self.conv1(x)
        x = x.transpose(1, 2)  # Convert back to (batch, time, features)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


class LSTMNeuralMVBC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(64, 2)  # Output layer (same dimension as input)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # LSTM processes the sequence
        return self.fc(lstm_out)  # Fully connected layer for bias correction

class FCNeuralMVBC(torch.nn.Module):
    def __init__(self):
        super().__init__()

        lineartransformations = [128, 64, 32, 16]

        encoderlayers = []
        decoderlayers = []

        in_size = 365 * 2

        for out_size in lineartransformations:

            # print((in_size, out_size))

            encoderlayers.append(torch.nn.Linear(in_size, out_size))
            encoderlayers.append(torch.nn.ReLU())

            decoderlayers.append(torch.nn.ReLU())
            decoderlayers.append(torch.nn.Linear(out_size, in_size))

            in_size = out_size

        decoderlayers.reverse()
        decoderlayers.pop()
			
        self.encoder = torch.nn.Sequential(*encoderlayers)
        
        self.decoder = torch.nn.Sequential(*decoderlayers)

        self.neuralnetwork = torch.nn.Sequential(self.encoder, self.decoder)
        
        self.neuralnetwork = torch.nn.Sequential(
            torch.nn.Linear(365 * 2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 365 * 2),
        )

        # print(self.neuralnetwork)
	
    def forward(self, x):
        x = x.transpose(1, 2).flatten(1)
        y = self.neuralnetwork(x)
        y = y.unflatten(1, (2, 365)).transpose(2, 1)
        return y
    

class ConvNeuralMVBC(torch.nn.Module):
    def __init__(self):
        super().__init__()

        convolutions = [(4, 3, 1, 1), (16, 5, 2, 2), (32, 9, 4, 4), (32, 11, 5, 5)]

        encoderlayers = []
        decoderlayers = []

        in_size = 365
        in_channels = 2
        
        for (out_channels, kernel_size, stride, padding) in convolutions:

            out_size = (in_size - kernel_size + 2 * padding) // stride + 1

            encoderlayers.append(torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            encoderlayers.append(torch.nn.LeakyReLU(0.01))

            output_padding = in_size - (out_size - 1) * stride + 2 * padding - kernel_size
            print((in_size, out_size, output_padding))

            decoderlayers.append(torch.nn.LeakyReLU(0.01))
            decoderlayers.append(torch.nn.ConvTranspose1d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding))

            in_size = out_size
            in_channels = out_channels
			
        # feature_dim = in_size * in_channels

        # encoderlayers.append(torch.nn.Flatten())
        # encoderlayers.append(torch.nn.Linear(feature_dim, feature_dim // 4))
        # encoderlayers.append(torch.nn.LeakyReLU(0.01))

        # decoderlayers.append(torch.nn.Unflatten(1, (in_channels, in_size)))
        # decoderlayers.append(torch.nn.LeakyReLU(0.01))
        # decoderlayers.append(torch.nn.Linear(feature_dim // 4, feature_dim))

        decoderlayers.reverse()
        decoderlayers.pop()

        self.encoder = torch.nn.Sequential(*encoderlayers)
        self.decoder = torch.nn.Sequential(*decoderlayers)

        self.neuralnetwork = torch.nn.Sequential(self.encoder, self.decoder)
        
        print(self.neuralnetwork)
	
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.neuralnetwork(x)
        y = y.transpose(2, 1)
        return y
