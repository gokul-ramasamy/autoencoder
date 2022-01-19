import torch
from torchsummary import summary

class conv_AE(torch.nn.Module):
	def __init__(self):
		super().__init__()

		#Encoder
		self.encoder = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=1,out_channels=16,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=16,out_channels=32,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=64),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=64,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU(),
			# torch.nn.Conv2d(in_channels=256,out_channels=256,kernel_size=4),
			# torch.nn.BatchNorm2d(num_features=256),
			# torch.nn.ReLU(),
			torch.nn.Conv2d(in_channels=128,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU()
		)

		#Decoder
		self.decoder = torch.nn.Sequential(
			# torch.nn.ConvTranspose2d(in_channels=256,out_channels=256,kernel_size=4),
			# torch.nn.BatchNorm2d(num_features=256),
			# torch.nn.ReLU(),
			torch.nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU(),
			torch.nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.ConvTranspose2d(in_channels=128,out_channels=128,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=128),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=64),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=32),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.BatchNorm2d(num_features=16),
			torch.nn.ReLU(),
			torch.nn.Dropout(),
			torch.nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=4, stride=2, padding=(1,1)),
			torch.nn.Tanh()
		)

	def forward(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


