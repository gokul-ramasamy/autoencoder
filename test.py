import torch
from model import conv_AE
from matplotlib import pyplot as plt
import numpy as np
# from datagen import train_dataloader as train_loader
from datagen import test_dataloader as test_loader
import argparse


#Argument parser
parser = argparse.ArgumentParser(description = "Evaluating a model from a saved checkpoint")
parser.add_argument('--file_path', type=str, required=False)

args = parser.parse_args()

#Extracting the saved model path
if args.file_path is not None:
	SAVED_CHECKPOINT_PATH = args.file_path
else:
	raise ValueError('Model Path missing')

#use gpu if available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#Initializing the model
# Model Initialization
# model = AE().to(device)
model = conv_AE().to(device)


# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(model.parameters(),
							lr = 1e-1,
							weight_decay = 1e-8)

checkpoint = torch.load(SAVED_CHECKPOINT_PATH, map_location='cuda:2')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']


for a_batch in test_loader:
	image = a_batch['image']
	plot_img = a_batch['image']
	image = image.float()
	image = image.to(device)
	image = image[:,None,:,:]
	reconstructed = model(image)

	output = reconstructed[:,0,:,:].detach().cpu()


	for i in range(len(output)):
		f, arr = plt.subplots(1,2)
		arr[0].imshow(plot_img[i], cmap='gray')
		arr[1].imshow(output[i], cmap='gray')
		plt.show()



	
#Plotting the reconstructed images
# for i, item in enumerate(reconstructed):
# 	plot_item = item.detach().cpu().numpy()
# 	plot_item = plot_item.reshape(-1, 256, 256)
# 	# arr[0].imshow(plot_item[1])
# 	print(plot_item.shape)
# 	plt.imshow(plot_item[0], cmap='gray')
# 	plt.show()
# 	break

# model.eval()
