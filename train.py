#Importing the necessary modules
import torch
from torchvision import datasets
from torchvision import transforms
from datagen import train_dataloader as train_loader
from datagen import test_dataloader as test_loader
from matplotlib import pyplot as plt
import datetime
from model import conv_AE
import os
import time
import gc
from torchinfo import summary
from json_parser import MODEL_SAVE_PATH, RESULTS_PATH
import argparse


#Argument parser
parser = argparse.ArgumentParser(description = "Training a model from a saved checkpoint")
parser.add_argument('--file_path', type=str, required=False)

args = parser.parse_args()

#use gpu if available
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#Initializing the model
# Model Initialization
model = conv_AE()
#Parallelizing the model
# model = torch.nn.DataParallel(model)
#Pushing the model to GPU
model = model.to(device)
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()

# Using an Adam Optimizer with lr = 0.1   ---> Try changing the learning parameter
optimizer = torch.optim.Adam(model.parameters(),
							lr = 1e-1,
							weight_decay = 1e-8)

#Loading the saved checkpoints
if args.file_path is not None:
	SAVED_CHECKPOINT_PATH = args.file_path

	checkpoint = torch.load(SAVED_CHECKPOINT_PATH, map_location = "cuda:3")
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']

	#Loss writing text file name
	loss_text = 'loss_'+ SAVED_CHECKPOINT_PATH.split('/')[-2]+'.txt'
	#Extracting the previously saved checkpoint path
	CHECKPOINT_PATH = SAVED_CHECKPOINT_PATH.split('/')[-2] + '/'
	#Extracting the epoch number from the saved checkpoint
	save_count = int(SAVED_CHECKPOINT_PATH.split('/')[-1].split('.')[0].split('_')[-1]) 

#Starting the model training from scratch
else:
	#Loss writing text file name
	loss_text = 'loss_'+datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'.txt'
	with open(RESULTS_PATH+loss_text, 'w+'):
		pass

	#Checkpoints write-path
	CHECKPOINT_PATH = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'/'  #################
	if not os.path.isdir(MODEL_SAVE_PATH+CHECKPOINT_PATH):
		os.mkdir(MODEL_SAVE_PATH+CHECKPOINT_PATH)
	
	save_count = 0
	

#Output Generation
epochs = 1000
outputs = []
losses = []
for epoch in range(epochs):
	epoch_start_time = time.time()
	print(">> Epoch Number - {}".format(epoch+save_count))
	count = 0
	for a_batch in train_loader:
		print("Epoch {}, Batch {}".format(epoch, count))
		count += 1

		image = a_batch['image']
		image = image[:,None,:,:]
		image = image.float().to(device)
		
		# Output of Autoencoder
		reconstructed = model(image)
			
		# Calculating the loss function
		loss = loss_function(reconstructed, image)
	
		# The gradients are set to zero,
		# the the gradient is computed and stored.
		# .step() performs parameter update
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
	#Writing the loss function to a text file
	with open(RESULTS_PATH+loss_text, 'a') as f:
		f.writelines(str(loss.detach().cpu().numpy())+'\n')

	# Storing the losses in a list for plotting
	losses.append(loss)
	print('Train Loss: %.3f'%(loss))
	#Saving checkpoints
	torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': loss,
		}, MODEL_SAVE_PATH+CHECKPOINT_PATH+'epoch_'+str(epoch+save_count)+'.pth')

	outputs.append((epoch, image, reconstructed))
	epoch_end_time = time.time()
	print('>> Epoch time = {}'.format(epoch_end_time-epoch_start_time))

#Writing the losses to a text file
# losses = [str(i.detach().cpu().numpy())+'\n' for i in losses]
# with open(RESULTS_PATH+'all_losses.txt', 'a+') as f:
# 	f.writelines(losses)

#Plotting the reconstructed images
# for i, item in enumerate(reconstructed):
# 	plot_item = item.detach().cpu().numpy()
# 	plot_item = plot_item.reshape(-1, 28, 28)
# 	print(plot_item.shape)
# 	plt.imshow(plot_item[0], cmap='gray')
# 	plt.show()