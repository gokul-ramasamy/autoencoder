from matplotlib import pyplot as plt
from datagen import ChestXDataset
from datagen import train_transformed_dataset, test_transformed_dataset
from datagen import train_dataloader, test_dataloader
from model import conv_AE


train_paths = list()
with open("./txt_files/cxp_train.txt", 'r') as f:
    data = f.readlines()
    for i in data:
        train_paths.append(i[:-1])

train_paths = train_paths[:1]
print(train_paths)

### TESTING BLOCK BEGIN: datagen.py ###
#Testing the Dataset Class
sample_dataset = ChestXDataset(data_paths=train_paths)
#Iterating through the samples to check the __len__ and __getitem__ methods
for i in range(len(sample_dataset)):
    sample = sample_dataset[i]
    print(sample['image'].shape)
    plt.imshow(sample['image'], cmap='gray')
    plt.show()
    break

#Testing the dataset transformations
sample_dataset = ChestXDataset(
                                data_paths=train_paths,
                                transform = transforms.Compose([Rescale_Norm((256,256)),
                                                                ToTensor(),
                                                                RandomHorizontalFlip(),
                                                                RandomVerticalFlip()
                                                                ])
)

exit()
plt.imshow(sample['image'], cmap='gray')
plt.show()

#Testing the final datasets
sample = transformed_dataset[0]["image"]
plt.imshow(sample, cmap='gray')
plt.show()

#Helper function to show a batch
for i_batch, sample_batched in enumerate(train_dataloader):
    print(i_batch, sample_batched['image'].shape)
    for j in range(sample_batched['image'].shape[0]):
        plt.imshow(sample_batched['image'][j], cmap='gray')
        plt.show()
    break
### TESTING BLOCK END: datagen.py ###

### TESTING BLOCK BEGIN: model.py ###
# Model Architecture testing
# use gpu if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Initializing the model
# Model Initialization
model = conv_AE().to(device)

print(summary(model, (1,256,256)))
### TESTING BLOCK END: model.py ###