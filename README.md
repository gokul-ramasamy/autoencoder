This code corresponds to training an autoencoder with chest x-rays

To start training the model from a saved checkpoint (change the file path as required)
```
python3 train.py --file_path "./model/2022-01-18 15:59/epoch_17.pth"
```
To start training the model from scratch, simply run the following
```
python3 train.py
```
To test the model by reconstructing the input, run the following (change the file path as required)
```
python3 test.py --file_path "./model/2022-01-18 15:59/epoch_17.pth"
```