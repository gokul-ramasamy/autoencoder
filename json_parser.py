import json

JSON_FILE_PATH = './params.json'

#Reading the JSON file
with open(JSON_FILE_PATH, 'r') as f:
    params_dict = json.load(f)

#Variables for datagen.py
TRAIN_TEXT = params_dict['train_text'] 
TEST_TEXT = params_dict['test_text']

TRAIN_BATCH_SIZE = params_dict['train_batch']
TEST_BATCH_SIZE = params_dict['test_batch']

INPUT_SIZE = (params_dict['input_size'], params_dict['input_size'])

#Variables for train.py
MODEL_SAVE_PATH = params_dict['model_save_path'] 
RESULTS_PATH = params_dict['results_path']

