import json
import os


def get_dataset_config(dataset_name):
    config_file = os.path.join(os.getcwd(), 'config', 'dataset_config.json')
    with open(config_file) as file:
        config_data = json.load(file)
        files_ = config_data['datasets'][dataset_name]
    return files_


def get_model_config(model_name, checkpoint_name='checkpoint_1'):
    config_file = os.path.join(os.getcwd(), 'config', 'models_config.json')
    with open(config_file) as file:
        config_data = json.load(file)
        files_ = config_data['models'][model_name]['checkpoints'][checkpoint_name]
    return files_['name'], files_['dim']

