import torch
import lightning as L
from model import CLIPVisionClassifier
from pytorch_lightning.loggers import WandbLogger
import json
import argparse
import os
from data import Caltech101DataModule

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
root_dir = os.path.abspath(os.path.join(dir_name, os.pardir))

# Trains model based on config dict
def train(config):
    # Fix random seed for reproducible results
    L.seed_everything(config['random_seed'], workers=True)

    # Create model and set it to use GPU
    model = CLIPVisionClassifier(config['hidden_sizes'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Get Dataloaders
    img_size = (config['image_height'], config['image_width'])
    data_module = Caltech101DataModule(
        data_dir=os.path.join(root_dir, 'dataset'),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=(config['image_height'], config['image_width']),
        split_props=config['split_props']
    )

    # Train model
    wandb_logger = WandbLogger(log_model='all')
    trainer = L.Trainer(
        logger=wandb_logger, 
        max_epochs=config['num_epochs'], 
        accelerator='gpu', 

    )
    trainer.fit(model, data_module)

    # Test model
    trainer.test(dataloaders=test_loader)

if __name__ == '__main__':
    # Command line argument parser that reads in config file name
    parser = argparse.ArgumentParser(description="Python script to train CLIPVisionClassifier")
    parser.add_argument('-c', '--config', help='name of config file (must be placed in config/ directory)')
    args = parser.parse_args()
    arg_dict = vars(args)
    config_filename = os.path.join(root_dir, 'config', arg_dict['config'] if arg_dict['config'] is not None else 'config.json')

    with open(config_filename) as config_file:
        config = json.load(config_file)

    train(config)

