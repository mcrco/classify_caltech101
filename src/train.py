import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor
import lightning as L
from model import CLIPVisionClassifier
from pytorch_lightning.loggers import WandbLogger
import numpy as np
import json
import argparse
import os

abspath = os.path.abspath(__file__)
dir_name = os.path.dirname(abspath)
root_dir = os.path.abspath(os.path.join(dir_name, os.pardir))

# Returns preprocessed data ready for dataloaders
def get_filtered_data(img_size):
    # Resize all images to specified size and convert from jpg to torch tensors
    img_transform = transforms.Compose([
        Resize(size=img_size),
        ToTensor()
    ])
    data = datasets.Caltech101(root=os.path.join(root_dir, 'data'), download=True, transform=img_transform)

    # Filter out black and white photos
    mask = np.ones(len(data), dtype=bool)
    for i in range(len(data)):
        if data[i][0].shape[0] != 3:
            mask[i] = False
    return Subset(data, np.where(mask)[0])

# Splits dataset into train, test, val based on split_props and returns dataloaders
def get_dataloaders(dataset, split_props):
    train_data, test_data, val_data = random_split(dataset, split_props)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], num_workers=2)
    test_loader = DataLoader(test_data, batch_size=config['batch_size'], num_workers=2)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], num_workers=2)
    return train_loader, test_loader, val_loader

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
    data = get_filtered_data(img_size)
    split_props = [config['train_prop'], config['test_prop'], config['val_prop']]
    train_loader, test_loader, val_loader = get_dataloaders(data, split_props)

    # Train model
    wandb_logger = WandbLogger(log_model='all')
    trainer = L.Trainer(logger=wandb_logger, max_epochs=config['num_epochs'])
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

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

