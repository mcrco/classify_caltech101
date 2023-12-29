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

if __name__ == '__main__':
    # Parse config file option from command line
    parser = argparse.ArgumentParser(description="Python script to train CLIPVisionClassifier")
    parser.add_argument('-c', '--config', help='name of config file (must be placed in config/ directory)')
    args = parser.parse_args()
    arg_dict = vars(args)
    config_filename = arg_dict['config'] if arg_dict['config'] is not None else 'config.json'
    config_path = os.path.join(root_dir, 'config', config_filename)

    with open(config_path) as config_file:
        config = json.load(config_file)

    L.seed_everything(config['random_seed'], workers=True)

    data_module = Caltech101DataModule(
        data_dir=os.path.join(root_dir, 'data'),
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=(config['image_height'], config['image_width']),
        split_props=[config['train_prop'], config['test_prop'], config['val_prop']]
    )
    data_module.prepare_data()

    model = CLIPVisionClassifier(
        hidden_sizes=config['hidden_sizes'], 
        label_map=data_module.label_map,
        lr=config['learning_rate']
    )

    trainer = L.Trainer(
        logger=WandbLogger(log_model='all', name=config_filename[:-5]), 
        default_root_dir=os.path.join(root_dir, 'lightning_logs', config_filename[:-5]),
        max_epochs=config['num_epochs'],
        accelerator=config['accelerator'], 
        devices=config['devices'],
        deterministic=True,
    )

    trainer.fit(model, data_module)
    trainer.test(model, data_module)
