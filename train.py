import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
from torchvision.transforms import Resize, ToTensor
import lightning as L
from model import CLIPVisionClassifier
from pytorch_lightning.loggers import WandbLogger
import numpy as np

# Fix random seed for reproducible results
L.seed_everything(420, workers=True)

# Create model and set it to use GPU
model = CLIPVisionClassifier([256, 128])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load data and turn into tensor
img_transform = transforms.Compose([
    Resize(size=(300,300)),
    ToTensor()
])
data = datasets.Caltech101(root='data', download=True, transform=img_transform)

# Filter out black and white photos
mask = np.ones(len(data), dtype=bool)
for i in range(len(data)):
    if data[i][0].shape[0] != 3:
        mask[i] = False
data = Subset(data, np.where(mask)[0])

# Split data into train test val
train_data, test_data, val_data = random_split(data, [0.8, 0.1, 0.1])
train_loader = DataLoader(train_data, batch_size=64, num_workers=2)
test_loader = DataLoader(test_data, batch_size=64, num_workers=2)
val_loader = DataLoader(val_data, batch_size=64, num_workers=2)

# Train model
wandb_logger = WandbLogger(log_model='all')
trainer = L.Trainer(logger=wandb_logger, max_epochs=5)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Test model
trainer.test(dataloaders=test_loader)
