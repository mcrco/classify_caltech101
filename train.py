import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import lightning as L
from model import CLIPVisionClassifier

# Fix random seed for reproducible results
L.seed_everything(420, workers=True)

# Create model and set it to use GPU
model = CLIPVisionClassifier([256, 256])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load data and split into train, test, val
data = datasets.Caltech101(root='data', download=True, transform=ToTensor())
train_data, test_data, val_data = random_split(data, [0.8, 0.1, 0.1])
train_loader = DataLoader(train_data, num_workers=15)
test_loader = DataLoader(test_data)
val_loader = DataLoader(val_data)

# Train model
trainer = L.Trainer()
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

# Test model
trainer.test(dataloaders=test_loader)
